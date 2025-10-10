#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STL 参数化拟合（球/圆柱/长方体）→ 导出 JSON → 基于 JSON 参数修改 & 可视化

依赖：
  - numpy
  - trimesh

（可选可视化更好看）
  - pyglet 或 pyopengl 作为 trimesh 后端二选一

安装示例：
  pip install numpy trimesh pyglet

用法：
  1) 从 STL 拟合并导出 JSON：
     python stl_param_fit.py fit --stl input.stl --json params.json --samples 200000

  2) 从 JSON 修改参数并生成新网格（并可视化）：
     python stl_param_fit.py gen --json params.json --set radius=2.5 height=8.0 \
                                  --out new.stl --preview

  3) 直接可视化：
     python stl_param_fit.py preview --stl input.stl --json params.json

说明：
  - 本脚本不依赖 OCC/OpenCascade；仅用 numpy+trimesh 做解析与可视化。
  - 自动在球/圆柱/长方体三种模型中择优（按残差最小）。
  - 长方体是用 PCA 得到的有向包围盒（OBB）。
  - 圆柱轴向与高度由 PCA 估计，半径由到轴距离的中位数估计。
  - 球参数通过线性最小二乘解析求解。
"""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import numpy as np
import trimesh

# ----------------------- 工具函数 -----------------------

def load_points_from_stl(stl_path: str, n_samples: int = 200_000) -> Tuple[np.ndarray, trimesh.Trimesh]:
    """读取 STL 并从表面均匀采样点.
    返回 (N,3) 点云 和 原始 mesh 对象
    """
    mesh = trimesh.load(stl_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # 有些 STL 可能是 Scene，做个合并
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(geom for geom in mesh.dump().geometry.values()))
        else:
            raise TypeError("Unsupported STL content: not a mesh or scene")
    if n_samples <= 0:
        pts = mesh.vertices.copy()
    else:
        pts = mesh.sample(n_samples)
    return pts, mesh


def pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对点云做 PCA，返回 (centroid, R)；R 的行向量是主轴 (x',y',z')."""
    c = points.mean(axis=0)
    X = points - c
    # SVD on covariance
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt  # rows are principal directions
    # 统一右手系（det>0）
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return c, R

# ----------------------- 形状拟合 -----------------------

@dataclass
class SphereParam:
    center: Tuple[float, float, float]
    radius: float

@dataclass
class CylinderParam:
    center: Tuple[float, float, float]  # 轴线中点
    axis: Tuple[float, float, float]    # 单位向量
    radius: float
    height: float

@dataclass
class BoxParam:
    center: Tuple[float, float, float]
    R: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]  # 3x3 旋转矩阵（行向量为轴）
    extents: Tuple[float, float, float]  # 沿主轴长度 (lx, ly, lz)


def fit_sphere(points: np.ndarray) -> Tuple[SphereParam, float]:
    """解析法拟合球: x^2+y^2+z^2 + ax + by + cz + d = 0"""
    X = points
    A = np.c_[2*X[:, 0], 2*X[:, 1], 2*X[:, 2], np.ones(len(X))]
    b = (X**2).sum(axis=1)
    # 最小二乘解
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, cz, d = sol
    center = np.array([cx, cy, cz])
    r = math.sqrt((center @ center) + d)
    # 残差：|dist - r|
    dist = np.linalg.norm(X - center, axis=1)
    residual = np.abs(dist - r).mean()
    return SphereParam(tuple(center.tolist()), float(r)), float(residual)


def fit_cylinder(points: np.ndarray) -> Tuple[CylinderParam, float]:
    """用 PCA 估计圆柱轴，再估半径与高度。返回平均径向残差。"""
    c, R = pca_axes(points)
    axis = R[0]  # 主方向认为是轴向
    axis = axis / np.linalg.norm(axis)
    X = points - c
    # 轴向投影标量
    t = X @ axis
    tmin, tmax = float(t.min()), float(t.max())
    height = tmax - tmin
    # 轴线中点
    centerline_mid = c + axis * ((tmin + tmax) * 0.5)
    # 计算到轴的径向距离
    # 点到轴距离：||X - (X·axis)axis||
    radial = np.linalg.norm(X - np.outer(t, axis), axis=1)
    radius = float(np.median(radial))
    residual = float(np.abs(radial - radius).mean())
    return CylinderParam(tuple(centerline_mid.tolist()), tuple(axis.tolist()), radius, float(height)), residual


def fit_box(points: np.ndarray) -> Tuple[BoxParam, float]:
    """PCA 得到 OBB：中心 c，旋转 R，沿主轴投影的范围 extents。残差=点到最近面的法向距离均值。"""
    c, R = pca_axes(points)
    X = (points - c) @ R.T  # 转到盒坐标
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ext = maxs - mins
    # 最近面距离（法向方向）：min_i | |xi| - ext_i/2 |
    half = ext * 0.5
    absx = np.abs(X)
    d_face = np.min(np.abs(absx - half), axis=1)
    residual = float(d_face.mean())
    param = BoxParam(tuple(c.tolist()), tuple(map(tuple, R)), tuple(ext.tolist()))
    return param, residual

# ----------------------- 模型选择与 JSON -----------------------

def choose_best_model(points: np.ndarray) -> Dict[str, Any]:
    sp, r_s = fit_sphere(points)
    cy, r_c = fit_cylinder(points)
    bx, r_b = fit_box(points)

    # 用数据尺度归一化后比较更稳健
    scale = np.linalg.norm(points.std(axis=0)) + 1e-9
    scores = {
        'sphere': r_s / scale,
        'cylinder': r_c / scale,
        'box': r_b / scale,
    }
    best = min(scores, key=scores.get)

    if best == 'sphere':
        payload = {
            'type': 'sphere',
            'params': asdict(sp),
            'fit': {'score': scores['sphere']}
        }
    elif best == 'cylinder':
        payload = {
            'type': 'cylinder',
            'params': asdict(cy),
            'fit': {'score': scores['cylinder']}
        }
    else:
        payload = {
            'type': 'box',
            'params': asdict(bx),
            'fit': {'score': scores['box']}
        }
    return payload

# ----------------------- 基于参数生成网格 -----------------------

def mesh_from_params(spec: Dict[str, Any]) -> trimesh.Trimesh:
    t = spec['type']
    p = spec['params']
    if t == 'sphere':
        center = np.array(p['center'])
        radius = float(p['radius'])
        m = trimesh.creation.icosphere(subdivisions=4, radius=radius)
        m.apply_translation(center)
        return m

    if t == 'cylinder':
        center = np.array(p['center'])
        axis = np.array(p['axis'], dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        radius = float(p['radius'])
        height = float(p['height'])
        m = trimesh.creation.cylinder(radius=radius, height=height, sections=128)
        # 对齐 z 轴到目标轴
        R = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), axis)
        T = np.eye(4)
        T[:3, :3] = R
        m.apply_transform(T)
        m.apply_translation(center)
        return m

    if t == 'box':
        center = np.array(p['center'])
        R = np.array(p['R'], dtype=float)
        ext = np.array(p['extents'], dtype=float)
        m = trimesh.creation.box(extents=ext)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center
        m.apply_transform(T)
        return m

    raise ValueError(f"Unsupported type: {t}")

# ----------------------- 预览 -----------------------

def _bbox_lines_o3d_from_trimesh(mesh: trimesh.Trimesh):
    import open3d as o3d
    mn, mx = mesh.bounds
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=float)
    lines = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile([[0.2,0.8,0.2]], (len(lines),1)))
    return ls

def preview_scene(original: trimesh.Trimesh | None, fitted: trimesh.Trimesh | None, **kwargs):
    """强制用 Open3D 可视化：原始网格用绿色 AABB，拟合网格用实体 + 蓝色"""
    import open3d as o3d
    geoms = []
    if original is not None:
        # 原始网格用线框 AABB 做参照（避免加载两份大网格）
        geoms.append(_bbox_lines_o3d_from_trimesh(original))
    if fitted is not None:
        m = _trimesh_to_o3d(fitted)
        m.paint_uniform_color([0.2, 0.45, 1.0])
        geoms.append(m)
    # 坐标轴
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    if geoms:
        o3d.visualization.draw_geometries(geoms)

def _o3d_gui_modules():
    """
    Return (o3d, gui, rendering) with compatibility for builds where `o3d.gui`
    doesn't exist but `o3d.visualization.gui` does.
    """
    import open3d as o3d
    gui = getattr(o3d, "gui", None)
    rendering = None

    # Normal path (most wheels)
    if gui is not None:
        rendering = o3d.visualization.rendering if hasattr(o3d, "visualization") else None
    else:
        # Fallback: some builds expose GUI under o3d.visualization.gui
        vis = getattr(o3d, "visualization", None)
        if vis is not None:
            gui = getattr(vis, "gui", None)
            rendering = getattr(vis, "rendering", None)

    if gui is None or rendering is None:
        raise ImportError(
            "This Open3D build has no GUI backend (neither o3d.gui nor o3d.visualization.gui). "
            "Install a wheel with GUI support, or use `gen --set ... --preview` for non-GUI flow."
        )
    return o3d, gui, rendering

# ----------------------- CLI -----------------------

def parse_kv_updates(kvs):
    """将 --set key=val 解析成 dict，val 支持 float/三元组/列表。"""
    out = {}
    for item in kvs or []:
        if '=' not in item:
            raise ValueError(f"Bad --set item: {item}")
        k, v = item.split('=', 1)
        v = v.strip()
        try:
            # 尝试解析为 JSON（可写 [1,2,3] 或 {..} 或 1.23）
            parsed = json.loads(v)
        except Exception:
            try:
                parsed = float(v)
            except Exception:
                parsed = v
        out[k] = parsed
    return out


def cmd_fit(args):
    pts, mesh = load_points_from_stl(args.stl, n_samples=args.samples)
    spec = choose_best_model(pts)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        print(f"Saved params → {args.json}")
    if args.preview:
        fitted = mesh_from_params(spec)
        preview_scene(mesh, fitted, smooth=not getattr(args, 'flat', False))


def cmd_gen(args):
    with open(args.json, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    # 应用更新
    updates = parse_kv_updates(args.set)
    # 支持 params 下的直接键名
    for k, v in updates.items():
        if k in spec.get('params', {}):
            spec['params'][k] = v
        else:
            # 允许修改顶层 type
            if k == 'type':
                spec['type'] = v
            else:
                # 允许一次性替换整个 params
                if k == 'params' and isinstance(v, dict):
                    spec['params'] = v
                else:
                    print(f"[WARN] Unknown key '{k}', skipped.")
    mesh = mesh_from_params(spec)
    if args.out:
        mesh.export(args.out)
        print(f"Saved mesh → {args.out}")
    if args.preview:
        preview_scene(None, mesh, smooth=not getattr(args, 'flat', False))


def cmd_preview(args):
    orig_mesh = None
    spec = None
    if args.stl:
        _, orig_mesh = load_points_from_stl(args.stl, n_samples=20000)
    if args.json:
        with open(args.json, 'r', encoding='utf-8') as f:
            spec = json.load(f)
    fitted = mesh_from_params(spec) if spec else None
    preview_scene(orig_mesh, fitted, smooth=not getattr(args, 'flat', False))


# ----------------------- Open3D 交互 GUI（滑块调参） -----------------------

def _trimesh_to_o3d(mesh: trimesh.Trimesh):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(float))
    m.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    m.compute_vertex_normals()
    return m


def _make_axes(size: float = 1.0):
    import open3d as o3d
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def _bbox_lines_o3d(mesh: trimesh.Trimesh):
    import open3d as o3d
    aabb = mesh.bounds  # (min, max)
    mn, mx = aabb
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=float)
    lines = np.array([
        [0,1],[1,2],[2,3],[3,0], [4,5],[5,6],[6,7],[7,4], [0,4],[1,5],[2,6],[3,7]
    ], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    cols = np.tile([[0.2,0.8,0.2]], (len(lines),1))
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


def _spec_to_trimesh(spec: Dict[str, Any]) -> trimesh.Trimesh:
    return mesh_from_params(spec)


def _update_param(spec: Dict[str, Any], key: str, value):
    if key in spec.get('params', {}):
        spec['params'][key] = value


def _euler_to_Rxyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return (Rz @ Ry @ Rx)


def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n>1e-12 else v


def cmd_gui(args):
    o3d, gui, rendering = _o3d_gui_modules()

    # 加载参数
    with open(args.json, 'r', encoding='utf-8') as f:
        spec = json.load(f)

    orig_mesh = None
    if args.stl:
        _, orig_mesh = load_points_from_stl(args.stl, n_samples=min(args.samples, 50000))

    # 初始 mesh
    tri = _spec_to_trimesh(spec)

    # GUI 初始化
    app = gui.Application.instance
    app.initialize()

    w = gui.Application.instance.create_window("Param GUI", 1280, 800)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    scene.scene.set_background([1,1,1,1])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"

    # 添加几何体
    if orig_mesh is not None:
        o3 = _trimesh_to_o3d(orig_mesh)
        scene.scene.add_geometry("orig", o3, mat)
        scene.scene.show_geometry("orig", True)

    tri_o3 = _trimesh_to_o3d(tri)
    scene.scene.add_geometry("fit", tri_o3, mat)

    # 坐标轴与 bbox
    size_hint = float(np.linalg.norm(tri.bounding_box.extents) if hasattr(tri, 'bounding_box') else 1.0)
    axes = _make_axes(max(size_hint*0.25, 1e-3))
    scene.scene.add_geometry("axes", axes, mat)

    bbox_lines = _bbox_lines_o3d(tri)
    scene.scene.add_geometry("bbox", bbox_lines, mat)

    # 相机
    bounds = scene.scene.bounding_box
    scene.setup_camera(60.0, bounds, bounds.get_center())

    # 右侧面板（控件）
    panel = gui.Vert(0, gui.Margins(8,8,8,8))
    info = gui.Label("")

    def refresh_geometry():
        nonlocal tri_o3, bbox_lines
        # 重建拟合网格
        new_tri = _spec_to_trimesh(spec)
        new_o3 = _trimesh_to_o3d(new_tri)
        scene.scene.remove_geometry("fit")
        scene.scene.add_geometry("fit", new_o3, mat)
        tri_o3 = new_o3
        # bbox 更新
        scene.scene.remove_geometry("bbox")
        bbox_lines = _bbox_lines_o3d(new_tri)
        scene.scene.add_geometry("bbox", bbox_lines, mat)
        # 信息文本
        bb = new_tri.bounds
        ext = bb[1]-bb[0]
        info.text = f"AABB: {ext[0]:.3f} × {ext[1]:.3f} × {ext[2]:.3f}  |  type: {spec['type']}"

    # 根据类型创建滑块
    sliders = []
    def add_slider(name, vmin, vmax, vinit, on_change):
        lbl = gui.Label(name)
        sld = gui.Slider(gui.Slider.DOUBLE)
        sld.set_limits(float(vmin), float(vmax))
        sld.double_value = float(vinit)
        def _cb(val):
            on_change(float(val))
            refresh_geometry()
        sld.set_on_value_changed(_cb)
        panel.add_child(lbl)
        panel.add_child(sld)
        sliders.append(sld)
        return sld

    t = spec['type']
    p = spec['params']

    # 公共：中心
    cx, cy, cz = p.get('center', [0,0,0])
    span = max(1.0, np.max(np.abs([cx,cy,cz]))*3 + 1.0)
    add_slider('center.x', cx-span, cx+span, cx, lambda v: _update_param(spec,'center',[v, spec['params']['center'][1], spec['params']['center'][2]]))
    add_slider('center.y', cy-span, cy+span, cy, lambda v: _update_param(spec,'center',[spec['params']['center'][0], v, spec['params']['center'][2]]))
    add_slider('center.z', cz-span, cz+span, cz, lambda v: _update_param(spec,'center',[spec['params']['center'][0], spec['params']['center'][1], v]))

    if t == 'sphere':
        r = float(p['radius'])
        add_slider('radius', max(r*0.1, 1e-4), r*3+1.0, r, lambda v: _update_param(spec,'radius', v))

    elif t == 'cylinder':
        r = float(p['radius']); h = float(p['height'])
        ax = _normalize(p['axis'])
        add_slider('radius', max(r*0.1, 1e-4), r*3+1.0, r, lambda v: _update_param(spec,'radius', v))
        add_slider('height', max(h*0.1, 1e-4), h*3+1.0, h, lambda v: _update_param(spec,'height', v))
        ang_init = [0.0, 0.0, 0.0]
        def set_axis_from_euler(rx_deg=None, ry_deg=None, rz_deg=None):
            nonlocal ang_init
            if rx_deg is not None: ang_init[0] = rx_deg
            if ry_deg is not None: ang_init[1] = ry_deg
            if rz_deg is not None: ang_init[2] = rz_deg
            Rx, Ry, Rz = [math.radians(a) for a in ang_init]
            Rm = _euler_to_Rxyz(Rx, Ry, Rz)
            new_axis = (Rm @ np.array([0,0,1.0])).tolist()
            _update_param(spec,'axis', _normalize(new_axis).tolist())
        add_slider('axis.rx(deg)', -180, 180, 0.0, lambda v: set_axis_from_euler(rx_deg=v))
        add_slider('axis.ry(deg)', -180, 180, 0.0, lambda v: set_axis_from_euler(ry_deg=v))
        add_slider('axis.rz(deg)', -180, 180, 0.0, lambda v: set_axis_from_euler(rz_deg=v))

    elif t == 'box':
        ex, ey, ez = [float(x) for x in p['extents']]
        add_slider('extent.x', max(ex*0.1,1e-4), ex*3+1.0, ex, lambda v: _update_param(spec,'extents',[v, spec['params']['extents'][1], spec['params']['extents'][2]]))
        add_slider('extent.y', max(ey*0.1,1e-4), ey*3+1.0, ey, lambda v: _update_param(spec,'extents',[spec['params']['extents'][0], v, spec['params']['extents'][2]]))
        add_slider('extent.z', max(ez*0.1,1e-4), ez*3+1.0, ez, lambda v: _update_param(spec,'extents',[spec['params']['extents'][0], spec['params']['extents'][1], v]))
        def set_R_from_euler(rx_deg=None, ry_deg=None, rz_deg=None):
            rx = getattr(set_R_from_euler, 'rx', 0.0)
            ry = getattr(set_R_from_euler, 'ry', 0.0)
            rz = getattr(set_R_from_euler, 'rz', 0.0)
            if rx_deg is not None: rx = rx_deg
            if ry_deg is not None: ry = ry_deg
            if rz_deg is not None: rz = rz_deg
            setattr(set_R_from_euler,'rx', rx)
            setattr(set_R_from_euler,'ry', ry)
            setattr(set_R_from_euler,'rz', rz)
            Rm = _euler_to_Rxyz(math.radians(rx), math.radians(ry), math.radians(rz))
            spec['params']['R'] = tuple(map(tuple, Rm.astype(float)))
        add_slider('rot.rx(deg)', -180, 180, 0.0, lambda v: set_R_from_euler(rx_deg=v))
        add_slider('rot.ry(deg)', -180, 180, 0.0, lambda v: set_R_from_euler(ry_deg=v))
        add_slider('rot.rz(deg)', -180, 180, 0.0, lambda v: set_R_from_euler(rz_deg=v))

    panel.add_child(gui.Label(""))
    panel.add_child(info)

    # 布局
    horiz = gui.Horiz()
    horiz.add_child(scene, stretch=4)
    scroll = gui.ScrollView(panel)
    horiz.add_child(scroll, stretch=1)

    w.add_child(horiz)
    refresh_geometry()

    app.run()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='STL 参数化拟合与生成 (sphere/cylinder/box)')
    sub = ap.add_subparsers(required=True)

    p_fit = sub.add_parser('fit', help='从 STL 拟合并导出 JSON 参数')
    p_fit.add_argument('--stl', default="models/sphere.STL", required=True)
    p_fit.add_argument('--json', default="jsons/sphere.json", required=True)
    p_fit.add_argument('--samples', type=int, default=200000)
    p_fit.add_argument('--preview', action='store_true')
    p_fit.add_argument('--flat', action='store_true', help='禁用平滑着色（避免依赖 networkx/scipy）')
    p_fit.set_defaults(func=cmd_fit)

    p_gen = sub.add_parser('gen', help='从 JSON 生成新网格，支持用 --set 修改参数')
    p_gen.add_argument('--json', default="jsons/sphere.json", required=True)
    p_gen.add_argument('--set', nargs='*', default=[], help='形如 key=val，例如 radius=2.5 或 extents=[2,3,4]')
    p_gen.add_argument('--out', default="outs/sphere.stl", help='输出新 STL/OBJ 路径，后缀决定格式')
    p_gen.add_argument('--preview', action='store_true')
    p_gen.add_argument('--flat', action='store_true', help='禁用平滑着色（避免依赖 networkx/scipy）')
    p_gen.set_defaults(func=cmd_gen)

    p_prev = sub.add_parser('preview', help='可视化原始 STL 与 JSON 拟合模型')
    p_prev.add_argument('--stl', default="models/sphere.STL")
    p_prev.add_argument('--json', default="jsons/sphere.json")
    p_prev.add_argument('--flat', action='store_true', help='禁用平滑着色（避免依赖 networkx/scipy）')
    p_prev.set_defaults(func=cmd_preview)

    # 交互 GUI 子命令（Open3D）
    p_gui = sub.add_parser('gui', help='Open3D 交互式预览，带滑块调参与坐标轴/尺寸标注')
    p_gui.add_argument('--json', default="jsons/sphere.json", help='参数 JSON 文件')
    p_gui.add_argument('--stl', default="models/sphere.STL", help='可选：原始 STL（作为参考线框或实体）')
    p_gui.add_argument('--samples', type=int, default=80000, help='若传入 --stl 用于抽样显示')
    p_gui.set_defaults(func=cmd_gui)

    args = ap.parse_args()
    args.func(args)
