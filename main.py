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
def preview_scene(original: trimesh.Trimesh | None, fitted: trimesh.Trimesh | None, **kwargs):
    """使用 Open3D 可视化：原始网格用线框，拟合网格用实体。"""
    import open3d as o3d
    geoms = []
    if original is not None:
        geoms.append(_trimesh_wireframe_o3d(original))  # 原始：线框
    if fitted is not None:
        m = _trimesh_to_o3d(fitted)                     # 拟合：实体
        m.paint_uniform_color([0.2, 0.45, 1.0])         # 蓝色
        geoms.append(m)
    if geoms:
        o3d.visualization.draw_geometries(geoms)

def _trimesh_to_o3d(mesh: trimesh.Trimesh):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(float))
    m.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    m.compute_vertex_normals()
    return m

def _trimesh_wireframe_o3d(mesh: trimesh.Trimesh):
    import open3d as o3d
    V = mesh.vertices.astype(float)
    F = mesh.faces.astype(np.int32)
    # 收集三角形的三条边并去重
    edges = set()
    for i0, i1, i2 in F:
        a, b = sorted((int(i0), int(i1))); edges.add((a, b))
        a, b = sorted((int(i1), int(i2))); edges.add((a, b))
        a, b = sorted((int(i2), int(i0))); edges.add((a, b))
    edges = np.array(sorted(edges), dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(V)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile([[0.6, 0.6, 0.6]], (len(edges), 1)))  # 灰色线框
    return ls

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


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='STL 参数化拟合与生成 (sphere/cylinder/box)')
    sub = ap.add_subparsers(required=True)

    p_fit = sub.add_parser('fit', help='从 STL 拟合并导出 JSON 参数')
    p_fit.add_argument('--stl', required=True)
    p_fit.add_argument('--json', required=True)
    p_fit.add_argument('--samples', type=int, default=200000)
    p_fit.add_argument('--preview', action='store_true')
    p_fit.add_argument('--flat', action='store_true', help='禁用平滑着色（避免依赖 networkx/scipy）')
    p_fit.set_defaults(func=cmd_fit)

    p_gen = sub.add_parser('gen', help='从 JSON 生成新网格，支持用 --set 修改参数')
    p_gen.add_argument('--json', required=True)
    p_gen.add_argument('--set', nargs='*', default=[], help='形如 key=val，例如 radius=2.5 或 extents=[2,3,4]')
    p_gen.add_argument('--out', default=None, help='输出新 STL/OBJ 路径，后缀决定格式')
    p_gen.add_argument('--preview', action='store_true')
    p_gen.add_argument('--flat', action='store_true', help='禁用平滑着色（避免依赖 networkx/scipy）')
    p_gen.set_defaults(func=cmd_gen)

    p_prev = sub.add_parser('preview', help='可视化原始 STL 与 JSON 拟合模型')
    p_prev.add_argument('--stl', default=None)
    p_prev.add_argument('--json', default=None)
    p_prev.add_argument('--flat', action='store_true', help='禁用平滑着色（避免依赖 networkx/scipy）')
    p_prev.set_defaults(func=cmd_preview)

    args = ap.parse_args()
    args.func(args)
