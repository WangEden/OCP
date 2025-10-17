#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STL 参数化拟合（球/圆柱/长方体）→ 导出 JSON → 基于 JSON 参数修改 & 可视化

依赖：
  - numpy
  - trimesh
  - open3d

安装示例：
  pip install numpy trimesh pyglet open3d

用法：
  1) 从 STL 拟合并导出 JSON：
     python main.py fit --stl input.stl --json params.json --samples 200000
     或 python main.py fit

  2) 从 JSON 修改参数并生成新网格（并可视化）：
     python main.py gen --json params.json --set radius=2.5 height=8.0 \
                                  --out new.stl --preview
     或 python main.py gen

  3) 直接可视化：
     python main.py preview --stl input.stl --json params.json

  4) GUI 交互调参（需要 Open3D 支持 GUI 的版本）：
     python main.py gui --stl input.stl --json params.json --samples 50000
     或 python main.py gui

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
from typing import Dict, Any, Tuple, List
from waverider import Conical_flow, streamline

import numpy as np
import trimesh

# ----------------------- 工具函数 -----------------------
def read_plot3d_xyz(path: str, order: str = "xzy"):
    """
    读取简单的 ASCII Plot3D 多块文件。
    参数 order 指明写入顺序：大多数是 'xyz'，但 waverider.py 写的是 'xzy'。
    返回 [(X, Y, Z), ...]；每个是 (nk, nj, ni) 的 ndarray（一般 nk=1 表面）
    """
    import numpy as np
    with open(path, 'r') as f:
        toks = f.read().split()
    it = iter(toks)
    nblk = int(next(it))
    dims = []
    for _ in range(nblk):
        ni, nj, nk = int(next(it)), int(next(it)), int(next(it))
        dims.append((ni, nj, nk))

    blocks = []
    for (ni, nj, nk) in dims:
        size = ni * nj * nk
        A = np.fromiter((float(next(it)) for _ in range(size)), float, count=size).reshape((nk, nj, ni))
        B = np.fromiter((float(next(it)) for _ in range(size)), float, count=size).reshape((nk, nj, ni))
        C = np.fromiter((float(next(it)) for _ in range(size)), float, count=size).reshape((nk, nj, ni))
        if order.lower() == "xyz":
            X, Y, Z = A, B, C
        elif order.lower() == "xzy":
            X, Y, Z = A, C, B   # waverider.py 用的就是这个顺序
        else:
            raise ValueError("Unknown order, expect 'xyz' or 'xzy'")
        blocks.append((X, Y, Z))
    return blocks


def plot3d_blocks_to_trimesh(blocks):
    """
    把 Plot3D 结构网格曲面（三维阵列 nk×nj×ni，通常 nk=1）转成 trimesh 三角网格。
    支持多块：合并成一体。
    """
    import numpy as np, trimesh
    all_vertices = []
    all_faces = []
    v_offset = 0
    for (X, Y, Z) in blocks:
        X, Y, Z = X[0], Y[0], Z[0]   # 取 nk=1 的曲面层 → (nj, ni)
        nj, ni = X.shape
        V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # (nj*ni, 3)
        all_vertices.append(V)

        # 逐网格单元剖分为两三角（i: 0..ni-2, j: 0..nj-2）
        ii, jj = np.meshgrid(np.arange(ni-1), np.arange(nj-1))
        ii = ii.ravel(); jj = jj.ravel()
        # 顶点索引（行主序）
        def vid(j, i): return j*ni + i
        f1 = np.stack([vid(jj, ii), vid(jj, ii+1), vid(jj+1, ii+1)], axis=1)
        f2 = np.stack([vid(jj, ii), vid(jj+1, ii+1), vid(jj+1, ii)], axis=1)
        F = np.vstack([f1, f2]).astype(np.int64) + v_offset
        all_faces.append(F)
        v_offset += V.shape[0]

    V_all = np.vstack(all_vertices)
    F_all = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int64)
    return trimesh.Trimesh(vertices=V_all, faces=F_all, process=False)


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


def load_mesh_as_trimesh(path: str) -> trimesh.Trimesh:
    """
    把各种可能的 STL 载入结果（Scene / list / Trimesh）统一成单个 Trimesh。
    """
    m = trimesh.load(path, force='mesh')
    # 1) 如果是 Scene
    if hasattr(m, 'geometry'):
        geoms = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RuntimeError(f"No mesh geometry found in scene: {path}")
        return trimesh.util.concatenate(geoms)

    # 2) 如果是 list[Trimesh]
    if isinstance(m, (list, tuple)):
        geoms = [g for g in m if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RuntimeError(f"No trimesh in list for: {path}")
        return trimesh.util.concatenate(geoms)

    # 3) 已经是 Trimesh
    if isinstance(m, trimesh.Trimesh):
        return m

    # 4) 防御：如果是 numpy 数组，无法直接用 —— 明确报错
    if isinstance(m, np.ndarray):
        raise TypeError("Loaded object is numpy.ndarray, expected a triangulated mesh. "
                        "Please provide an STL with faces.")

    # 兜底
    raise TypeError(f"Unsupported mesh type: {type(m)} from {path}")


def _split_upper_lower_from_stl(mesh: trimesh.Trimesh, ny_thresh=0.98):
    """
    通过法向把一个 Trimesh 分为上/下表面。
    传入的 mesh 必须是 Trimesh（如果不是，请先用 load_mesh_as_trimesh）。
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"_split_upper_lower_from_stl expects Trimesh, got {type(mesh)}")

    m = mesh.copy()
    m.rezero()
    fn = m.face_normals  # (F,3)
    # 上表面：法向的 y 分量绝对值很大（近似平行 ±Y）
    upper_mask = np.abs(fn[:, 1]) >= ny_thresh

    faces_up = np.nonzero(upper_mask)[0]
    faces_lo = np.nonzero(~upper_mask)[0]

    if len(faces_up) == 0 or len(faces_lo) == 0:
        # 放宽阈值再试一次
        upper_mask = np.abs(fn[:, 1]) >= 0.95
        faces_up = np.nonzero(upper_mask)[0]
        faces_lo = np.nonzero(~upper_mask)[0]
        if len(faces_up) == 0 or len(faces_lo) == 0:
            raise RuntimeError("Failed to split upper/lower surfaces by normals; "
                               "try cleaning normals or adjust --ny-thresh.")

    up = m.submesh([faces_up], append=True)
    lo = m.submesh([faces_lo], append=True)

    # 如果是多连通，取面积最大的一块
    if isinstance(up, list):
        up = max(up, key=lambda g: g.area)
    if isinstance(lo, list):
        lo = max(lo, key=lambda g: g.area)

    return up, lo


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

@dataclass
class WaveriderParams:
    Ma: float = 6.0
    shock_angle_deg: float = 30.0
    n_arcs: int = 3
    phi_arc_deg: List[float] = None
    R_arc: List[float] = None
    zu: List[float] = None
    yu: List[float] = None
    n_stream: int = 600
    n_out: int = 800
    scale: float = 1.0
    translate_x: float = 0.0
    translate_y: float = 0.0
    translate_z: float = 0.0

def save_waverider_params_json(p: WaveriderParams, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(p), f, ensure_ascii=False, indent=2)

def load_waverider_params_json(path: str) -> WaveriderParams:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return WaveriderParams(**d)

def _split_upper_lower_from_stl(mesh: trimesh.Trimesh, ny_thresh=0.98) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    通过法向分上/下表面：上表面接近 y 常量，法向 ~ ±ŷ（|ny| 大）。
    ny_thresh 默认 0.98，必要时可调到 0.95。
    """
    mesh = mesh.copy()
    mesh.rezero()
    fn = mesh.face_normals  # (F,3)
    upper_mask = np.abs(fn[:,1]) >= ny_thresh
    # 取连通分量里最大的那一块，避免碎片
    up = mesh.submesh([np.nonzero(upper_mask)[0]], append=True)
    lo = mesh.submesh([np.nonzero(~upper_mask)[0]], append=True)
    # 如果有多个组件，保留面积最大的
    if isinstance(up, list): up = max(up, key=lambda m: m.area)
    if isinstance(lo, list): lo = max(lo, key=lambda m: m.area)
    return up, lo

def _extract_upline_from_upper(upper: trimesh.Trimesh, tol=1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    上表面是 y,z 常量的条带：顶点的 (y,z) 成离散集合。
    我们把 (y,z) 做网格化/去重，按 z 排序得到 (zu, yu)。
    """
    if not isinstance(upper, trimesh.Trimesh):
        raise TypeError(f"_extract_upline_from_upper expects Trimesh, got {type(upper)}")
    upper = ensure_trimesh(upper)
    V = upper.vertices  # (N,3) → x,y,z
    yz = V[:,[1,2]]
    # 归一化后四舍五入聚类
    scale = max(upper.extents) if upper.extents.max() > 0 else 1.0
    q = np.round(yz / (tol*scale), 0)
    # 唯一键
    uniq, idx = np.unique(q, axis=0, return_index=True)
    sel = yz[idx]
    # 某些重复的 z 值可能对应不同 y（数值噪声），按 z 聚成条，只取众数 y
    z_vals = sel[:,1]
    order = np.argsort(z_vals)
    sel = sel[order]
    # 对相同 z（±tol*scale）求 y 的中位数
    z_out, y_out = [], []
    cur = []
    for y,z in sel:
        if (len(cur)==0) or (abs(z - cur[-1][1]) <= tol*scale):
            cur.append((y,z))
        else:
            arr = np.array(cur); z_out.append(np.median(arr[:,1])); y_out.append(np.median(arr[:,0])); cur=[(y,z)]
    if cur:
        arr = np.array(cur); z_out.append(np.median(arr[:,1])); y_out.append(np.median(arr[:,0]))
    zu = np.array(z_out, float)
    yu = np.array(y_out, float)
    # 去重并按 z 升序
    ord2 = np.argsort(zu); zu = zu[ord2]; yu = yu[ord2]
    return zu, yu

def _extract_wavecurve_from_lower(lower: trimesh.Trimesh, k_percent=5, bins=200):
    """
    近似提取 (zw,yw)：在 x 最靠近鼻尖的一端（x 的最小若干百分位）投影到 (z,y) 平面，
    再按 z 分箱取最大 y，得到一条“外轮廓”曲线（和 waverider 的波接触线一致性相当好）。
    """
    lower = ensure_trimesh(lower)
    V = lower.vertices
    x = V[:,0]; y = V[:,1]; z = V[:,2]
    x_cut = np.percentile(x, k_percent)  # 取最靠前的若干点
    mask = x <= x_cut
    y2 = y[mask]; z2 = z[mask]
    # 按 z 分箱
    zmin, zmax = z2.min(), z2.max()
    edges = np.linspace(zmin, zmax, bins+1)
    inds = np.digitize(z2, edges) - 1
    zw = []; yw = []
    for i in range(bins):
        m = inds == i
        if m.any():
            zw.append(0.5*(edges[i]+edges[i+1]))
            yw.append(np.max(y2[m]))
    return np.array(zw,float), np.array(yw,float)

def _fit_piecewise_arcs(zw: np.ndarray, yw: np.ndarray, n_arcs: int):
    """
    与之前相同：把 (z_w, y_w) 按弧长等分 n_arcs 段，每段拟合圆（最小二乘），
    得每段半径 R_i 与角跨度 phi_i（度）。
    """
    pts = np.stack([zw, yw], axis=1)
    ds = np.sqrt(((np.diff(pts, axis=0))**2).sum(axis=1))
    s = np.hstack([[0.0], np.cumsum(ds)])
    cuts = np.linspace(0, s[-1], n_arcs+1)
    R_list, phi_list = [], []
    for k in range(n_arcs):
        s0, s1 = cuts[k], cuts[k+1]
        mask = (s >= s0) & (s <= s1)
        seg = pts[mask]
        if len(seg) < 3:
            i0 = np.searchsorted(s, s0); i1 = min(len(pts), np.searchsorted(s, s1)+2)
            seg = pts[max(0,i0-1):i1]
        X = seg
        A = np.c_[2*X[:,0], 2*X[:,1], np.ones(len(X))]
        b = (X**2).sum(axis=1)
        (a,b2,c), *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = a, b2
        R = float(np.sqrt(cx*cx + cy*cy + c))
        ang = np.arctan2(seg[:,1]-cy, seg[:,0]-cx)
        dphi = float(np.abs(ang[-1]-ang[0]) % (2*np.pi))
        if dphi > np.pi: dphi = 2*np.pi - dphi
        R_list.append(R); phi_list.append(np.degrees(dphi))
    return np.array(phi_list), np.array(R_list)

def waverider_from_params(p: WaveriderParams):
    """
    原样复刻你 waverider.py 的生成流程（见你脚本里 Conical_flow/streamline 的用法）。
    返回 Upper_surface, Lower_surface，shape=(nj, ni, 3)
    """
    deg = np.pi/180.0
    Ma = p.Ma
    shockangle = p.shock_angle_deg * deg
    n_arcs = int(p.n_arcs)
    phi_arc = np.array([v*deg for v in p.phi_arc_deg], float)
    R_arc = np.array(p.R_arc, float)

    zw0 = np.zeros(n_arcs+1); yw0 = np.zeros(n_arcs+1); phi0 = np.zeros(n_arcs+1)
    zc = np.zeros(n_arcs); yc = np.zeros(n_arcs)
    for i in range(n_arcs):
        phi0[i+1] = phi0[i] + phi_arc[i]
        zw0[i+1] = zw0[i] + (np.sin(phi0[i+1])-np.sin(phi0[i]))*R_arc[i]
        yw0[i+1] = yw0[i] + (-np.cos(phi0[i+1])+np.cos(phi0[i]))*R_arc[i]
        zc[i] = zw0[i] - np.sin(phi0[i])*R_arc[i]
        yc[i] = yw0[i] + np.cos(phi0[i])*R_arc[i]
    for i in range(n_arcs):
        yw0[i] -= yw0[n_arcs]
        yc[i]  -= yw0[n_arcs]
    yw0[n_arcs] = 0.0

    zu = np.asarray(p.zu, float).copy()
    yu = np.asarray(p.yu, float).copy()
    nu = len(zu)
    zu = zu * (zw0[n_arcs] / zu[-1]); yu = yu * (zw0[n_arcs] / zu[-1])

    zw = np.zeros(nu); yw = np.zeros(nu); phiu = np.zeros(nu); Ru = np.zeros(nu); Rw = np.zeros(nu)
    i = 0
    for j in range(nu-1):
        while (np.arctan((zu[j]-zc[i])/(yc[i]-yu[j])) - phi0[i] > phi_arc[i]) and (i < n_arcs-1):
            i += 1
        phiu[j] = np.arctan((zu[j]-zc[i])/(yc[i]-yu[j]))
        zw[j] = zw0[i] + (np.sin(phiu[j]) - np.sin(phi0[i]))*R_arc[i]
        yw[j] = yw0[i] + (-np.cos(phiu[j]) + np.cos(phi0[i]))*R_arc[i]
        Rw[j] = R_arc[i]
        Ru[j] = np.sqrt((zw[j]-zu[j])**2 + (yw[j]-yu[j])**2)
    zw[nu-1] = zw0[n_arcs]; Rw[nu-1] = R_arc[n_arcs-1]

    thetas, vxs, vys = Conical_flow(Ma, shockangle, n_out=p.n_out)
    n_stream = int(p.n_stream)
    Lower_surface = np.zeros((nu, n_stream, 3))
    for ii in range(nu-1):
        xs, ys = streamline(shockangle, Rw[ii], Ru[ii], n_stream, vxs, vys, thetas)
        x0 = -xs[-1]; z0 = zu[ii]; y0 = yu[ii]
        Lower_surface[ii,:,0] = x0 + xs
        Lower_surface[ii,:,1] = y0 - ys*np.cos(phiu[ii])
        Lower_surface[ii,:,2] = z0 + ys*np.sin(phiu[ii])
    Lower_surface[nu-1,:,2] = zu[nu-1]

    Upper_surface = np.zeros_like(Lower_surface)
    for ii in range(nu):
        Upper_surface[ii,:,0] = Lower_surface[ii,:,0]
        Upper_surface[ii,:,1] = yu[ii]
        Upper_surface[ii,:,2] = zu[ii]

    # scale/translate
    if p.scale != 1.0:
        Lower_surface *= p.scale; Upper_surface *= p.scale
    t = np.array([p.translate_x, p.translate_y, p.translate_z], float)
    if np.linalg.norm(t) > 0:
        Lower_surface += t; Upper_surface += t
    return Upper_surface, Lower_surface

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
    d_shell = np.max(np.abs(absx - half), axis=1)
    residual = float(np.mean(np.abs(d_shell)))
    param = BoxParam(tuple(c.tolist()), tuple(map(tuple, R)), tuple(ext.tolist()))
    return param, residual

# ----------------------- 模型选择与 JSON -----------------------
def _shape_metrics(points: np.ndarray):
    c = points.mean(axis=0)
    X = points - c
    # AABB
    mins = X.min(axis=0); maxs = X.max(axis=0)
    ext = maxs - mins
    # 各向异性：最大/最小边
    ex, ey, ez = ext
    eps = 1e-12
    aniso_aabb = (ext.max() + eps) / (ext.min() + eps)

    # PCA 奇异值
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  # S ~ sqrt(eigvals of cov up to scale)
    s1, s2, s3 = S
    aniso_pca12 = (s1 + eps) / (s2 + eps)
    aniso_pca23 = (s2 + eps) / (s3 + eps)

    # 球一致性：到中心距离的变异系数 CoV
    dist = np.linalg.norm(X, axis=1)
    cov_sphere = (dist.std() + eps) / (dist.mean() + eps)

    # 圆柱一致性：沿主轴投影 t 与径向 r
    axis = Vt[0]  # 主轴
    t = X @ axis
    radial = np.linalg.norm(X - np.outer(t, axis), axis=1)
    # 去掉两端 5%（端盖）
    q1, q99 = np.percentile(t, [5, 95])
    mask_side = (t >= q1) & (t <= q99)
    cov_cyl = (radial[mask_side].std() + eps) / (radial[mask_side].mean() + eps)
    height = float(t.max() - t.min())
    radius_med = float(np.median(radial[mask_side])) if mask_side.any() else float(np.median(radial))
    hw_ratio = (height + eps) / (2.0 * radius_med + eps)

    return {
        "extents": ext,
        "aniso_aabb": float(aniso_aabb),
        "aniso_pca12": float(aniso_pca12),
        "aniso_pca23": float(aniso_pca23),
        "cov_sphere": float(cov_sphere),
        "cov_cyl": float(cov_cyl),
        "height": float(height),
        "radius_med": float(radius_med),
        "hw_ratio": float(hw_ratio),  # 高度/直径
        "axis": axis
    }


def _adjust_scores_by_priors(points: np.ndarray, raw_scores: dict):
    m = _shape_metrics(points)
    scores = raw_scores.copy()

    # --- Sphere 先验 ---
    # 球应当：到心距离 CoV 很小 & AABB 各向同性
    if m["cov_sphere"] > 0.07 or m["aniso_aabb"] > 1.15:
        scores["sphere"] *= 1.8  # 惩罚
    else:
        scores["sphere"] *= 0.9  # 合理就给点偏好

    # --- Cylinder 先验 ---
    # 圆柱应当：半径 CoV 小、长高比合适、PCA 第一特征显著
    if m["cov_cyl"] > 0.12 or m["hw_ratio"] < 0.8 or m["aniso_pca12"] < 1.2:
        scores["cylinder"] *= 1.4
    else:
        scores["cylinder"] *= 0.9

    # --- Cuboid 先验 ---
    # 盒子：三个主轴差异较明显（不全接近），且不是接近球体的等边
    near_cube = (m["aniso_aabb"] < 1.12)
    axial_separation_ok = (m["aniso_pca12"] > 1.08) or (m["aniso_pca23"] > 1.08)
    if near_cube and (m["cov_sphere"] < 0.06):  # 很像球就惩罚盒子
        scores["cuboid"] *= 1.5
    elif not axial_separation_ok:
        scores["cuboid"] *= 1.2
    else:
        scores["cuboid"] *= 0.95

    return scores, m


def choose_best_model(points: np.ndarray, prefer: str | None = None) -> Dict[str, Any]:
    sp, r_s = fit_sphere(points)
    cy, r_c = fit_cylinder(points)
    bx, r_b = fit_box(points)

    # 归一化后分数（你原来怎么做就怎么做；这里示例用 std 尺度）
    scale = np.linalg.norm(points.std(axis=0)) + 1e-9
    scores = {
        'sphere': r_s / scale,
        'cylinder': r_c / scale,
        'cuboid': r_b / scale,
    }

    # 可选：轻微惩罚某些更“宽容”的模型，防止一边倒（按需保留/调整）
    # scores['cuboid']      *= 1.05   # 盒子不再过度占优
    # scores['cylinder'] *= 1.00   # 也可微调圆柱
    # scores['sphere']   *= 1.00

    # 先取原始最优
    best = min(scores, key=scores.get)

    # 领先阈值：最优与次优差距不足 8% 时，允许 --prefer 覆盖
    sorted_vals = sorted((v, k) for k, v in scores.items())
    best_val, best_key = sorted_vals[0]
    second_val, second_key = sorted_vals[1]
    lead_ratio = (second_val + 1e-12) / (best_val + 1e-12)  # 次优/最优

    if prefer in scores and lead_ratio < 1.08:
        # 分差接近时采用用户偏好
        best = prefer

    # 组装返回 payload（保持你原来的结构）
    if best == 'sphere':
        payload = {'type': 'sphere', 'params': asdict(sp), 'fit': {'score': scores['sphere'], 'all': scores}}
    elif best == 'cylinder':
        payload = {'type': 'cylinder', 'params': asdict(cy), 'fit': {'score': scores['cylinder'], 'all': scores}}
    else:
        payload = {'type': 'cuboid', 'params': asdict(bx), 'fit': {'score': scores['cuboid'], 'all': scores}}

    return payload



# ----------------------- 基于参数生成网格 -----------------------
def _to_4x4(M):
    M = np.asarray(M, dtype=float)
    if M.shape == (4, 4):
        return M
    if M.shape == (3, 3):
        T = np.eye(4)
        T[:3, :3] = M
        return T
    raise ValueError(f"Expected 3x3 or 4x4 matrix, got {M.shape}")

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

        # 对齐 z 轴到目标轴：align_vectors 返回 4x4
        M = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), axis)
        T = _to_4x4(M)
        m.apply_transform(T)
        m.apply_translation(center)
        return m

    # —— 修改 mesh_from_params 中的盒子分支（兼容 R 既可能是 3x3 也可能是 4x4）：
    if t == 'cuboid':
        center = np.array(p['center'])
        R = _to_4x4(p['R'])   # 这里兼容 3x3 / 4x4
        ext = np.array(p['extents'], dtype=float)
        m = trimesh.creation.box(extents=ext)
        T = R.copy()
        T[:3, 3] = center     # 再加上平移
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

    # 1) 强制类型：直接拟合对应形状并组装 spec
    if getattr(args, 'force_type', None):
        forced = args.force_type
        print(f"[INFO] 强制拟合类型: {forced}")
        if forced == 'sphere':
            param, _ = fit_sphere(pts)
        elif forced == 'cylinder':
            param, _ = fit_cylinder(pts)
        else:
            param, _ = fit_box(pts)
        spec = {
            'type': forced,
            'params': asdict(param),
            'fit': {'forced': True}
        }

    else:
        # 2) 正常自动判别，但允许 --prefer 在分差接近时生效
        spec = choose_best_model(pts, prefer=getattr(args, 'prefer', None))

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
    import os
    o3d, gui, rendering = _o3d_gui_modules()

    # 加载参数
    has_spec = bool(getattr(args, "json", None)) and os.path.exists(args.json)
    spec = None
    if has_spec:
        with open(args.json, 'r', encoding='utf-8') as f:
            spec = json.load(f)

    orig_mesh = None
    if args.stl:
        _, orig_mesh = load_points_from_stl(args.stl, n_samples=min(args.samples, 50000))

    # 初始 mesh
    tri = None
    if has_spec:
        tri = _spec_to_trimesh(spec)

    if not has_spec and orig_mesh is None:
        raise SystemExit("cmd_gui: 既没有 --json 也没有 --stl，无法展示。请至少提供一个。")

    # GUI 初始化
    app = gui.Application.instance
    app.initialize()

    w = gui.Application.instance.create_window("Param GUI", 1280, 800)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    scene.scene.set_background([1,1,1,1])

    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultLit"   # 三角网格用带光照的

    mat_line = rendering.MaterialRecord()
    mat_line.shader = "unlitLine"    # 线框要用 unlitLine 才不会缺属性
    mat_line.line_width = 2.0        # 线宽可调

    # 添加几何体
    if orig_mesh is not None:
        o3 = _trimesh_to_o3d(orig_mesh)
        scene.scene.add_geometry("orig", o3, mat_mesh)
        scene.scene.show_geometry("orig", True)

    if tri is not None:
        tri_o3 = _trimesh_to_o3d(tri)
        scene.scene.add_geometry("fit", tri_o3, mat_mesh)

    # 坐标轴与 bbox
    # size_hint = float(np.linalg.norm(tri.bounding_box.extents) if hasattr(tri, 'bounding_box') else 1.0)
    # axes = _make_axes(max(size_hint*0.25, 1e-3))
    # scene.scene.add_geometry("axes", axes, mat_line)

    # bbox_lines = _bbox_lines_o3d(tri)
    # scene.scene.add_geometry("bbox", bbox_lines, mat_line)

    # # 相机
    # bounds = scene.scene.bounding_box
    # scene.setup_camera(60.0, bounds, bounds.get_center())

    ref_mesh = tri if tri is not None else orig_mesh
    size_hint = float(np.linalg.norm(ref_mesh.bounding_box.extents) if hasattr(ref_mesh, 'bounding_box') else 1.0)
    axes = _make_axes(max(size_hint * 0.25, 1e-3))
    scene.scene.add_geometry("axes", axes, mat_line)

    bbox_lines = _bbox_lines_o3d(ref_mesh)
    scene.scene.add_geometry("bbox", bbox_lines, mat_line)

    # 相机
    bounds = scene.scene.bounding_box
    scene.setup_camera(60.0, bounds, bounds.get_center())

    if has_spec:
        # 右侧面板（控件）
        panel = gui.Vert(0, gui.Margins(8,8,8,8))
        info = gui.Label("")

        def refresh_geometry():
            nonlocal tri_o3, bbox_lines
            # 重建拟合网格
            new_tri = _spec_to_trimesh(spec)
            new_o3 = _trimesh_to_o3d(new_tri)
            scene.scene.remove_geometry("fit")
            scene.scene.add_geometry("fit", new_o3, mat_mesh)
            tri_o3 = new_o3
            # bbox 更新
            scene.scene.remove_geometry("bbox")
            bbox_lines = _bbox_lines_o3d(new_tri)
            scene.scene.add_geometry("bbox", bbox_lines, mat_line)
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

        elif t == 'cuboid':
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
        # —— 新布局：优先用 Splitter，失败则用 on_layout —— 
        if hasattr(gui, "Splitter"):
            splitter = gui.Splitter(gui.Splitter.HORIZONTAL)
            splitter.add_child(scene)

            # 有的构建没有 ScrollView：前面你已经做了兼容，变量名统一用 scroll_or_panel
            right = scroll_or_panel if 'scroll_or_panel' in locals() else panel
            splitter.add_child(right)

            # 初始分割比：左 75% / 右 25%
            # 注意：需要在窗口布局完成后设置一次
            def _on_layout(_):
                r = w.content_rect
                splitter.frame = r
                splitter.set_divider_position(int(r.width * 0.75))
            w.set_on_layout(_on_layout)

            w.add_child(splitter)

        else:
            # 极限降级（没有 Splitter）：手动 on_layout
            # 把 scene 放左侧 75%，右侧放 panel
            w.add_child(scene)
            right = scroll_or_panel if 'scroll_or_panel' in locals() else panel
            w.add_child(right)

            def _on_layout(_):
                r = w.content_rect
                left_w = int(r.width * 0.75)
                scene.frame = gui.Rect(r.x, r.y, left_w, r.height)
                right.frame = gui.Rect(r.x + left_w, r.y, r.width - left_w, r.height)
            w.set_on_layout(_on_layout)

        # 生成一次以更新 info
        refresh_geometry()
    else:
        # —— 无 JSON：仅展示 STL，全屏场景，无滑块 —— 
        w.add_child(scene)
        def _on_layout(_):
            scene.frame = w.content_rect
        w.set_on_layout(_on_layout)

    app.run()


def cmd_preview_plot3d(args):
    import trimesh
    # 汇总多个 .xyz（例如上/下表面），并可选不同顺序
    meshes = []
    for p in args.xyz:
        blocks = read_plot3d_xyz(p, order=args.order)
        meshes.append(plot3d_blocks_to_trimesh(blocks))
    merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    if args.out:
        merged.export(args.out)
        print(f"Saved mesh → {args.out}")
    # 复用现有的 Open3D 预览管线（与你的 preview/gen 一致）
    preview_scene(None, merged, smooth=not getattr(args, 'flat', False))


import os

def cmd_wfit_stl(args):
    # 读取 STL
    mesh = load_mesh_as_trimesh(args.stl)
    mesh = ensure_trimesh(mesh)
    
    # 分上/下
    upper, lower = _split_upper_lower_from_stl(mesh, ny_thresh=args.ny_thresh)
    # 提取 (zu,yu)
    zu, yu = _extract_upline_from_upper(upper, tol=args.cluster_tol)
    # 提取 (zw,yw)
    zw, yw = _extract_wavecurve_from_lower(lower, k_percent=args.x_head_percent, bins=args.z_bins)
    # 拟合扇形圆弧
    phi_arc_deg, R_arc = _fit_piecewise_arcs(zw, yw, n_arcs=args.n_arcs)

    p = WaveriderParams(
        Ma=args.Ma, shock_angle_deg=args.shock,
        n_arcs=args.n_arcs,
        phi_arc_deg=[float(v) for v in phi_arc_deg],
        R_arc=[float(v) for v in R_arc],
        zu=[float(v) for v in zu],
        yu=[float(v) for v in yu],
        n_stream=args.n_stream, n_out=args.n_out,
        scale=1.0
    )
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    save_waverider_params_json(p, args.json)
    print(f"[wfit_stl] saved → {args.json}")

    if args.preview:
        U, L = waverider_from_params(p)
        blocks = [(U[None,:,:,0], U[None,:,:,1], U[None,:,:,2]),
                  (L[None,:,:,0], L[None,:,:,1], L[None,:,:,2])]
        mesh2 = plot3d_blocks_to_trimesh(blocks)
        preview_scene(None, mesh2)


def ensure_trimesh(obj) -> trimesh.Trimesh:
    """
    把 obj 统一成 Trimesh。
    支持：
      - Trimesh 直接返回
      - (V,F) / [V,F] 其中 V=(N,3), F=(M,3) -> Trimesh
      - list/tuple 的 Trimesh 集合 -> 合并
    其余类型报错。
    """
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, (list, tuple)):
        # 1) list/tuple[Trimesh...] -> 合并
        if all(isinstance(x, trimesh.Trimesh) for x in obj):
            return trimesh.util.concatenate(list(obj))
        # 2) (V,F)
        if len(obj) == 2 and all(isinstance(x, np.ndarray) for x in obj):
            V, F = obj
            if V.ndim == 2 and V.shape[1] == 3 and F.ndim == 2 and F.shape[1] == 3:
                return trimesh.Trimesh(vertices=V, faces=F, process=True)
    # 3) 单个 ndarray 不支持（没有 faces）
    if isinstance(obj, np.ndarray):
        raise TypeError("Got a NumPy array (vertices) but no faces. Need a triangulated mesh.")
    raise TypeError(f"Unsupported mesh type: {type(obj)}")


if __name__ == '__main__':
    geometry = "cylinder" # 可选 sphere/cylinder/cuboid

    model_path = f"models/{geometry}.STL"
    json_path = f"jsons/{geometry}.json"
    out_path = f"outs/{geometry}.stl"
    prefer = geometry
    forceType = geometry

    waverider_stl = "waverider/waverider_solid.stl"
    waverider_json = "jsons/waverider_params.json"

    ap = argparse.ArgumentParser(description='STL 参数化拟合与生成 (sphere/cylinder/cuboid)')
    sub = ap.add_subparsers(required=True)

    p_fit = sub.add_parser('fit', help='从 STL 拟合并导出 JSON 参数')
    p_fit.add_argument('--stl', default=model_path)
    p_fit.add_argument('--json', default=json_path)
    p_fit.add_argument('--samples', type=int, default=200000)
    p_fit.add_argument('--preview', default=True, action='store_true')
    p_fit.add_argument('--flat', action='store_true', help='禁用平滑着色（避免依赖 networkx/scipy）')
    p_fit.add_argument('--force-type', default=geometry, choices=['sphere', 'cylinder', 'cuboid'],
                   help='强制拟合为指定类型（跳过自动判别）')
    p_fit.add_argument('--prefer', default=geometry, choices=['sphere', 'cylinder', 'cuboid'],
                    help='当残差接近时优先此类型')
    p_fit.set_defaults(func=cmd_fit)

    p_gen = sub.add_parser('gen', help='从 JSON 生成新网格，支持用 --set 修改参数')
    p_gen.add_argument('--json', default=json_path)
    p_gen.add_argument('--set', nargs='*', default=[], help='形如 key=val，例如 radius=2.5 或 extents=[2,3,4]')
    p_gen.add_argument('--out', default=out_path, help='输出新 STL/OBJ 路径，后缀决定格式')
    p_gen.add_argument('--preview', action='store_true')
    p_gen.add_argument('--flat', action='store_true', help='禁用平滑着色（避免依赖 networkx/scipy）')
    p_gen.set_defaults(func=cmd_gen)

    p_prev = sub.add_parser('preview', help='可视化原始 STL 与 JSON 拟合模型')
    p_prev.add_argument('--stl', default=model_path)
    p_prev.add_argument('--json', default=json_path)
    p_prev.add_argument('--flat', action='store_true', help='禁用平滑着色（避免依赖 networkx/scipy）')
    p_prev.set_defaults(func=cmd_preview)

    # 交互 GUI 子命令（Open3D）
    p_gui = sub.add_parser('gui', help='Open3D 交互式预览，带滑块调参与坐标轴/尺寸标注')
    # p_gui.add_argument('--json', default=json_path, help='参数 JSON 文件')
    p_gui.add_argument('--json', help='参数 JSON 文件')
    p_gui.add_argument('--stl', default=model_path, help='可选：原始 STL（作为参考线框或实体）')
    p_gui.add_argument('--samples', type=int, default=80000, help='若传入 --stl 用于抽样显示')
    p_gui.set_defaults(func=cmd_gui)

    p_p3d = sub.add_parser('preview_plot3d', help='预览 Plot3D (.xyz) 曲面（支持多个文件合并显示）')
    p_p3d.add_argument('--xyz', nargs='+', required=True, help='一个或多个 Plot3D .xyz 路径（可传上下表面）')
    p_p3d.add_argument('--order', default='xzy', choices=['xyz','xzy'],
                    help='文件内坐标写入顺序（waverider.py 输出是 xzy）')
    p_p3d.add_argument('--out', default=None, help='可选：把合并后的网格导出为 STL/OBJ')
    p_p3d.add_argument('--flat', action='store_true', help='禁用平滑着色')
    p_p3d.set_defaults(func=cmd_preview_plot3d)

    # 注册 Waverider CLI
    p_wfit_stl = sub.add_parser('wfit_stl', help='从 STL 拟合 waverider 参数并保存 JSON')
    p_wfit_stl.add_argument('--stl', default=waverider_stl, help='输入 STL（建议是整机或半机；最好是刚缝合的 waverider）')
    p_wfit_stl.add_argument('--n-arcs', type=int, default=3)
    p_wfit_stl.add_argument('--Ma', type=float, default=6.0)
    p_wfit_stl.add_argument('--shock', type=float, default=30.0)
    p_wfit_stl.add_argument('--n-stream', type=int, default=600)
    p_wfit_stl.add_argument('--n-out', type=int, default=800)
    p_wfit_stl.add_argument('--json', default=waverider_json)
    # 下面是 STL 解析的鲁棒参数
    p_wfit_stl.add_argument('--ny-thresh', type=float, default=0.95, help='上表面判定阈值 |n·ŷ|≥此值')
    p_wfit_stl.add_argument('--cluster-tol', type=float, default=5e-4, help='(y,z) 聚类公差（相对尺度）')
    p_wfit_stl.add_argument('--x-head-percent', type=float, default=5.0, help='取最靠前 x 的百分位来估计波接触线')
    p_wfit_stl.add_argument('--z-bins', type=int, default=200, help='(zw,yw) z 分箱数')
    p_wfit_stl.add_argument('--preview', default=True, action='store_true')
    p_wfit_stl.set_defaults(func=cmd_wfit_stl)

    args = ap.parse_args()
    args.func(args)
