# view_3d.py
import os, sys, tempfile
import numpy as np
import open3d as o3d

# 可选：仅用于 PyVista 可视化
try:
    import pyvista as pv
    _HAS_PYVISTA = True
except Exception:
    _HAS_PYVISTA = False

from cadquery import importers, exporters


def load_mesh_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    mesh_exts = {".stl", ".obj", ".ply", ".off", ".gltf", ".glb"}
    cad_step = {".step", ".stp"}
    cad_iges = {".iges", ".igs"}

    if ext in mesh_exts:
        m = o3d.io.read_triangle_mesh(path)
        if len(m.vertices) == 0:
            pcd = o3d.io.read_point_cloud(path)
            if len(pcd.points) == 0:
                raise RuntimeError("Failed to read mesh/pcd.")
            return pcd
        m.compute_vertex_normals()
        return m

    if ext in cad_step | cad_iges:
        shape = importers.importStep(path) if ext in cad_step else importers.importIGES(path)
        tmp_stl = tempfile.NamedTemporaryFile(suffix=".stl", delete=False).name
        try:
            exporters.export(shape, tmp_stl)   # 三角化
            m = o3d.io.read_triangle_mesh(tmp_stl)
            if len(m.vertices) == 0:
                raise RuntimeError("Tessellation produced empty mesh.")
            m.compute_vertex_normals()
            return m
        finally:
            try: os.remove(tmp_stl)
            except: pass

    raise ValueError(f"Unsupported extension: {ext}")


def to_pyvista(geo):
    import pyvista as pv
    if isinstance(geo, o3d.geometry.TriangleMesh):
        v = np.asarray(geo.vertices)
        f = np.asarray(geo.triangles)
        faces = np.hstack([np.full((f.shape[0], 1), 3), f]).astype(np.int64).ravel()
        return pv.PolyData(v, faces)
    elif isinstance(geo, o3d.geometry.PointCloud):
        pts = np.asarray(geo.points)
        return pv.PolyData(pts)
    else:
        raise TypeError(f"Unsupported type: {type(geo)}")


def show_with_pyvista(geo, title="3D Viewer"):
    pvgeo = to_pyvista(geo)
    pl = pv.Plotter(window_size=(1200, 800))
    pl.add_title(title)
    pl.add_axes()
    if isinstance(geo, o3d.geometry.TriangleMesh):
        pl.add_mesh(pvgeo, show_edges=True, smooth_shading=True, opacity=1.0)
    else:
        pl.add_points(pvgeo, render_points_as_spheres=True, point_size=4.0, opacity=0.9)
    pl.enable_anti_aliasing()
    pl.camera_position = "iso"
    pl.show()


def show_with_open3d(geo):
    # open3d 的原生查看器（兜底）
    if isinstance(geo, o3d.geometry.PointCloud):
        o3d.visualization.draw_geometries([geo])
    elif isinstance(geo, o3d.geometry.TriangleMesh):
        o3d.visualization.draw_geometries([geo])
    else:
        raise TypeError(type(geo))
    

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import cadquery as cq

# ----------------- 拟合：平面/交线/圆柱（固定轴） -----------------
def plane_fit_ransac(pcd, dist=0.001, n=3, iters=2000):
    model, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=n, num_iterations=iters)
    a,b,c,d = model
    nrm = np.array([a,b,c], dtype=float)
    nrm /= np.linalg.norm(nrm)
    return dict(a=a,b=b,c=c,d=d, normal=nrm, inliers=np.array(inliers))

def plane_intersection_line(plane1, plane2):
    n1, n2 = plane1["normal"], plane2["normal"]
    u = np.cross(n1, n2)
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        raise RuntimeError("Planes are parallel or ill-conditioned")
    u /= nu
    A = np.vstack([n1, n2])
    b = -np.array([plane1["d"], plane2["d"]])
    p0 = np.linalg.lstsq(A, b, rcond=None)[0]
    return p0, u

def point_plane_distance(points, plane):
    a,b,c,d = plane["a"], plane["b"], plane["c"], plane["d"]
    n = np.array([a,b,c])
    nrm = np.linalg.norm(n)
    return (points @ n + d)/nrm

def extract_fillet_band(points, plane_top, plane_side, near=0.01, far=0.0005):
    dt = np.abs(point_plane_distance(points, plane_top))
    ds = np.abs(point_plane_distance(points, plane_side))
    sel = (dt < near) & (ds < near) & (dt > far) & (ds > far)
    return points[sel]

def fit_cylinder_fixed_axis(points, line_p0, axis_u, r0=None):
    if points is None or len(points) == 0:
        raise ValueError("No points provided for cylinder fitting (band is empty). "
                         "Try relaxing thresholds or using extract_fillet_band_auto.")
    u = axis_u / (np.linalg.norm(axis_u) + 1e-12)

    # 初值
    t0 = 0.0
    if r0 is None:
        v = points - line_p0
        v_par = (v @ u)[:, None] * u
        v_perp = v - v_par
        radii = np.linalg.norm(v_perp, axis=1)
        if not np.isfinite(radii).any():
            raise ValueError("Cannot compute initial radius: all radii are non-finite.")
        r0 = float(np.nanmedian(radii))
    if not np.isfinite(r0) or r0 <= 0:
        # 最后兜底：用包围盒尺寸当一个粗略半径
        bbox = points.max(axis=0) - points.min(axis=0)
        r0 = float(np.linalg.norm(bbox)) * 0.02  # 取对角线 2%

    x0 = np.array([t0, r0], dtype=float)
    if not np.isfinite(x0).all():
        raise ValueError(f"Initial guess contains NaN/Inf: x0={x0}")

    def residuals(x):
        t, r = x
        p0 = line_p0 + t * u
        v = points - p0
        v_par = (v @ u)[:, None] * u
        v_perp = v - v_par
        return np.linalg.norm(v_perp, axis=1) - r

    # 明确给 bounds，避免 r 走到负数或异常大
    # r 下界 > 0；上界用模型尺度的 0.5 做个安全帽
    bbox = points.max(axis=0) - points.min(axis=0)
    r_upper = max(1e-6, float(np.linalg.norm(bbox))) * 0.5
    bounds = ([-np.inf, 1e-9], [np.inf, r_upper])

    res = least_squares(residuals, x0, loss="soft_l1", f_scale=0.5, bounds=bounds)
    t_fit, r_fit = res.x
    p0_fit = line_p0 + t_fit * u
    return p0_fit, u, abs(r_fit), res

def extract_fillet_band_auto(points, plane_top, plane_side,
                             min_count=300,   # 希望至少这么多点参与拟合
                             max_iter=6):     # 最多放宽几轮
    # 距离两平面的绝对值
    dt = np.abs(point_plane_distance(points, plane_top))
    ds = np.abs(point_plane_distance(points, plane_side))

    # 用模型尺度归一（避免单位不同）
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = np.linalg.norm(bbox_max - bbox_min) + 1e-12
    eps = 1e-4 * bbox_size    # 用来排除“完全在平面里”的点

    # 迭代放宽：从较严格的分位数开始，逐步放宽
    # q_start 越小越严格（更靠近平面），q_inc 每轮放宽一些
    q = 0.03    # 初始取 3% 分位
    q_inc = 0.03

    for _ in range(max_iter):
        thr_t = np.quantile(dt, q)  # 距离 top 小于该阈值视为“靠近”
        thr_s = np.quantile(ds, q)  # 距离 side 小于该阈值视为“靠近”
        # 同时靠近两平面，但又不能太贴某一个平面（> eps）
        sel = (dt < thr_t) & (ds < thr_s) & (dt > eps) & (ds > eps)
        band = points[sel]
        if band.shape[0] >= min_count:
            return band, dict(thr_top=thr_t, thr_side=thr_s, eps=eps, q=q)
        q = min(q + q_inc, 0.35)  # 最多放宽到 35%
    # 仍然不够，就退一步：不做 “> eps” 限制（容忍带少量平面点）
    sel = (dt < np.quantile(dt, q)) & (ds < np.quantile(ds, q))
    band = points[sel]
    return band, dict(thr_top=float(np.quantile(dt, q)), thr_side=float(np.quantile(ds, q)), eps=eps, q=q, relaxed=True)


# ---------- 拟合盒子圆角 ----------
def fit_fillet_box(mesh: o3d.geometry.TriangleMesh, sample_pts=25000,
                   ransac_dist=None):
    # 点云
    pcd = mesh.sample_points_uniformly(sample_pts) if isinstance(mesh, o3d.geometry.TriangleMesh) else mesh
    P = np.asarray(pcd.points)
    # 尺度
    s = np.linalg.norm(P.max(axis=0) - P.min(axis=0)) + 1e-12
    if ransac_dist is None:
        ransac_dist = 0.002 * s   # 对角线的 0.2% 作为拟合阈值，可再根据效果调

    # 顶面
    plane_top = plane_fit_ransac(pcd, dist=ransac_dist)
    in_top = plane_top["inliers"]

    # 3) 两个侧面
    mask_rest = np.ones(len(P), dtype=bool)
    mask_rest[in_top] = False
    pcd_rest = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[mask_rest]))

    plane_sideA = plane_fit_ransac(pcd_rest, dist=ransac_dist)
    in_sideA_global = np.where(mask_rest)[0][plane_sideA["inliers"]]
    mask_rest[in_sideA_global] = False
    pcd_rest2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[mask_rest]))
    plane_sideB = plane_fit_ransac(pcd_rest2, dist=ransac_dist)

    # 4) 顶∩侧A、顶∩侧B 的交线
    p0A, uA = plane_intersection_line(plane_top, plane_sideA)
    p0B, uB = plane_intersection_line(plane_top, plane_sideB)

    # 5) 取圆角带点
    bandA, infoA = extract_fillet_band_auto(P, plane_top, plane_sideA)
    bandB, infoB = extract_fillet_band_auto(P, plane_top, plane_sideB)
    print(f"[bandA] {bandA.shape[0]} pts, info={infoA}")
    print(f"[bandB] {bandB.shape[0]} pts, info={infoB}")

    # 6) 拟合两条圆边的半径
    p0A_fit, uA_fit, rA, _ = fit_cylinder_fixed_axis(bandA, p0A, uA)
    p0B_fit, uB_fit, rB, _ = fit_cylinder_fixed_axis(bandB, p0B, uB)

    # 7) 估计 L/W/H（正交化基）
    N = np.stack([plane_top["normal"], plane_sideA["normal"], plane_sideB["normal"]], axis=0)
    U,_,_ = np.linalg.svd(N.T)
    ex, ey, ez = U[:,0], U[:,1], U[:,2]
    proj_x = P @ ex; proj_y = P @ ey; proj_z = P @ ez
    L = float(proj_x.max() - proj_x.min())
    W = float(proj_y.max() - proj_y.min())
    H = float(proj_z.max() - proj_z.min())

    return {
        "planes": dict(top=plane_top, sideA=plane_sideA, sideB=plane_sideB),
        "edges": dict(A=dict(p0=p0A_fit, u=uA_fit, r=rA), B=dict(p0=p0B_fit, u=uB_fit, r=rB)),
        "LWH": (L, W, H),
        "fillet_radius": float(np.median([rA, rB])),
        "basis": (ex, ey, ez)  # 如需做坐标对齐，可用
    }


# ----------------- 参数化重建（顶面两条边圆角） -----------------
def rebuild_box_with_two_top_fillets(L, W, H, r):
    wp = cq.Workplane("XY").box(L, W, H)  # 以中心为原点
    # 顶面两组边：与 X 平行的一组，与 Y 平行的一组
    solid = (wp.faces(">Z").edges("|X").fillet(r)
               .faces(">Z").edges("|Y").fillet(r))
    return solid

# 将 CadQuery 实体转为 open3d 三角网（用于显示）
def cad_to_o3d_mesh(solid: cq.Workplane):
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False).name
    try:
        exporters.export(solid, tmp)  # 三角化
        m = o3d.io.read_triangle_mesh(tmp)
        m.compute_vertex_normals()
        return m
    finally:
        try: os.remove(tmp)
        except: pass

# ----------------- 主函数：拟合 → 交互“捏参数” → 导出 -----------------
def main(path: str):
    # 读取 + 拟合
    src_geo = load_mesh_any(path)
    if isinstance(src_geo, o3d.geometry.TriangleMesh):
        src_geo.compute_vertex_normals()

    result = fit_fillet_box(src_geo, sample_pts=25000, ransac_dist=0.001,
                            band_near=0.01, band_far=0.0005)
    L, W, H = result["LWH"]
    r = result["fillet_radius"]
    print(f"Initial params: L={L:.4f}, W={W:.4f}, H={H:.4f}, r={r:.4f}")

    # 初始重建
    solid = rebuild_box_with_two_top_fillets(L, W, H, r)
    fit_mesh = cad_to_o3d_mesh(solid)

    # 转为 PyVista
    def o3d_to_pv(mesh: o3d.geometry.TriangleMesh):
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        faces = np.hstack([np.full((f.shape[0], 1), 3), f]).astype(np.int64).ravel()
        return pv.PolyData(v, faces)

    pv_src = o3d_to_pv(src_geo) if isinstance(src_geo, o3d.geometry.TriangleMesh) else pv.PolyData(np.asarray(src_geo.points))
    pv_fit = o3d_to_pv(fit_mesh)

    pl = pv.Plotter(window_size=(1250, 820))
    pl.add_title(os.path.basename(path))
    pl.add_axes()

    # 原始模型（半透明参考）
    if isinstance(src_geo, o3d.geometry.TriangleMesh):
        actor_src = pl.add_mesh(pv_src, color=None, opacity=0.3, show_edges=True, smooth_shading=True)
    else:
        actor_src = pl.add_points(pv_src, render_points_as_spheres=True, point_size=3.5, opacity=0.6)

    # 拟合/参数化模型（可交互更新）
    actor_fit = pl.add_mesh(pv_fit, color=None, opacity=0.95, show_edges=True, smooth_shading=True)

    # 当前参数（闭包可修改）
    state = {"L": L, "W": W, "H": H, "r": r}

    def rebuild_and_update():
        solid_new = rebuild_box_with_two_top_fillets(state["L"], state["W"], state["H"], state["r"])
        m = cad_to_o3d_mesh(solid_new)
        pv_new = o3d_to_pv(m)
        actor_fit.mapper.set_input_data(pv_new)  # 替换几何
        pl.render()

    # 滑条：范围按估计值的比例给（可按你的模型调大/调小）
    def s_L(val):
        state["L"] = max(1e-6, float(val)); rebuild_and_update()
    def s_W(val):
        state["W"] = max(1e-6, float(val)); rebuild_and_update()
    def s_H(val):
        state["H"] = max(1e-6, float(val)); rebuild_and_update()
    def s_r(val):
        # 圆角半径不能超过 (L,W,H) 的一半，简单限制一下
        state["r"] = max(1e-6, min(float(val), 0.49*min(state["L"], state["W"]))); rebuild_and_update()

    pl.add_slider_widget(s_L, [0.5*L, 1.5*L], value=L, title="L", pointa=(.02,.12), pointb=(.32,.12))
    pl.add_slider_widget(s_W, [0.5*W, 1.5*W], value=W, title="W", pointa=(.02,.06), pointb=(.32,.06))
    pl.add_slider_widget(s_H, [0.5*H, 1.5*H], value=H, title="H", pointa=(.36,.12), pointb=(.66,.12))
    pl.add_slider_widget(s_r, [0.2*r, 2.0*r], value=r, title="r (fillet)", pointa=(.36,.06), pointb=(.66,.06))

    # 导出 STEP / STL
    def export_step():
        solid_export = rebuild_box_with_two_top_fillets(state["L"], state["W"], state["H"], state["r"])
        exporters.export(solid_export, "edited_model.step")
        print("Exported: edited_model.step")

    def export_stl():
        solid_export = rebuild_box_with_two_top_fillets(state["L"], state["W"], state["H"], state["r"])
        exporters.export(solid_export, "edited_model.stl")
        print("Exported: edited_model.stl")

    pl.add_text("Keys: S = export STEP | L = export STL", position="lower_left", font_size=10)
    pl.add_key_event("s", export_step)
    pl.add_key_event("l", export_stl)

    pl.camera_position = "iso"
    pl.enable_anti_aliasing()
    pl.show()


if __name__ == "__main__":
    path = "param_test.step"
    main(path)
