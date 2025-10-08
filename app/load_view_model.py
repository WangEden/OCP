# load_view_model.py
# 读取 CAD/网格/点云模型，并用 open3d 或 pyvista 可视化
# 读取 step/iges/stl/obj/ply/off/gltf/glb/pcd 等格式的网格或点云
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
    if not os.path.isfile(path):
        raise FileNotFoundError(os.path.abspath(path))
    ext = os.path.splitext(path)[1].lower()

    mesh_exts = {".stl", ".obj", ".ply", ".off", ".gltf", ".glb"}
    cad_step = {".step", ".stp"}
    cad_iges = {".iges", ".igs"}
    point_exts = {".pcd", ".xyz", ".txt", ".csv"}

    # 1) 直接是网格格式 → open3d 读
    if ext in mesh_exts:
        m = o3d.io.read_triangle_mesh(path)
        if len(m.vertices) == 0:
            pcd = o3d.io.read_point_cloud(path)
            if len(pcd.points) == 0:
                raise RuntimeError("Failed to read mesh/pcd.")
            return pcd
        m.compute_vertex_normals()
        return m

    # 2) CAD 格式（STEP/IGES）→ cadquery 读 → 暂存 STL → open3d 读
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
    
    # 3) 点云格式 → open3d 读
    if ext in point_exts:
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise RuntimeError("Failed to read point cloud.")
        return pcd

    raise ValueError(f"Unsupported extension: {ext}")


# 利用 pyvista 显示
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


# 利用 open3d 显示
def show_with_open3d(geo):
    # open3d 的原生查看器（兜底）
    if isinstance(geo, o3d.geometry.PointCloud):
        o3d.visualization.draw_geometries([geo])
    elif isinstance(geo, o3d.geometry.TriangleMesh):
        o3d.visualization.draw_geometries([geo])
    else:
        raise TypeError(type(geo))
    

def main(path):
    geo = load_mesh_any(path)
    print(f"Loaded: {type(geo).__name__}, verts={len(geo.vertices) if hasattr(geo,'vertices') else len(geo.points)}")

    if _HAS_PYVISTA:
        show_with_pyvista(geo, title=os.path.basename(path))
    else:
        print("PyVista/VTK not available; falling back to Open3D viewer.")
        print("Tip: conda install -c conda-forge vtk pyvista")
        show_with_open3d(geo)


if __name__ == "__main__":
    # path = "models/param_test.step"
    path = "models/ess_ploy.step"
    main(path)
