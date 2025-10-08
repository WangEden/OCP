# load_model.py
import os, tempfile
import numpy as np
import open3d as o3d
import pyvista as pv

# ---- 仅当你需要 meshio 兜底读取 OBJ/STL 时再引入 ----
# import meshio

# STEP/IGES 走 cadquery
from cadquery import importers, exporters

def load_mesh(path: str) -> o3d.geometry.TriangleMesh | o3d.geometry.PointCloud:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {os.path.abspath(path)}")

    ext = os.path.splitext(path)[1].lower()

    # 1) 直接是网格格式 → open3d 读
    mesh_exts = {".stl", ".obj", ".ply", ".off", ".gltf", ".glb"}
    if ext in mesh_exts:
        m = o3d.io.read_triangle_mesh(path)
        if len(m.vertices) == 0:
            # 某些 OBJ 只有点集，尝试当点云读
            pcd = o3d.io.read_point_cloud(path)
            if len(pcd.points) == 0:
                raise RuntimeError(f"Failed to read mesh or point cloud from {path}")
            return pcd
        m.compute_vertex_normals()
        return m

    # 2) CAD 格式（STEP/IGES）→ cadquery 读 → 暂存 STL → open3d 读
    cad_step_exts = {".step", ".stp"}
    cad_iges_exts = {".iges", ".igs"}
    if ext in cad_step_exts | cad_iges_exts:
        try:
            if ext in cad_step_exts:
                shape = importers.importStep(path)
            else:
                shape = importers.importIGES(path)
        except Exception as e:
            raise RuntimeError(f"CadQuery failed to import CAD file: {e}")

        # 导出为临时 STL（三角化）
        tmp_stl = tempfile.NamedTemporaryFile(suffix=".stl", delete=False).name
        try:
            exporters.export(shape, tmp_stl)   # 由 OCP/OCCT 完成三角化
            m = o3d.io.read_triangle_mesh(tmp_stl)
            if len(m.vertices) == 0:
                raise RuntimeError("Tessellation produced empty mesh.")
            m.compute_vertex_normals()
            return m
        finally:
            # 你也可以选择保留 STL 便于调试；这里默认删除
            try:
                os.remove(tmp_stl)
            except Exception:
                pass

    # 3) 其它扩展名：明确报错
    raise ValueError(
        f"Unsupported or unknown extension '{ext}'. "
        f"Mesh: {sorted(mesh_exts)} | CAD: {sorted(cad_step_exts | cad_iges_exts)}"
    )


def to_pyvista_geometry(geo):
    if isinstance(geo, o3d.geometry.TriangleMesh):
        vertices = np.asarray(geo.vertices)
        faces = np.asarray(geo.triangles)
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).ravel()
        return pv.PolyData(vertices, faces_pv)
    elif isinstance(geo, o3d.geometry.PointCloud):
        return pv.PolyData(np.asarray(geo.points))
    else:
        raise TypeError(f"Unsupported type: {type(geo)}")


if __name__ == "__main__":
    path = "param_test.step"   # 或 .stp/.iges/.igs/.stl/.obj...
    geo = load_mesh(path)
    print(type(geo), geo)
    pvgeo = to_pyvista_geometry(geo)
    print(pvgeo)