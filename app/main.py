from load_view_model import load_mesh_any, show_with_pyvista, show_with_open3d, _HAS_PYVISTA
from Geometry import split_fit_and_export

import os
SPLIT = True
VIEW_3D = False

if __name__ == "__main__":
    if SPLIT:
        geo = load_mesh_any("models/ess_ploy.step")
        parts, params = split_fit_and_export(geo, out_dir="out_parts", base="geom")
        print("参数已写入 out_parts/geom_params.json")

    if VIEW_3D:
        path = "models/ess_ploy.step"
        geo = load_mesh_any(path)
        print(f"Loaded: {type(geo).__name__}, verts={len(geo.vertices) if hasattr(geo,'vertices') else len(geo.points)}")

        if _HAS_PYVISTA:
            show_with_pyvista(geo, title=os.path.basename(path))
        else:
            print("PyVista/VTK not available; falling back to Open3D viewer.")
            print("Tip: conda install -c conda-forge vtk pyvista")
            show_with_open3d(geo)