# sew_waverider_solid.py —— 读取上下 IGES → 镜像 → sewing(仅按 FACE) → 壳 → 实体
from OCP.IGESControl import IGESControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Trsf
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Transform
)
from OCP.BRep import BRep_Builder
from OCP.TopAbs import TopAbs_ShapeEnum
from OCP.TopoDS import TopoDS_Shape, TopoDS_Shell, TopoDS_Face
from OCP.TopExp import TopExp_Explorer
from OCP.ShapeFix import ShapeFix_Shape
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.StlAPI import StlAPI_Writer

TOL = 2e-3  # 适当放大缝合容差，乘波体薄边更容易闭合

def read_iges_one_shape(path: str) -> TopoDS_Shape:
    r = IGESControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError(f"Read IGES failed: {path}")
    r.TransferRoots()
    return r.OneShape()

def mirror_about_xz(shape: TopoDS_Shape) -> TopoDS_Shape:
    tr = gp_Trsf()
    ax2 = gp_Ax2(gp_Pnt(0,0,0), gp_Dir(0,1,0), gp_Dir(1,0,0))  # XZ 面镜像（法向 +Y）
    tr.SetMirror(ax2)
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()

def sew_faces_to_solid(shapes, tol=TOL):
    # 1) sewing
    sew = BRepBuilderAPI_Sewing(tol)
    for s in shapes:
        sew.Add(s)
    sew.Perform()
    sewed = sew.SewedShape()

    # 2) 仅基于 FACE 拼一个 shell（避免 shell downcast）
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)

    expf = TopExp_Explorer(sewed, TopAbs_ShapeEnum.TopAbs_FACE)
    added = False
    while expf.More():
        face = TopoDS_Face(expf.Current())
        builder.Add(shell, face)
        added = True
        expf.Next()
    if not added:
        raise RuntimeError("No faces after sewing; check IGES inputs or tolerance.")

    # 3) 壳 → 实体
    mk = BRepBuilderAPI_MakeSolid()
    mk.Add(shell)
    solid = mk.Solid()

    fixer = ShapeFix_Shape(solid)
    fixer.Perform()
    return fixer.Shape()

if __name__ == "__main__":
    lower = read_iges_one_shape("dump/lower_surface.igs")
    upper = read_iges_one_shape("dump/upper_surface.igs")

    lower_m = mirror_about_xz(lower)
    upper_m = mirror_about_xz(upper)

    solid = sew_faces_to_solid([lower, upper, lower_m, upper_m], tol=TOL)

    # STEP
    w = STEPControl_Writer()
    w.Transfer(solid, STEPControl_AsIs)
    w.Write("waverider_solid.step")
    print("Saved → waverider_solid.step")

    # STL
    StlAPI_Writer().Write(solid, "waverider_solid.stl")
    print("Saved → waverider_solid.stl")
