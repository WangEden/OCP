# sew_hypersonic_solid.py
from OCP.IGESControl import IGESControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Trsf
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Transform
from OCP.BRep import BRep_Builder
from OCP.TopAbs import TopAbs_ShapeEnum
from OCP.TopoDS import TopoDS_Shape, TopoDS_Shell, TopoDS_Face
from OCP.TopExp import TopExp_Explorer
from OCP.ShapeFix import ShapeFix_Shape
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.StlAPI import StlAPI_Writer

PARTS = ["forebody","body1","uppersurf","lowersurf","ending_upper","ending_lower","side"]  # 视输出而定
TOL = 2e-3   # 缝合容差，可按需要加大些

def read_iges(path:str)->TopoDS_Shape:
    r = IGESControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError(f"Read IGES failed: {path}")
    r.TransferRoots()
    return r.OneShape()

def mirror_about_xz(shape: TopoDS_Shape) -> TopoDS_Shape:
    # 关于 XZ 平面镜像（法向 +Y），把半机变全机
    tr = gp_Trsf()
    tr.SetMirror(gp_Ax2(gp_Pnt(0,0,0), gp_Dir(0,1,0), gp_Dir(1,0,0)))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()

from OCP.TopoDS import TopoDS_Shape, TopoDS_Shell, TopoDS_Face, TopoDS

def sew_faces_to_solid(shapes, tol=TOL):
    sew = BRepBuilderAPI_Sewing(tol)
    for s in shapes:
        sew.Add(s)
    sew.Perform()
    sewed = sew.SewedShape()

    # 用 BRep_Builder 把所有面拼成壳
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)

    expf = TopExp_Explorer(sewed, TopAbs_ShapeEnum.TopAbs_FACE)
    has_face = False
    while expf.More():
        # ✅ 使用 TopoDS.Face() 正确 downcast
        face = TopoDS.Face(expf.Current())
        builder.Add(shell, face)
        has_face = True
        expf.Next()

    if not has_face:
        raise RuntimeError("No faces found after sewing — check IGES inputs or tolerance.")

    mk = BRepBuilderAPI_MakeSolid()
    mk.Add(shell)
    solid = mk.Solid()

    fixer = ShapeFix_Shape(solid)
    fixer.Perform()
    return fixer.Shape()


if __name__ == "__main__":
    # 读取半机各部件
    halves = []
    for p in PARTS:
        halves.append(read_iges(f"./hypersonic_aircraft/{p}.igs"))

    # 镜像得到另一半（如果你的脚本已经输出全机，可跳过镜像）
    mirrored = [mirror_about_xz(s) for s in halves]

    # 缝合成一个实体
    solid = sew_faces_to_solid(halves + mirrored, tol=TOL)

    # 导出 STEP / STL
    w = STEPControl_Writer(); w.Transfer(solid, STEPControl_AsIs); w.Write("model/hypersonic.step")
    StlAPI_Writer().Write(solid, "model/hypersonic.stl")
    print("Saved: hypersonic.step, hypersonic.stl")
