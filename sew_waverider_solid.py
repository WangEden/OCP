# waverider_close_and_export.py  —— pythonocc-core 版本：扇形尾盖 + 缝合导出
import os

# ---- Core imports (pythonocc-core 命名空间) ----
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Trsf
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopAbs import TopAbs_ShapeEnum, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Shell, topods
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_Transform, BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid
)
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.StlAPI import StlAPI_Writer

from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.GC import GC_MakeSegment
from OCC.Core.gp import gp_Pnt
from OCC.Core.GeomAbs import GeomAbs_C0
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
)
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopoDS import topods

TOL = 2e-3  # sewing 容差，可按需调到 5e-3

# ---------- 工具函数 ----------
def read_iges(path: str) -> TopoDS_Shape:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    r = IGESControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError(f"Read IGES failed: {path}")
    r.TransferRoots()
    s = r.OneShape()
    print(f"[read] {path}: IsNull={s.IsNull()} type={s.ShapeType()}")
    return s

def mirror_about_xz(shape: TopoDS_Shape) -> TopoDS_Shape:
    tr = gp_Trsf()
    tr.SetMirror(gp_Ax2(gp_Pnt(0,0,0), gp_Dir(0,1,0), gp_Dir(1,0,0)))  # 关于 XZ 平面镜像（Y 取反）
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()

def bbox_of_shape(shape):
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    return box.Get()  # xmin, ymin, zmin, xmax, ymax, zmax

def bbox_of_edge(edge):
    box = Bnd_Box()
    brepbndlib.Add(edge, box)
    return box.Get()

def edge_endpoints(edge):
    """返回两端点 (v_lo,p_lo), (v_hi,p_hi)，按 y 从小到大排序"""
    verts = []
    expv = TopExp_Explorer(edge, TopAbs_VERTEX)
    while expv.More():
        verts.append(topods.Vertex(expv.Current()))
        expv.Next()
    if len(verts) != 2:
        raise RuntimeError("Edge does not have exactly 2 vertices.")
    p1 = BRep_Tool.Pnt(verts[0])
    p2 = BRep_Tool.Pnt(verts[1])
    if p1.Y() <= p2.Y():
        return (verts[0], p1), (verts[1], p2)
    else:
        return (verts[1], p2), (verts[0], p1)

def find_trailing_edge(face, global_xmax, tol=1e-3):
    """找到 face 中 x 接近 global_xmax 且沿 y 方向跨度最大的那条“尾缘边”"""
    best = None
    best_span = -1.0
    expe = TopExp_Explorer(face, TopAbs_EDGE)
    while expe.More():
        e = topods.Edge(expe.Current())
        xmin, ymin, zmin, xmax, ymax, zmax = bbox_of_edge(e)
        if abs(xmax - global_xmax) < tol and (xmax - xmin) < tol * 10:
            span = ymax - ymin
            if span > best_span:
                best_span = span
                best = e
        expe.Next()
    return best

def _bbox_edge(edge):
    box=Bnd_Box(); brepbndlib.Add(edge, box); return box.Get()

def _edge_endpoints_sorted_by_y(edge):
    vs=[]; ex=TopExp_Explorer(edge, TopAbs_VERTEX)
    while ex.More():
        vs.append(topods.Vertex(ex.Current())); ex.Next()
    if len(vs)!=2: raise RuntimeError("edge without 2 vertices")
    p1=BRep_Tool.Pnt(vs[0]); p2=BRep_Tool.Pnt(vs[1])
    return ((vs[0],p1),(vs[1],p2)) if p1.Y()<=p2.Y() else ((vs[1],p2),(vs[0],p1))

def _find_trailing_edge(face, global_xmax, tol=1e-3):
    best=None; best_span=-1.0
    ex=TopExp_Explorer(face, TopAbs_EDGE)
    while ex.More():
        e=topods.Edge(ex.Current())
        xmin,ymin,zmin,xmax,ymax,zmax=_bbox_edge(e)
        if abs(xmax-global_xmax)<tol and (xmax-xmin)<tol*10:
            span=ymax-ymin
            if span>best_span:
                best_span=span; best=e
        ex.Next()
    return best

def _dist(p,q):
    return ((p.X()-q.X())**2 + (p.Y()-q.Y())**2 + (p.Z()-q.Z())**2) ** 0.5

def make_tail_cap_fan(upper_face, lower_face, tol_plane=1e-3, conn_tol=1e-6):
    # 1) 全局 xmax
    boxu=Bnd_Box(); brepbndlib.Add(upper_face, boxu)
    boxl=Bnd_Box(); brepbndlib.Add(lower_face, boxl)
    _,_,_,xu,_,_=boxu.Get(); _,_,_,xl,_,_=boxl.Get()
    global_xmax=max(xu,xl)

    # 2) 找各自的尾缘边
    e_up=_find_trailing_edge(upper_face, global_xmax, tol_plane)
    e_lo=_find_trailing_edge(lower_face, global_xmax, tol_plane)
    if e_up is None or e_lo is None:
        raise RuntimeError("未找到尾缘边，调大 tol_plane 再试")

    # 3) 端点按 y 从小到大
    (v_u0,p_u0),(v_u1,p_u1)=_edge_endpoints_sorted_by_y(e_up)
    (v_l0,p_l0),(v_l1,p_l1)=_edge_endpoints_sorted_by_y(e_lo)

    # 4) 构造连接边（长度>conn_tol 才创建）
    wire_maker=BRepBuilderAPI_MakeWire()
    wire_maker.Add(e_up)                    # 上尾缘
    if _dist(p_u1,p_l1) > conn_tol:         # 上端连接
        wire_maker.Add(BRepBuilderAPI_MakeEdge(p_u1,p_l1).Edge())
    # 下尾缘用反向，保证闭环走向
    wire_maker.Add(topods.Edge(e_lo.Reversed()))
    if _dist(p_u0,p_l0) > conn_tol:         # 下端连接
        wire_maker.Add(BRepBuilderAPI_MakeEdge(p_u0,p_l0).Edge())

    wire=wire_maker.Wire()

    # 5) 优先尝试平面面片
    try:
        face=BRepBuilderAPI_MakeFace(wire, True).Face()
        return face
    except Exception:
        pass  # 退路：剖分面

    # 6) 退路：在上、下尾缘之间生成“扇形”封面（规则曲面），不强制平面
    up_wire_m=BRepBuilderAPI_MakeWire(); up_wire_m.Add(e_up); up_wire=up_wire_m.Wire()
    lo_wire_m=BRepBuilderAPI_MakeWire(); lo_wire_m.Add(e_lo); lo_wire=lo_wire_m.Wire()

    mk=BRepOffsetAPI_ThruSections(False, True, conn_tol)  # (isSolid, isRuled, pres3d)
    mk.AddWire(up_wire)
    mk.AddWire(lo_wire)
    mk.Build()
    return mk.Shape()  # 返回的是一个面/壳；后续 sewing 时当作一块 surface 加入


def sew_faces_to_solid(shapes, tol=TOL) -> TopoDS_Shape:
    """sewing → shell → solid"""
    sew = BRepBuilderAPI_Sewing(tol)
    for s in shapes:
        sew.Add(s)
    sew.Perform()
    sewed = sew.SewedShape()
    print(f"[sew] type={sewed.ShapeType()}")

    # 把 sewing 后的所有 FACE 拼成一个 shell
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)

    expf = TopExp_Explorer(sewed, TopAbs_ShapeEnum.TopAbs_EDGE)  # 只是为了触发构建；马上遍历 FACE
    expf = TopExp_Explorer(sewed, TopAbs_ShapeEnum.TopAbs_FACE)
    faces = 0
    while expf.More():
        builder.Add(shell, topods.Face(expf.Current()))
        faces += 1
        expf.Next()
    print(f"[shell] faces={faces}")
    if faces == 0:
        raise RuntimeError("No faces after sewing.")

    mk = BRepBuilderAPI_MakeSolid()
    mk.Add(shell)
    solid = mk.Solid()
    print(f"[solid] type={solid.ShapeType()} (2=SOLID)")
    return solid

def mesh_and_write_stl(shape: TopoDS_Shape, stl_path: str, deflection=1.0, angle=0.8):
    BRepMesh_IncrementalMesh(shape, deflection, False, angle, True)
    w = StlAPI_Writer()
    w.SetASCIIMode(False)
    w.Write(shape, stl_path)
    print(f"[stl] {os.path.abspath(stl_path)} (exists={os.path.exists(stl_path)})")

# ---------- 主流程 ----------
if __name__ == "__main__":
    # 1) 读半机上下表面
    lower = read_iges("waverider/lower_surface.igs")
    upper = read_iges("waverider/upper_surface.igs")

    # 2) 基于尾缘自动生成“扇形尾盖”
    tail = make_tail_cap_fan(upper, lower, tol_plane=1e-3, conn_tol=1e-6)

    # 3) 镜像出另一半
    lower_m = mirror_about_xz(lower); upper_m = mirror_about_xz(upper); tail_m = mirror_about_xz(tail)
    upper_m = mirror_about_xz(upper)
    tail_m  = mirror_about_xz(tail)

    # 4) 缝合 → 实体
    solid = sew_faces_to_solid([upper, lower, tail, upper_m, lower_m, tail_m], tol=TOL)
    if solid.IsNull() or solid.ShapeType() != TopAbs_ShapeEnum.TopAbs_SOLID:
        raise RuntimeError("MakeSolid failed: non-solid (还缺侧边封口？可再补 y=min/max 处的扇形侧盖)")

    # 5) 导出 STEP / STL
    os.makedirs("waverider", exist_ok=True)
    step_path = "waverider/waverider_solid.step"
    w = STEPControl_Writer()
    w.Transfer(solid, STEPControl_AsIs)
    w.Write(step_path)
    print(f"[step] {os.path.abspath(step_path)}")

    mesh_and_write_stl(solid, "waverider/waverider_solid.stl", deflection=1.0, angle=0.8)
