from OCP.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Shell
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing

# 构造一个简单面，比如一个平面
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

face = BRepBuilderAPI_MakeFace(gp_Ax2(gp_Pnt(0,0,0), gp_Dir(0,0,1))).Face()
shell = TopoDS_Shell()  # 空壳
sew = BRepBuilderAPI_Sewing(1e-3)
sew.Add(face)
sew.Perform()
print("Sewed shape:", sew.SewedShape())