# 基本几何体

import os, json, math, copy
import numpy as np
import open3d as o3d


class BaseGeometry:
    kind = "base"
    def to_open3d(self): raise NotImplementedError
    def to_dict(self): return {"kind": self.kind}


class Cube(BaseGeometry):
    """使用有向包围盒拟合 -> 近似长方体；如果三边近等，可视为正方体"""
    kind = "cube"
    def __init__(self, center, extents, R):
        self.center = np.asarray(center, float)
        self.extents = np.asarray(extents, float)  # 长宽高
        self.R = np.asarray(R, float)              # 3x3 旋转

    @staticmethod
    def from_geometry(geo):
        pts = _points_of(geo)
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts))
        return Cube(obb.center, obb.extent, obb.R)

    def to_open3d(self):
        m = o3d.geometry.TriangleMesh.create_box(*self.extents)
        m.compute_vertex_normals()
        m.rotate(self.R, center=(0,0,0))
        m.translate(self.center - self.R @ (self.extents/2.0))
        return m

    def to_dict(self):
        return {"kind": self.kind, "center": self.center.tolist(),
                "extents": self.extents.tolist(), "R": self.R.tolist()}
    

class Sphere(BaseGeometry):
    """用最小二乘法拟合球面"""
    kind = "sphere"
    def __init__(self, center, radius):
        self.center = np.asarray(center, float); self.radius = float(radius)

    @staticmethod
    def from_geometry(geo):
        pts = _points_of(geo)
        c, r = _fit_sphere_least_squares(pts)
        return Sphere(c, r)

    def to_open3d(self, resolution=30):
        m = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius, resolution=resolution)
        m.compute_vertex_normals(); m.translate(self.center)
        return m

    def to_dict(self):
        return {"kind": self.kind, "center": self.center.tolist(), "radius": self.radius}
    

class Cylinder(BaseGeometry):
    """用 PCA 拟合轴线；半径=到轴线的中位距离；高度=投影长度"""
    kind = "cylinder"
    def __init__(self, center, axis, radius, height):
        self.center = np.asarray(center, float)   # 几何中心（轴线中点）
        self.axis = _unit(axis)                   # 轴方向单位向量
        self.radius = float(radius)
        self.height = float(height)

    @staticmethod
    def from_geometry(geo):
        pts = _points_of(geo)
        c, axis = _pca_axis(pts)
        # 投影到轴方向
        t = (pts - c) @ axis
        h = t.max() - t.min()
        mid = c + axis * (t.min() + h/2.0)
        # 半径：点到轴线的距离中位数
        radial = np.linalg.norm(np.cross(pts - c, axis), axis=1)
        r = np.median(radial)
        return Cylinder(mid, axis, r, h)

    def to_open3d(self, resolution=64, split=4):
        m = o3d.geometry.TriangleMesh.create_cylinder(radius=self.radius, height=self.height,
                                                       resolution=resolution, split=split)
        m.compute_vertex_normals()
        # 默认 Z 轴为轴线，构造把Z旋到axis的旋转矩阵
        z = np.array([0,0,1.0]); v = np.cross(z, self.axis)
        c = np.dot(z, self.axis)
        if np.linalg.norm(v) < 1e-8 and c > 0.999999:
            R = np.eye(3)
        elif np.linalg.norm(v) < 1e-8 and c < -0.999999:
            R = _rot_matrix(np.array([1,0,0]), math.pi)
        else:
            s = np.linalg.norm(v)
            vx = _skew(v/s)
            R = np.eye(3)*c + vx + (1-c)*np.outer(self.axis, z)  # 直接用罗德里格公式更安全
            # 更稳：罗德里格
            k = v/np.linalg.norm(v)
            theta = math.acos(np.clip(c,-1,1))
            R = _rot_rodrigues(k, theta)
        m.rotate(R, center=(0,0,0))
        m.translate(self.center - self.axis*(self.height/2.0))
        return m

    def to_dict(self):
        return {"kind": self.kind, "center": self.center.tolist(),
                "axis": self.axis.tolist(), "radius": self.radius, "height": self.height}
    
class TriangularPrism(BaseGeometry):
    """
    三棱柱：用 PCA 得轴，截面在正交平面投影后取2D凸包，选3个极点近似三角形。
    参数：中心、轴、高度、三角形三个顶点（在局部截面坐标系）。
    """
    kind = "triangular_prism"
    def __init__(self, center, axis, height, tri2d_pts, basis_u, basis_v):
        self.center = np.asarray(center, float)
        self.axis = _unit(axis)
        self.height = float(height)
        self.tri2d_pts = np.asarray(tri2d_pts, float).reshape(3,2)  # 截面三角形
        self.u = _unit(basis_u); self.v = _unit(basis_v)            # 截面平面基

    @staticmethod
    def from_geometry(geo):
        pts = _points_of(geo)
        c, axis = _pca_axis(pts)
        # 轴向投影
        t = (pts - c) @ axis
        h = t.max() - t.min()
        mid = c + axis * (t.min() + h/2.0)
        # 截面基（与axis正交）
        u = _orthonormal(axis)
        v = np.cross(axis, u)
        # 投影到(u,v)平面
        uv = np.c_[ (pts - mid) @ u, (pts - mid) @ v ]
        hull = _convex_hull_2d(uv)
        tri = _pick_triangle_from_hull(hull)  # 选三个分布最广的点
        return TriangularPrism(mid, axis, h, tri, u, v)

    def to_open3d(self):
        # 构造局部三角形两个端面 + 侧面
        tri = self.tri2d_pts
        # 两端面中心沿轴 +/- h/2
        c0 = self.center - self.axis*(self.height/2.0)
        c1 = self.center + self.axis*(self.height/2.0)
        # 2D->3D
        def P(c, uv): return c + self.u*uv[0] + self.v*uv[1]
        v0 = [P(c0, tri[i]) for i in range(3)]
        v1 = [P(c1, tri[i]) for i in range(3)]
        verts = np.array(v0+v1, float)
        # 面：两个端面(0,1,2)和(3,4,5)，以及三条长方面
        faces = np.array([
            [0,1,2], [5,4,3],             # 注意第二个端面反向
            [0,1,4], [0,4,3],
            [1,2,5], [1,5,4],
            [2,0,3], [2,3,5],
        ], int)
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                      o3d.utility.Vector3iVector(faces))
        m.compute_vertex_normals()
        return m

    def to_dict(self):
        return {"kind": self.kind, "center": self.center.tolist(),
                "axis": self.axis.tolist(), "height": self.height,
                "tri2d_pts": self.tri2d_pts.tolist(),
                "basis_u": self.u.tolist(), "basis_v": self.v.tolist()}
    

# ------------------- 辅助函数 -------------------
def _points_of(geo, nsamp=4000):
    if isinstance(geo, o3d.geometry.TriangleMesh):
        return np.asarray(geo.sample_points_uniformly(min(nsamp, max(len(geo.vertices)*3, nsamp))).points)
    elif isinstance(geo, o3d.geometry.PointCloud):
        pts = np.asarray(geo.points)
        if len(pts) > nsamp:  # 采样减负
            idx = np.random.choice(len(pts), nsamp, replace=False)
            pts = pts[idx]
        return pts
    else:
        raise TypeError(type(geo))

def _fit_sphere_least_squares(pts):
    # |x-c|^2 = r^2 -> x^2 - 2c·x + c·c - r^2 = 0 -> 线性化解c与r
    A = np.c_[2*pts, np.ones(len(pts))]
    b = (pts**2).sum(axis=1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3]; r = math.sqrt(max(x[3] + (c@c), 0.0))
    return c, r

def _pca_axis(pts):
    c = pts.mean(axis=0)
    X = pts - c
    C = X.T @ X / len(X)
    w, V = np.linalg.eigh(C)
    axis = V[:, np.argmax(w)]
    return c, _unit(axis)

def _unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v/n if n > 0 else v

def _orthonormal(n):
    n = _unit(n)
    # 取与n不共线的一向量
    a = np.array([1,0,0]) if abs(n[0])<0.8 else np.array([0,1,0])
    u = a - n*np.dot(a,n)
    return _unit(u)

def _skew(k):
    kx,ky,kz = k
    return np.array([[0,-kz,ky],[kz,0,-kx],[-ky,kx,0]], float)

def _rot_rodrigues(k, theta):
    k = _unit(k)
    K = _skew(k)
    return np.eye(3) + math.sin(theta)*K + (1-math.cos(theta))*(K@K)

def _rot_matrix(axis, theta):
    axis = _unit(axis)
    a = math.cos(theta/2.0)
    b,c,d = -axis*math.sin(theta/2.0)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]], float)

def _convex_hull_2d(pts):
    # 简单 Andrew 单调链
    P = np.unique(np.round(pts, 8), axis=0)
    if len(P) <= 3: return P
    P = P[np.lexsort((P[:,1], P[:,0]))]
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    L=[]
    for p in P:
        while len(L)>=2 and cross(L[-2], L[-1], p) <= 0: L.pop()
        L.append(p)
    U=[]
    for p in P[::-1]:
        while len(U)>=2 and cross(U[-2], U[-1], p) <= 0: U.pop()
        U.append(p)
    return np.array(L[:-1]+U[:-1])

def _pick_triangle_from_hull(hull):
    if len(hull) < 3:
        # 退化：返回等边小三角
        r = 1.0
        return np.array([[r,0], [-r/2, r*math.sqrt(3)/2], [-r/2, -r*math.sqrt(3)/2]])
    # 取三点使得周长最大（O(n^3) 小 n 可接受）
    n = len(hull)
    best = None; best_peri = -1
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                peri = np.linalg.norm(hull[i]-hull[j]) + np.linalg.norm(hull[j]-hull[k]) + np.linalg.norm(hull[k]-hull[i])
                if peri > best_peri:
                    best_peri = peri; best = np.vstack([hull[i], hull[j], hull[k]])
    return best

# ---------- segmentation ----------
def split_triangle_mesh(mesh: o3d.geometry.TriangleMesh):
    labels, tri_counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    parts = []
    for lab in np.unique(labels):
        tri_idx = np.where(labels==lab)[0]
        sub = copy.deepcopy(mesh)
        mask = np.ones(len(sub.triangles), dtype=bool)
        mask[tri_idx] = False   # 保留tri_idx
        sub.remove_triangles_by_mask(mask)
        sub.remove_unreferenced_vertices()
        sub.compute_vertex_normals()
        if len(sub.triangles) > 0 and len(sub.vertices) > 0:
            parts.append(sub)
    return parts

def split_point_cloud(pcd: o3d.geometry.PointCloud, eps=2.0, min_points=50):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return [pcd]
    parts=[]
    for lab in np.unique(labels[labels>=0]):
        idx = np.where(labels==lab)[0]
        sub = pcd.select_by_index(idx)
        parts.append(sub)
    return parts

# ---------- classification & fitting ----------
def fit_primitive(geo):
    # 简单基于“曲率/形状”启发式：先尝试球；再圆柱；再立方体；最后三棱柱
    # 更稳健的做法是用法向直方图/主曲率阈值做投票，这里给出轻量可用版本
    pts = _points_of(geo)
    # 球：点到最佳球面距离的中位数占半径的比例很小则判为球
    c, r = _fit_sphere_least_squares(pts)
    sphere_resid = np.median(np.abs(np.linalg.norm(pts - c, axis=1) - r)) / max(r, 1e-6)

    # 圆柱：使用 PCA 轴，径向距离的 MAD/中位径向
    cc, axis = _pca_axis(pts)
    radial = np.linalg.norm(np.cross(pts - cc, axis), axis=1)
    r_med = np.median(radial)
    cyl_resid = np.median(np.abs(radial - r_med)) / max(r_med, 1e-6)

    if sphere_resid < 0.02:
        return Sphere.from_geometry(geo)
    if cyl_resid < 0.05:
        return Cylinder.from_geometry(geo)

    # 立方体 vs 三棱柱：比较投影截面（u,v）上凸包边数，边数≈4/3
    c0, ax = _pca_axis(pts)
    u = _orthonormal(ax); v = np.cross(ax, u)
    uv = np.c_[ (pts - c0) @ u, (pts - c0) @ v ]
    hull = _convex_hull_2d(uv)
    if len(hull) >= 4:
        return Cube.from_geometry(geo)
    else:
        return TriangularPrism.from_geometry(geo)

# ---------- pipeline ----------
def split_fit_and_export(geo, out_dir="out_parts", base="part"):
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(geo, o3d.geometry.TriangleMesh):
        parts = split_triangle_mesh(geo)
        save = lambda g, pth: o3d.io.write_triangle_mesh(pth, g)
        ext = ".stl"
    elif isinstance(geo, o3d.geometry.PointCloud):
        parts = split_point_cloud(geo)
        save = lambda g, pth: o3d.io.write_point_cloud(pth, g)
        ext = ".ply"
    else:
        raise TypeError(type(geo))

    fits = []
    for i, g in enumerate(parts):
        path_i = os.path.join(out_dir, f"{base}_{i}{ext}")
        save(g, path_i)
        prim = fit_primitive(g)
        fits.append(prim.to_dict())
        # 同时导出参数化重建网格做校验
        recon = prim.to_open3d()
        recon_path = os.path.join(out_dir, f"{base}_{i}_recon.stl")
        o3d.io.write_triangle_mesh(recon_path, recon)
        print(f"[OK] saved: {path_i}, recon: {recon_path}, kind={prim.kind}")

    # 存参数
    with open(os.path.join(out_dir, f"{base}_params.json"), "w", encoding="utf-8") as f:
        json.dump(fits, f, ensure_ascii=False, indent=2)
    return parts, fits


def main():
    pass

if __name__ == "__main__":
    main()