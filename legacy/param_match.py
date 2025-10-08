import open3d as o3d
import numpy as np


# 用 RANSAC 拟合平面
def fit_plane_ransac(pcd: o3d.geometry.PointCloud, distance_thresh=1e-3, ransac_n=3, num_iters=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_thresh,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iters)
    # ax + by + cz + d = 0
    a, b, c, d = plane_model
    normal = np.array([a,b,c])
    normal = normal / np.linalg.norm(normal)
    return dict(a=a,b=b,c=c,d=d, normal=normal, inliers=np.array(inliers))


from scipy.optimize import least_squares

# 用最小二乘法拟合球面
def fit_sphere_leastsq(points: np.ndarray):
    # 初值：用点云质心 & 平均半径
    c0 = points.mean(axis=0)
    r0 = np.mean(np.linalg.norm(points - c0, axis=1))

    def residuals(x):
        cx, cy, cz, r = x
        return np.linalg.norm(points - np.array([cx,cy,cz]), axis=1) - r

    x0 = np.r_[c0, r0]
    res = least_squares(residuals, x0)
    cx, cy, cz, r = res.x
    return dict(center=np.array([cx,cy,cz]), radius=float(abs(r)))