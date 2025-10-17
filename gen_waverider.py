# gen_waverider.py
import os, sys
import numpy as np

# ---- 参数（随便改形状） -----------------------------------------------
L = 10.0            # 机身长度（x 向）
beta_deg = 10.0     # 虚拟圆锥激波半角（度），越大越“肥”
thickness_max = 0.25 # 最大厚度（上表面相对下表面的距离缩放）
planform_taper = 0.15 # 尾缘收缩比（越大尾部越尖）
ni, nj = 301, 121    # 网格密度（弦向 x、展向 y）

# ---- 生成下表面：贴虚拟“锥形激波” -----------------------------------
def make_waverider_blocks(L, beta_deg, thickness_max, planform_taper, ni, nj):
    beta = np.deg2rad(beta_deg)
    x = np.linspace(0.001, L, ni)                   # 避免 x=0
    # 每个 x 截面的“激波圆截半径”，决定展向范围
    r = x * np.tan(beta)

    # 让尾部平面收缩（delta 翼感觉）
    taper = (1 - planform_taper) + planform_taper * (x / L)  # [1-planform_taper, 1]
    r_eff = r * taper

    # 规则网格
    X = np.repeat(x[None, :], nj, axis=0)                   # (nj, ni)
    y = np.linspace(-1.0, 1.0, nj)                          # 归一展向
    Y = (r_eff[None, :] * y[:, None])                       # (nj, ni)

    # 下表面：锥面圆截：y^2 + z^2 = r(x)^2 → z = -sqrt(r(x)^2 - y^2)
    Z_lower = -np.sqrt(np.maximum(r_eff[None, :]**2 - Y**2, 0.0))

    # 上表面：在下表面上加一个厚度（随 x 衰减）
    t_dist = thickness_max * (1 - (x / L)**0.7)             # 厚度沿弦向衰减
    Z_upper = Z_lower + t_dist[None, :]

    # 组装成 Plot3D 多块（每块 nk=1 的曲面）
    # Block 1: 上表面
    X1 = X[None, :, :] ; Y1 = Y[None, :, :] ; Z1 = Z_upper[None, :, :]
    # Block 2: 下表面
    X2 = X[None, :, :] ; Y2 = Y[None, :, :] ; Z2 = Z_lower[None, :, :]
    return [(X1, Y1, Z1), (X2, Y2, Z2)]  # 每块形状 (1, nj, ni)

def write_plot3d_multiblock_ascii(fname, blocks):
    with open(fname, 'w') as f:
        f.write(f"{len(blocks)}\n")
        for (X, Y, Z) in blocks:
            nk, nj, ni = X.shape
            f.write(f"{ni} {nj} {nk}\n")
        for (A,) in [(np.array([b[0]]),) for b in []]: pass   # 仅为结构对齐，无实际作用
        # Plot3D 顺序通常是先 x 全体、再 y 全体、后 z 全体
        for (X, Y, Z) in blocks:
            nk, nj, ni = X.shape
            f.write(" ".join(map(lambda v: f"{v:.8e}", X.flatten(order='C'))) + "\n")
            f.write(" ".join(map(lambda v: f"{v:.8e}", Y.flatten(order='C'))) + "\n")
            f.write(" ".join(map(lambda v: f"{v:.8e}", Z.flatten(order='C'))) + "\n")

if __name__ == "__main__":
    blocks = make_waverider_blocks(
        L=L, beta_deg=beta_deg, thickness_max=thickness_max,
        planform_taper=planform_taper, ni=ni, nj=nj
    )
    out_xyz = "output/waverider.xyz"
    write_plot3d_multiblock_ascii(out_xyz, blocks)
    print("Saved →", out_xyz)

    # 可选：转 IGES（需要 OCP / pythonocc-core）
    try:
        from cst_modeling.io import plot3d_to_igs
        plot3d_to_igs("waverider")
        print("Saved → waverider.igs")
    except Exception as e:
        print("[Skip IGES] 需要 OCP/pythonocc-core：", e)
