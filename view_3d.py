import pyvista as pv
import numpy as np
import os


def read_plot3d_multiblock_ascii(fname):
    with open(fname) as f:
        first = f.readline().split()
        nblk = int(first[0]) if len(first)==1 else 1
        if nblk == 1:
            ni,nj,nk = map(int, f.readline().split())
            size = ni*nj*nk
            x = np.fromfile(f, sep=' ', count=size).reshape((nk,nj,ni))
            y = np.fromfile(f, sep=' ', count=size).reshape((nk,nj,ni))
            z = np.fromfile(f, sep=' ', count=size).reshape((nk,nj,ni))
            return [(x,y,z)]
        dims=[tuple(map(int,f.readline().split())) for _ in range(nblk)]
        blocks=[]
        for (ni,nj,nk) in dims:
            size = ni*nj*nk
            x = np.fromfile(f, sep=' ', count=size).reshape((nk,nj,ni))
            y = np.fromfile(f, sep=' ', count=size).reshape((nk,nj,ni))
            z = np.fromfile(f, sep=' ', count=size).reshape((nk,nj,ni))
            blocks.append((x,y,z))
        return blocks

path = "./output"
model_name = "waverider"
# file = os.path.join(path, 'aircraft.xyz')
file = os.path.join(path, 'waverider.xyz')

blocks = read_plot3d_multiblock_ascii(file)

plotter = pv.Plotter(off_screen=True)
for x,y,z in blocks:
    surf = pv.StructuredGrid(x,y,z)
    plotter.add_mesh(surf, color='green', smooth_shading=True)
plotter.show_grid(xlabel='X', ylabel='Y', zlabel='Z')
plotter.show(screenshot=os.path.join(path, f"{model_name}_pyvista.png"))
