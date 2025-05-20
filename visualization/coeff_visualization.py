import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

npy_file_path = 'paper_img/wavelet_example.npy'
data = np.load(npy_file_path)
 
if data.ndim != 3:
    raise ValueError("The input .npy file must be a 3D array.")
 
# data = []
shape = data.shape
 
grid = pv.StructuredGrid()
 
grid.dimensions = shape
 
grid.spacing = (1.0, 1.0, 1.0)  
 
x = np.arange(0, shape[0])
y = np.arange(0, shape[1])
z = np.arange(0, shape[2])
 
x, y, z = np.meshgrid(x, y, z)
 
points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
 
grid.points = points
 
grid.point_data['values'] = data.flatten()
 
slices = grid.slice_orthogonal()
cmap = plt.get_cmap("viridis", 100)
 
# Rotate the view by 90 degrees around the x-y plane
plotter = pv.Plotter(notebook=False,window_size=[1024, 1024])
for i, slice in enumerate(slices):
    slice.rotate_x(90, inplace=True)
    slice.rotate_y(90, inplace=True)
    slice.rotate_z(-90, inplace=True)
    if i == 0:
        _opa = 0.6
    else:
        _opa = 0.8
    plotter.add_mesh(slice, cmap=cmap, clim=[-1.48, -0.8], opacity=_opa)
# plotter.(False)
# plotter.view_vector([0.4, 0.2, 0.1])
plotter.view_vector([1, -0.3, 0.6])
light = pv.Light([0, -1, -1], color='orange', light_type='headlight')
plotter.add_light(light)
# actor.orientation = (0, 75, 135)
plotter.show(interactive=True) 