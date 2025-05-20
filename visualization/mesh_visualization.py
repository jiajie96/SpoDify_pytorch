import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

light_blue = [0.6, 0.8, 1.0]  # RGB values for a lighter blue

# light = pv.Light([0, -1, -1], color='blue', light_type='headlight')
# light = pv.Light(light_type='headlight', intensity=0.5, position=(0, 1, 1))
light = pv.Light(
    position=(0, 0.2, 1.0),
    focal_point=(0, 0, 0),
    color=[1.0, 1.0, 0.9843, 1.0],  # Color temp. 5400 K
    # color = 'lightblue',
    intensity=0.2,
)
# shadow_pass = pv.ShadowMapPass(light)
plotter = pv.Plotter(window_size=(800, 800))
plotter.add_light(light)
# Define a lighter blue color
# Camera position for diagonal view
camera_position = [(-1, -1, 0.5), (0, 0, 0), (0, 0, 1)]  # [position, focus, view up]
mesh = pv.read('paper_img/model_normalized.obj')
# mesh.rotate_z(180, inplace=True)
mesh.rotate_x(90, inplace=True)
# mesh.rotate_y(180, inplace=True)
# Add a thin box below the mesh
bounds = mesh.bounds
print(bounds)
rnge = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

expand = 10
height = rnge[2] * 0.05
center = np.array(mesh.center)
center -= [0, 0, mesh.center[2] - bounds[4] + height / 2]

width = rnge[0] * (1 + expand)
length = rnge[1] * (1 + expand)
base_mesh = pv.Cube(center, width, length, height)

plotter.add_mesh(base_mesh, color='white', metallic=0.0, roughness=1)
# actor = plotter.add_mesh(mesh, color=light_blue)
# actor = plotter.add_mesh(mesh, color="lightblue", split_sharp_edges=True, 
#     # ambient=0.2,
#     # diffuse=0.5,
#     # specular=0.5,
#     # specular_power=90,
#     metallic=1.0, roughness=0.3,
#     smooth_shading=False,)

plotter.add_mesh(mesh, cmap="blue", split_sharp_edges=True, clim=[2, 5], metallic=0.5,  roughness=0.5)
# actor = plotter.add_mesh(mesh, color="lightblue", split_sharp_edges=True, pbr=True, metallic=1.0, roughness=0.3)
plotter.camera_position = camera_position
# actor.orientation = (90-180, 75-180, 135)
# light = pv.Light([0, -1, -1], color='blue', light_type='headlight')
# light = pv.Light(light_type='headlight', intensity=0.5, position=(0, 1, 1))
# plotter.add_light(light)
plotter.enable_shadows()
plotter.show()