import os
from dm_control import mjcf
from dm_control import viewer
import numpy as np

# Create the root MJCF model
model = mjcf.RootElement()
model.option.gravity = [0, 0, -9.81]

# Add a default material
model.asset.add(
    'texture',
    name='sky',
    type='skybox',
    builtin='gradient',
    width=512,
    height=512,
    rgb1=[0.2, 0.3, 0.4],
    rgb2=[0, 0, 0],
)
model.asset.add('material', name='hold_material', rgba=[0.8, 0.3, 0.2, 1])

model.asset.add('mesh', name='hold_1', file=os.path.join(os.path.dirname(__file__), 'meshes', 'Jug_1.STL'))
# Add a ground plane and climbing wall
world = model.worldbody
world.add('geom', type='plane', size=[2, 2, 0.1], rgba=[0.7, 0.7, 0.7, 1])
wall = world.add('body', name='wall', pos=[0, 0, 1])
wall.add('geom', type='box', size=[0.01, 1.0, 1.0], pos=[0, 0, 0], rgba=[0.6, 0.6, 0.6, 1])

# Define function to add holds
def add_hold(parent, hold_type, pos, color):
    if hold_type == "jug":
        parent.add('geom', type='mesh', mesh='hold_1', pos=pos, rgba=color)
    elif hold_type == "crimp":
        parent.add('geom', type='box', size=[0.06, 0.01, 0.015], pos=pos, rgba=color)
    elif hold_type == "sloper":
        parent.add('geom', type='mesh', mesh='hold_1',size=[0.3, 0.3, 0.3], pos=pos, rgba=color)
    elif hold_type == "pinch":
        parent.add('geom', type='capsule', fromto=[pos[0]-0.03, pos[1], pos[2],
                                                   pos[0]+0.03, pos[1], pos[2]],
                   size=[0.02], rgba=color)
    elif hold_type == "pocket":
        parent.add('geom', type='ellipsoid', size=[0.05, 0.04, 0.02], pos=pos, rgba=color)
        parent.add('geom', type='sphere', size=[0.03], pos=[pos[0], pos[1], pos[2]+0.02], rgba=[0, 0, 0, 1])
    else:
        print(f"Unknown hold type: {hold_type}")

# Create some holds on the wall
hold_types = ["jug", "crimp", "sloper", "pinch", "pocket"]
colors = [
    [0.9, 0.2, 0.2, 1],
    [0.2, 0.9, 0.2, 1],
    [0.2, 0.2, 0.9, 1],
    [0.9, 0.9, 0.2, 1],
    [0.9, 0.2, 0.9, 1]
]

for i, htype in enumerate(hold_types):
    add_hold(wall, htype, pos=[0.02, 0.0, -0.8 + i * 0.4], color=colors[i])

# Add light and camera
world.add('light', pos=[0.3, -0.3, 2.0], dir=[-1, 1, -1])
world.add('camera', name='cam', pos=[1.5, 0, 1], zaxis=[-1, 0, 0])

# Create physics and visualize
from dm_control import mjcf
import mujoco
from mujoco import viewer as mj_viewer

# (rest of your code that builds `model`)
physics = mjcf.Physics.from_mjcf_model(model)

# ✅ Correct viewer for static visualization
print("Launching MuJoCo viewer (no control loop)...")
mj_viewer.launch(physics.model.ptr, physics.data.ptr)

