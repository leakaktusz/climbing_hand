import dm_env
from dm_control import mjcf, viewer
from pathlib import Path
import numpy as np

model = mjcf.RootElement()
model.asset.add('mesh', name='hold_1', file=str(Path(__file__).parent / "meshes" / "Jug_1.STL"))
model.worldbody.add('geom', type='plane', size=[2, 2, 0.1])
wall = model.worldbody.add('body', pos=[0,0,1])
wall.add('geom', type='mesh', mesh='hold_1', size=[0.2,0.2,0.2], pos=[0.05,0,0])

physics = mjcf.Physics.from_mjcf_model(model)

# Minimal dummy environment
class DummyEnv(dm_env.Environment):
    def __init__(self, physics):
        self.physics = physics
    def reset(self):
        return dm_env.restart(observation={})
    def step(self, action):
        return dm_env.transition(reward=0.0, observation={}, discount=1.0)
    def action_spec(self):
        return dm_env.specs.BoundedArray(shape=(0,), dtype=np.float32, minimum=[], maximum=[])
    
    def observation_spec(self):
        return {}

env = DummyEnv(physics)

# Launch viewer
viewer.launch(env)