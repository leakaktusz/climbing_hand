# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hold with shadow hands environment."""

import dm_env
import numpy as np
from absl import app, flags
from dm_control.mjcf import export_with_assets
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco import viewer as mujoco_viewer
from dm_control.mjcf import Physics
import mujoco


from robopianist import suite


from dm_control import composer, mjcf,viewer
from robopianist.suite.tasks.hold_grabber import HoldWithShadowHands

def _make_jug_task(**kwargs):
    return HoldWithShadowHands(gravity_compensation=False)



def main(_) -> None:
    task=_make_jug_task()
    env = composer.Environment(task)
    spec = env.action_spec()
    # Define a simple policy that drives actuators programmatically
    step_count = [0]  # mutable container so inner function can update

    def wiggle_policy(time_step):
        step_count[0] += 1
        alpha = 0.5 * (1 + np.sin(0.1 * step_count[0]))
        # Interpolate between min and max
        return spec.minimum + alpha * (spec.maximum - spec.minimum)
    def nothing_policy(time_step):
        return np.zeros(spec.shape,dtype=spec.dtype)

    # Launch the interactive viewer with your policy
    viewer.launch(environment_loader=lambda: env, policy=wiggle_policy)


if __name__ == "__main__":
    app.run(main)
