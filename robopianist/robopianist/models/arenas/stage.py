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

"""Suite arenas."""

from mujoco_utils import composer_utils


class Stage(composer_utils.Arena):
    """A custom arena with a ground plane, lights and a starry night sky."""

    def _build(self, name: str = "stage") -> None:
        super()._build(name=name)

        # Change free camera settings.
        self._mjcf_root.statistic.extent = 0.6
        self._mjcf_root.statistic.center = (0.2, 0.1, 0.1)  # Center between hand and jug
        getattr(self._mjcf_root.visual, "global").azimuth = 90  # View from right-front
        getattr(self._mjcf_root.visual, "global").elevation = -30  # Slightly above

        
        #self._mjcf_root.statistic.center = (0.2, 0, 0.3)
       # getattr(self._mjcf_root.visual, "global").azimuth = 180
        #getattr(self._mjcf_root.visual, "global").elevation = -50



        self._mjcf_root.visual.scale.forcewidth = 0.04
        self._mjcf_root.visual.scale.contactwidth = 0.2
        self._mjcf_root.visual.scale.contactheight = 0.03

        # Lights.
        self._mjcf_root.worldbody.add("light", pos=(0, 0, 1))
        self._mjcf_root.worldbody.add(
            "light", pos=(0.3, 0, 1), dir=(0, 0, -1), directional=False
        )

        # Dark checkerboard floor.
        self._mjcf_root.asset.add(
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            width=512,
            height=512,
            rgb1=[0.1, 0.1, 0.1],
            rgb2=[0.2, 0.2, 0.2],
        )
        self._mjcf_root.asset.add(
            "material",
            name="grid",
            texture="grid",
            texrepeat=(1, 1),
            texuniform=True,
            reflectance=0.2,
        )
        # Starry night sky.
        self._mjcf_root.asset.add(
            "texture",
            name="skybox",
            type="skybox",
            builtin="gradient",
            rgb1=[0.2, 0.2, 0.2],
            rgb2=[0.0, 0.0, 0.0],
            width=800,
            height=800,
            mark="random",
            markrgb=[1, 1, 1],
        )

        # Camera for hold_grabber environment (side view of hand and jug)
        # Jug is at approximately (-0.05, 0.0, 0.03), hand starts at (0.4, 0.15, 0.15)
        # Position camera to capture both
        # azimuth=180, elevation=-30 calculated to xyaxes
        self._mjcf_root.worldbody.add(
            "camera",
            name="hold_view",
            pos=(0.03, 0.5, 0.4),
            mode="fixed",
            xyaxes=(-1, 0, 0, 0.8, -0.5, 0.656)  # azimuth=180, elevation=-30
        )
        
