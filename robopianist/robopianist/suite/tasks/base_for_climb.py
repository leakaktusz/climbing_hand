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

"""Base hold composer task."""

from typing import Sequence

import mujoco
import numpy as np
from dm_control import composer
from mujoco_utils import composer_utils, physics_utils

from robopianist.models.hands import HandSide, shadow_hand
from robopianist.models.holds.jug import Jug

# Timestep of the physics simulation, in seconds.
_PHYSICS_TIMESTEP = 0.005

# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP = 0.05  # 20 Hz.

# Default position and orientation of the hands.
_LEFT_HAND_POSITION = (0.4, -0.15, 0.13)
_LEFT_HAND_QUATERNION = (-1, -1, 1, 1)
_RIGHT_HAND_POSITION = (0.4, 0.1, 0.15)
_RIGHT_HAND_QUATERNION = (-1, -1, 1, 1)

_ATTACHMENT_YAW = 0  # Degrees.


class JugOnlyTask(composer.Task):
    """Jug task with no hands."""

    def __init__(
        self,
        arena: composer_utils.Arena,
        change_color_on_activation: bool = False,
       # add_jug_actuators: bool = False,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
    ) -> None:
        self._arena = arena
        self._jug = Jug()
        arena.attach(self._jug)

        # Harden the piano keys.
        # The default solref parameters are (0.02, 1). In particular, the first
        # parameter specifies -stiffness, and so decreasing it makes the contacts
        # harder. The documentation recommends keeping the stiffness at least 2x larger
        # than the physics timestep, see:
        # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=stiffness#solver-parameters
       # self._piano.mjcf_model.default.geom.solref = (physics_timestep * 2, 1)

        self.set_timesteps(
            control_timestep=control_timestep, physics_timestep=physics_timestep
        )

    # Accessors.

    @property
    def root_entity(self):
        return self._arena

    @property
    def arena(self):
        return self._arena

    @property
    def jug(self) -> Jug:
        return self._jug

    # Composer methods.

    def get_reward(self, physics) -> float:
        del physics  # Unused.
        return 0.0


class JugTask(JugOnlyTask):
    """Base class for jug tasks."""

    def __init__(
        self,
        arena: composer_utils.Arena,
        gravity_compensation: bool = False,
        primitive_fingertip_collisions: bool = False,
        reduced_action_space: bool = False,
        attachment_yaw: float = 0.0,
        forearm_dofs: Sequence[str] = shadow_hand._DEFAULT_FOREARM_DOFS,

    ) -> None:
        super().__init__(
            arena=arena,
        )

        self._right_hand = self._add_hand(
            hand_side=HandSide.RIGHT,
            position=_RIGHT_HAND_POSITION,
            quaternion=_RIGHT_HAND_QUATERNION,
            gravity_compensation=gravity_compensation,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
            attachment_yaw=attachment_yaw,
            forearm_dofs=forearm_dofs,
        )
    

    # Accessors.

    @property
    def left_hand(self) -> shadow_hand.ShadowHand:
        return self._left_hand

    @property
    def right_hand(self) -> shadow_hand.ShadowHand:
        return self._right_hand

    # Helper methods.

    def _add_hand(
        self,
        hand_side: HandSide,
        position,
        quaternion,
        gravity_compensation: bool,
        primitive_fingertip_collisions: bool,
        reduced_action_space: bool,
        attachment_yaw: float,
        forearm_dofs: Sequence[str],
    ) -> shadow_hand.ShadowHand:
        joint_range = [-0.2, 0.2] 
        # Offset the joint range by the hand's initial position.
        joint_range[0] -= position[1]
        joint_range[1] -= position[1]

        hand = shadow_hand.ShadowHand(
            side=hand_side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            restrict_wrist_yaw_range=False,
            reduced_action_space=reduced_action_space,
            forearm_dofs=forearm_dofs,
        )
        hand.root_body.pos = position

        # Slightly rotate the forearms inwards (Z-axis) to mimic human posture.
        rotate_axis = np.asarray([0, 0, 1], dtype=np.float64)
        rotate_by = np.zeros(4, dtype=np.float64)
        sign = -1 if hand_side == HandSide.LEFT else 1
        angle = np.radians(sign * attachment_yaw)
        mujoco.mju_axisAngle2Quat(rotate_by, rotate_axis, angle)
        final_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(final_quaternion, rotate_by, quaternion)
        hand.root_body.quat = final_quaternion

        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)

        # Override forearm translation joint range.
        forearm_tx_joint = hand.mjcf_model.find("joint", "forearm_tx")
        if forearm_tx_joint is not None:
            forearm_tx_joint.range = joint_range
        forearm_tx_actuator = hand.mjcf_model.find("actuator", "forearm_tx")
        if forearm_tx_actuator is not None:
            forearm_tx_actuator.ctrlrange = joint_range
        
        # Attach the hand to the arena using the composer API
        wrapper_body = self._arena.attach(hand)
        for body in self._arena.mjcf_model.find_all("body"):
            if body.tag != "body":
                continue

            # Some internal wrappers have tag="body" but STILL no .name
            if not hasattr(body, "name"):
                continue
            if body.tag == "body":  # ensures this is a real body element
                print(
                    body.name,
                    [(j.name, j.type) for j in body.joint]
                )
        # Add freejoint to hand_root body after attachment (now it's at top level in arena)
        rh_shadow_hand= wrapper_body 
        if rh_shadow_hand is not None:
            rh_shadow_hand.add(
                        "inertial",
                        mass="0.0001",
                        diaginertia="1e-6 1e-6 1e-6",
                        pos="0 0 0"
            )
            rh_shadow_hand.add("freejoint", name="hand_root_free")
        for body in self._arena.mjcf_model.find_all("body"):
            if body.tag != "body":
                continue

            # Some internal wrappers have tag="body" but STILL no .name
            if not hasattr(body, "name"):
                continue
            if body.tag == "body":  # ensures this is a real body element
                print(
                    body.name,
                    [(j.name, j.type) for j in body.joint]
                )
        return hand
