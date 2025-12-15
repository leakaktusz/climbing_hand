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

"""A task where a shadow hands grabs a hold"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from dm_control import mjcf
from dm_control.composer import variation as base_variation
from dm_control.composer.observation import observable
from dm_control.mjcf import commit_defaults
from dm_control.utils.rewards import tolerance
from dm_env import specs
from mujoco_utils import collision_utils, spec_utils

import robopianist.models.hands.shadow_hand_constants as hand_consts
from robopianist.models.arenas import stage
from robopianist.suite import composite_reward
from robopianist.suite.tasks import base_for_climb




class HoldWithShadowHands(base_for_climb.JugTask):
    def __init__(
        self,
        randomize_hand_positions: bool = False,
        max_episode_steps: int = 100,
        disable_contact_reward: bool = False,
        energy_penalty_coef: float = 5e-3,
        **kwargs,
    ) -> None:
        """Task constructor.
        
        Args:
            max_episode_steps: Maximum timesteps before episode timeout.
            disable_contact_reward: If True, disables contact reward shaping.
            randomize_hand_positions: If True, randomizes initial hand position.
            energy_penalty_coef: Coefficient for energy penalty.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(arena=stage.Stage(), **kwargs)
        
        # Store configuration
        self._max_episode_steps = max_episode_steps
        self._disable_contact_reward = disable_contact_reward
        self._randomize_hand_positions = randomize_hand_positions
        self._energy_penalty_coef = energy_penalty_coef
        
        # Initialize episode state
        self._reset_quantities_at_episode_init()
        
        # Setup observables and rewards
        self._add_observables()
        self._set_rewards()

    def _reset_quantities_at_episode_init(self) -> None:
        """Reset internal state variables at episode start."""
        self._t_idx: int = 0
        self._should_terminate: bool = False
        self._failure_termination: bool = False
        self._discount: float = 1.0
        self._hand_height: float = 0.0
        self._initial_hand_height: float = 0.0
        self._hand_fell: bool = False

    def _set_rewards(self) -> None:
        """Configure the composite reward function."""
        self._reward_fn = composite_reward.CompositeReward( #plot rewards and penalties
            #grip_reward=self._compute_grip_reward, #no height, slope
            finger_proximity_reward=self._compute_finger_proximity_reward,
            survival_reward=self._compute_survival_reward,  # Reward for each timestep alive
            #height_reward=self._compute_height_reward, # reward fot the amount of steps it stays above the treshold (its hard for the algorithm to learn)   
            #contact_reward=self._compute_contact_reward,
            #energy_reward=self._compute_energy_reward,
        )

    def _add_observables(self) -> None:
        """Configure observables for the agent."""
        # Enable hand joint positions
        self.right_hand.observables.joints_pos.enabled = True
        
        # Add hand height observable
        def _get_hand_height(physics) -> np.ndarray:
            hand_pos = physics.bind(self.right_hand.root_body).xpos
            return np.array([hand_pos[2]], dtype=np.float64)
        
        height_obs = observable.Generic(_get_hand_height)
        height_obs.enabled = True
        
        # Add hand position observable
        def _get_hand_position(physics) -> np.ndarray:
            return physics.bind(self.right_hand.root_body).xpos.copy()
        
        pos_obs = observable.Generic(_get_hand_position)
        pos_obs.enabled = True
        
        # Add hold position observable (target)
        def _get_hold_position(physics) -> np.ndarray:
            jug_body = self.jug.mjcf_model.find('body', 'jug_body')
            return physics.bind(jug_body).xpos.copy()
        
        hold_pos_obs = observable.Generic(_get_hold_position)
        hold_pos_obs.enabled = True

        # Add fingertip-to-hold surface distance observable (per finger).
        # def _get_fingertip_distances(physics) -> np.ndarray:
        #     dists = []
        #     for site in self.right_hand.fingertip_sites:
        #         fingertip_pos = physics.bind(site).xpos
        #         d = self.jug.closest_surface_distance(physics, fingertip_pos)
        #         dists.append(d)
        #     return np.array(dists, dtype=np.float64)

        # fingertip_dist_obs = observable.Generic(_get_fingertip_distances)
        # fingertip_dist_obs.enabled = True
        
        self._task_observables = {
            "hand_height": height_obs,
            "hand_position": pos_obs,
            "hold_position": hold_pos_obs,
            # "fingertip_to_hold_distances": fingertip_dist_obs,
        }
        #for element in self._task_observables.values():
         #   print("Observable added:", element)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Initialize episode state and optionally randomize start position."""
        self._reset_quantities_at_episode_init()
        if self._randomize_hand_positions:
            self._randomize_initial_hand_positions(physics, random_state)

    def before_step(
            self, physics: mjcf.Physics, action: np.ndarray, random_state: np.random.RandomState,
        ) -> None:
            """Applies the control to the hands."""
            self.right_hand.apply_action(physics, action, random_state)
    
    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Update state and check termination conditions after physics step."""
        del random_state  # Unused
        
        # Increment timestep
        self._t_idx += 1
        
        # Get current hand height
        hand_pos = physics.bind(self.right_hand.root_body).xpos
        self._hand_height = hand_pos[2]
        
        # Get jug's lowest point to determine failure threshold
        jug_body = self.jug.mjcf_model.find('body', 'jug_body')
    
        jug_pos = physics.bind(jug_body).xpos
        jug_lowest_point = jug_pos[2]  # Jug z-position (lowest point)
        #print("Jug lowest point:", jug_lowest_point)
        # Check if hand fell below jug's lowest point
        self._hand_fell = self._hand_height < jug_lowest_point - 0.5  # 50cm margin
        
        # Check timeout
        if self._t_idx >= self._max_episode_steps:
            self._should_terminate = True
        
        # Set failure flag if hand fell
        if self._hand_fell:
            self._failure_termination = True

    def get_reward(self, physics: mjcf.Physics) -> float:
        """Returns a reward based on the current state of the physics."""
        return self._reward_fn.compute(physics)

    def _compute_grip_reward(self, physics) -> float:
        del physics  
        if self._hand_height > -0.1:
            return 1.0
        elif self._hand_height > -0.5: 
            # Linear interpolation from 0.0 at -0.05 to 1.0 at 0.05
            ratio = (self._hand_height - (-0.5)) / (-0.1 - (-0.5))
            return ratio
        return 0.0

    def _compute_survival_reward(self, physics: mjcf.Physics) -> float:
        """Reward for each timestep the hand stays above failure threshold."""
        del physics  # Unused
        # Give small constant reward for survival each step
        # Over 1000 steps, this accumulates to significant total reward
        return 0.01 if not self._hand_fell else -0.01

    def _compute_height_reward(self, physics: mjcf.Physics) -> float:
        """Reward for keeping hand at target height."""
        del physics  # Unused
        # Smooth reward using tolerance function
        return tolerance(
            self._hand_height,
            bounds=(0.10, 0.5),  # Target range
            margin=0.1,
            sigmoid="gaussian",
        )

    def _compute_contact_reward(self, physics: mjcf.Physics) -> float:
        """Reward based on actual physical contacts between hand and jug."""
        contact_count = 0
        
        # Check all active contacts in the simulation
        if physics.data.ncon > 0:
            for i in range(physics.data.ncon):
                contact = physics.data.contact[i]
                
                # Get geometry names involved in this contact
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                
                if geom1_id >= 0 and geom2_id >= 0:
                    geom1_name = physics.model.id2name(geom1_id, 'geom')
                    geom2_name = physics.model.id2name(geom2_id, 'geom')
                    
                    # Check if contact involves jug and any hand geom
                    involves_jug = ('jug' in str(geom1_name).lower() or 
                                   'jug' in str(geom2_name).lower())
                    involves_hand = ('hand' in str(geom1_name).lower() or 
                                    'hand' in str(geom2_name).lower() or
                                    'finger' in str(geom1_name).lower() or 
                                    'finger' in str(geom2_name).lower())
                    
                    if involves_jug and involves_hand:
                        contact_count += 1
        
        # Normalize by a reasonable max number of contacts (e.g., 10)
        # Returns value between 0 and 1
        return contact_count

    def _compute_finger_proximity_reward(self, physics: mjcf.Physics) -> float:
        """Dense reward: encourage fingertips to move close to the hold.

        Uses per-finger smooth shaping based on Euclidean distance between
        fingertip site position and jug position. A tolerance-shaped curve
        gives high reward inside a small radius (contact vicinity) while still
        providing gradient up to a larger margin so the policy can learn
        approach behaviors before touching.
        """
        per_finger_values = []
        for site in self.right_hand.fingertip_sites:
            fingertip_pos = physics.bind(site).xpos.copy()
            d = self.jug.closest_surface_distance(physics, fingertip_pos)
            #print(d)
            # Reward high inside 5mm shell, gradient out to 5cm.
            # tolerance() returns 1.0 when distance is within bounds [0, 5mm]
            # and decays toward 0 as distance increases to margin (5cm)
            r = tolerance(
                d,
                bounds=(0.0, 0.005),  # treat <5mm as optimal surface proximity
                margin=0.05,          # learning signal until 5cm away
                sigmoid="gaussian",   # sharper near-surface emphasis
            )
            per_finger_values.append(r / self._max_episode_steps)

        if not per_finger_values:
            return 0.0
        return float(np.mean(per_finger_values))

    def get_discount(self, physics: mjcf.Physics) -> float:
        """Returns the discount based on the current state of the physics."""
        return self._discount

    def _compute_energy_reward(self, physics: mjcf.Physics) -> float:
        """Penalty for excessive actuator usage."""
        power = self.right_hand.observables.actuators_power(physics).copy()
        return -self._energy_penalty_coef * np.sum(power)

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        """Check if episode should terminate."""
        
        # Terminate on success (reached max steps without falling)
        if self._should_terminate and not self._failure_termination:
            self._discount = 1.0  # Mark as success
            return True  # discount stays 1.0 (success)
        
        # Terminate on failure (hand fell)
        if self._failure_termination:
            self._discount = 0.0  # Mark as failure
            return True
        
        return False     

    
    @property
    def task_observables(self):
        """Return task-specific observables."""
        return self._task_observables

    @property
    def reward_fn(self) -> composite_reward.CompositeReward:
        """Return the composite reward function."""
        return self._reward_fn

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        """Returns the action spec for the task."""
        return self.right_hand.action_spec(physics)

    def _randomize_initial_hand_positions(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Randomize the initial position of the hand."""
        if not self._randomize_hand_positions:
            return
        offset_y = random_state.uniform(low=-0.05, high=0.05)
        offset_z = random_state.uniform(low=-0.05, high=0.05)
        self.right_hand.shift_pose(physics, (0, offset_y, offset_z))
