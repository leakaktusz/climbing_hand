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
from dm_control import composer, mjcf
from pathlib import Path
import numpy as np
import struct


POS_WALL_AND_JUG=[-0.05, -0.0, 0.03]
POS_EASIER_JUG=[0.0, 0.15, 0.13]
POS_PINCH=[-0.02, 0.1, 0.12]
POS_SLOPER=[-0.04, 0.1, 0.07]
POS_LEDGE=[-0.02, 0.1, 0.09]
POS_CRIMP=[-0.04, 0.12, 0.11]
POS_SMALL_CRIMP=[-0.03, 0.12, 0.11]
POS_WEIRD_CRIMP=[-0.12, 0.27, 0.17]
# Simple Jug entity compatible with RoboPianist
class Jug(composer.Entity):
    def _build(self, name="jug"):
        self._mjcf_root = mjcf.RootElement(model=name)
        
        # Increase solver iterations for better contact resolution
        self._mjcf_root.option.iterations = 100  # Default is 50
        self._mjcf_root.option.ls_iterations = 100  # Line search iterations
        #Jug_1 appiglio_binary 
        # Blue_Forged_Jug_Pinch 
        # White_Ball_Sloper_Climbing_Hold
        # Purple_Cobble_Scoop_Climbing_Hold
        #Sample_Crimp_8 big
        #Sample_Crimp_1 small
        #Sample_Crimp_4 weird
        #strange_hold-V1
        mesh_file = Path(__file__).parent / "meshes" / "Sample_Crimp_8.stl"
        mesh_asset_name = "jug_mesh_asset"
        self._mjcf_root.asset.add(
            "mesh",
            name=mesh_asset_name,
            file=str(mesh_file.resolve()),
            scale=[0.001, 0.001, 0.001]# for jug
        )

        # Pre-load mesh vertices for closest-surface queries (reward shaping).
        # We assume STL format (binary or ASCII). Vertices stored in millimeters,
        # scale is applied (0.001) to convert to meters like in MJCF.
        self._vertices_local = self._load_stl_vertices(mesh_file) * 0.001  # apply scale
        # Cache for world transformed vertices per physics step
        self._cached_step = -1
        self._cached_world_vertices = None

        # Jug body
        self._body = self._mjcf_root.worldbody.add(
            "body",
            name="jug_body",
            pos=POS_SLOPER,
            euler=[0, np.pi/2, np.pi/2],  #JUG
            #euler=[0, np.pi/2, 0], #easier jug
            #euler=[0, np.pi/2, -np.pi/2],  # for crimp 1
            #euler=[-np.pi/2,np.pi - 2 * np.pi / 180, np.pi/2],  # for strange hold
        )

        # Jug mesh geom
        self._geom = self._body.add(
            "geom",
            name="jug_mesh_asset",
            type="mesh",
            mesh=mesh_asset_name,        
            rgba=[0.8, 0.8, 0.8, 1],
            friction=[5.0, 3.0, 3.0],       # very high friction - non-slippery surface
            condim=6,                         # 6D friction cone for full contact modeling
            contype=1,
            conaffinity=1,
            margin=0.002,                      # 2mm collision margin to catch contacts early
            solimp=[0.99999, 0.99999, 0.000001], # maximum stiffness (closest to 1.0)
            solref=[0.0001, 2.0],              # extremely stiff with heavy damping
        )
        self._wall = self._mjcf_root.worldbody.add(
            "geom",
            name="wall",
            type="box",
            size=[0.001, 0.3, 0.3],  # thickness, width, height
            pos=POS_WALL_AND_JUG,   # position in front of robot
            rgba=[0.0, 0.0, 0.0, 0.5],
            contype=1,
            conaffinity=1,
            solimp=[0.999, 0.999, 0.0001],   # extremely rigid
            solref=[0.001, 1.0],              # very stiff
            friction=[1.0, 0.1, 0.01],        # standard friction
        )       
    @property
    def mjcf_model(self):
        return self._mjcf_root

    def _load_stl_vertices(self, path: Path) -> np.ndarray:
        """Load STL vertices (ASCII or binary). Returns (N,3) array.
        Only unique vertices are kept to reduce per-step distance work.
        """
        if not path.exists():
            return np.zeros((0,3), dtype=np.float32)
        data = path.read_bytes()
        # Binary STL has an 80-byte header then uint32 triangle count.
        if len(data) > 84:
            tri_count = struct.unpack('<I', data[80:84])[0]
            expected_len = 84 + tri_count * 50
            if len(data) == expected_len:  # likely binary
                vertices = []
                offset = 84
                for _ in range(tri_count):
                    # skip normal (12 bytes)
                    offset += 12
                    for _v in range(3):
                        x,y,z = struct.unpack('<fff', data[offset:offset+12])
                        vertices.append((x,y,z))
                        offset += 12
                    # skip attribute byte count (2 bytes)
                    offset += 2
                arr = np.array(vertices, dtype=np.float32)
                # Deduplicate
                _, idx = np.unique(arr, axis=0, return_index=True)
                return arr[idx]
        # Fallback ASCII parse
        verts = []
        for line in path.read_text(errors='ignore').splitlines():
            line = line.strip()
            if line.startswith('vertex'):
                parts = line.split()
                if len(parts) == 4:
                    verts.append(tuple(float(p) for p in parts[1:]))
        arr = np.array(verts, dtype=np.float32)
        if arr.size == 0:
            return arr
        _, idx = np.unique(arr, axis=0, return_index=True)
        return arr[idx]

    def closest_surface_distance(self, physics, point_world: np.ndarray) -> float:
        """Compute distance from a world point to closest mesh vertex (approx surface).
        Transforms cached local vertices to world using body pose each step.
        """
        if self._vertices_local.size == 0:
            return 0.0
        step = physics.data.time
        # Recompute world vertices only when pose changes (use time as proxy).
        if step != self._cached_step:
            body_xpos = physics.bind(self._body).xpos
            body_xmat = physics.bind(self._body).xmat.reshape(3,3)
            self._cached_world_vertices = body_xpos + self._vertices_local @ body_xmat.T
            self._cached_step = step
        diffs = self._cached_world_vertices - point_world
        return float(np.min(np.linalg.norm(diffs, axis=1)))