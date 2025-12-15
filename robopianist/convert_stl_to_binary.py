"""Convert ASCII STL files to binary STL format for MuJoCo."""

import numpy as np
from pathlib import Path
import struct

def read_ascii_stl(filename):
    """Read an ASCII STL file and return vertices and normals."""
    vertices = []
    normals = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_normal = None
    current_vertices = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('facet normal'):
            parts = line.split()
            current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
        elif line.startswith('vertex'):
            parts = line.split()
            current_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith('endfacet'):
            if len(current_vertices) == 3:
                normals.append(current_normal)
                vertices.extend(current_vertices)
            current_vertices = []
    
    return np.array(vertices).reshape(-1, 3, 3), np.array(normals)

def write_binary_stl(filename, faces, normals):
    """Write a binary STL file in proper format."""
    num_triangles = len(faces)
    
    with open(filename, 'wb') as f:
        # Header (80 bytes) - must be exactly 80 bytes
        header = b'Binary STL converted from ASCII' + b'\x00' * (80 - 31)
        f.write(header)
        
        # Number of triangles (4 bytes, little endian unsigned int)
        f.write(struct.pack('<I', num_triangles))
        
        # For each triangle: normal (3 floats) + 3 vertices (9 floats) + attribute (uint16)
        for i in range(num_triangles):
            # Pack as little-endian floats
            data = struct.pack('<fff', *normals[i])  # Normal
            for vertex in faces[i]:
                data += struct.pack('<fff', *vertex)  # Vertex
            data += struct.pack('<H', 0)  # Attribute byte count
            f.write(data)

def convert_stl(input_path, output_path=None):
    """Convert an ASCII STL file to binary format."""
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_binary.stl"
    else:
        output_path = Path(output_path)
    
    print(f"Reading ASCII STL from: {input_path}")
    faces, normals = read_ascii_stl(input_path)
    
    print(f"Found {len(faces)} triangles")
    print(f"Writing binary STL to: {output_path}")
    write_binary_stl(output_path, faces, normals)
    
    print("Conversion complete!")
    return output_path

if __name__ == "__main__":
    # Convert the appiglio.stl file
    stl_path = Path(__file__).parent / "robopianist" / "models" / "holds" / "meshes" / "appiglio.stl"
    
    if stl_path.exists():
        output_path = stl_path.parent / "appiglio_binary.stl"
        convert_stl(stl_path, output_path)
        print(f"\nNow update jug.py to use: {output_path.name}")
    else:
        print(f"File not found: {stl_path}")
