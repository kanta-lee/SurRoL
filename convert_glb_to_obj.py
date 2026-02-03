#!/usr/bin/env python3
"""
Convert GLB file to OBJ format for PyBullet compatibility.
"""
import os
import sys

try:
    import trimesh
except ImportError:
    print("trimesh is not installed. Installing...")
    os.system(f"{sys.executable} -m pip install trimesh --quiet")
    import trimesh

def convert_glb_to_obj(glb_path, obj_path=None):
    """Convert GLB file to OBJ format."""
    if obj_path is None:
        obj_path = glb_path.replace('.glb', '.obj')
    
    print(f"Loading GLB file: {glb_path}")
    mesh = trimesh.load(glb_path, force='mesh')
    
    # If it's a scene with multiple meshes, combine them
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    
    print(f"Exporting OBJ file: {obj_path}")
    mesh.export(obj_path)
    print(f"Conversion complete!")
    return obj_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_glb_to_obj.py <glb_file> [obj_file]")
        sys.exit(1)
    
    glb_file = sys.argv[1]
    obj_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_glb_to_obj(glb_file, obj_file)

