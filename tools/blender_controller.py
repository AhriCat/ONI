import subprocess
import os
import json
import time
import socket
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import tempfile
import base64

logger = logging.getLogger(__name__)

class BlenderController:
    """Controller for automating Blender through Python API and command line."""
    
    def __init__(self, blender_path: str = None):
        """
        Initialize Blender controller.
        
        Args:
            blender_path: Path to Blender executable
        """
        self.blender_path = blender_path or self._find_blender_installation()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="blender_controller_"))
        
    def __del__(self):
        """Clean up resources."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _find_blender_installation(self) -> str:
        """Find Blender installation automatically."""
        common_paths = [
            "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
            "C:/Program Files/Blender Foundation/Blender 3.6/blender.exe",
            "C:/Program Files/Blender Foundation/Blender 3.5/blender.exe",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/usr/bin/blender"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Try to find Blender in PATH
        try:
            result = subprocess.run(["blender", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return "blender"
        except:
            pass
        
        raise FileNotFoundError("Blender installation not found")
    
    def execute_script(self, script: str, blend_file: str = None, output_file: str = None) -> Dict[str, Any]:
        """Execute Python script in Blender."""
        try:
            # Create temporary script file
            script_path = self.temp_dir / "blender_script.py"
            with open(script_path, 'w') as f:
                f.write(script)
            
            # Build command
            cmd = [self.blender_path, "--background"]
            
            if blend_file:
                cmd.append(blend_file)
            
            cmd.extend(["--python", str(script_path)])
            
            if output_file:
                cmd.extend(["--render-output", output_file])
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            # Parse output for JSON result
            output = result.stdout
            error = result.stderr
            
            # Look for JSON result in output
            json_result = None
            try:
                start_marker = "PYTHON_RESULT_JSON_START"
                end_marker = "PYTHON_RESULT_JSON_END"
                
                if start_marker in output and end_marker in output:
                    start_idx = output.find(start_marker) + len(start_marker)
                    end_idx = output.find(end_marker)
                    json_str = output[start_idx:end_idx].strip()
                    json_result = json.loads(json_str)
            except:
                pass
            
            return {
                "success": result.returncode == 0,
                "output": output,
                "error": error,
                "result": json_result
            }
            
        except Exception as e:
            logger.error(f"Failed to execute Blender script: {e}")
            return {"success": False, "error": str(e)}
    
    def create_primitive(self, primitive_type: str = "cube", location: Tuple[float, float, float] = (0, 0, 0)) -> Dict[str, Any]:
        """Create a primitive object in Blender."""
        script = f"""
import bpy
import json

# Delete default cube
if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'])

# Create primitive
if '{primitive_type}' == 'cube':
    bpy.ops.mesh.primitive_cube_add(location=({location[0]}, {location[1]}, {location[2]}))
elif '{primitive_type}' == 'sphere':
    bpy.ops.mesh.primitive_uv_sphere_add(location=({location[0]}, {location[1]}, {location[2]}))
elif '{primitive_type}' == 'cylinder':
    bpy.ops.mesh.primitive_cylinder_add(location=({location[0]}, {location[1]}, {location[2]}))
elif '{primitive_type}' == 'cone':
    bpy.ops.mesh.primitive_cone_add(location=({location[0]}, {location[1]}, {location[2]}))
elif '{primitive_type}' == 'torus':
    bpy.ops.mesh.primitive_torus_add(location=({location[0]}, {location[1]}, {location[2]}))
else:
    raise ValueError(f"Unknown primitive type: {primitive_type}")

# Get created object
obj = bpy.context.active_object
obj.name = '{primitive_type.capitalize()}'

# Print result as JSON
result = {{
    "name": obj.name,
    "type": '{primitive_type}',
    "location": list(obj.location)
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def create_material(self, name: str, color: Tuple[float, float, float] = (1, 0, 0), metallic: float = 0.0, roughness: float = 0.5) -> Dict[str, Any]:
        """Create a material in Blender."""
        script = f"""
import bpy
import json

# Create material
material = bpy.data.materials.new(name="{name}")
material.use_nodes = True
nodes = material.node_tree.nodes

# Get the Principled BSDF node
principled = nodes.get("Principled BSDF")
if principled:
    # Set color
    principled.inputs["Base Color"].default_value = ({color[0]}, {color[1]}, {color[2]}, 1.0)
    # Set metallic
    principled.inputs["Metallic"].default_value = {metallic}
    # Set roughness
    principled.inputs["Roughness"].default_value = {roughness}

# Assign material to active object if there is one
if bpy.context.active_object and bpy.context.active_object.type == 'MESH':
    if bpy.context.active_object.data.materials:
        bpy.context.active_object.data.materials[0] = material
    else:
        bpy.context.active_object.data.materials.append(material)

# Print result as JSON
result = {{
    "name": material.name,
    "color": [{color[0]}, {color[1]}, {color[2]}],
    "metallic": {metallic},
    "roughness": {roughness}
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def import_model(self, file_path: str) -> Dict[str, Any]:
        """Import a 3D model into Blender."""
        script = f"""
import bpy
import json
import os

# Determine file type
file_ext = os.path.splitext('{file_path}')[1].lower()

# Import based on file type
imported_objects = []
if file_ext == '.obj':
    bpy.ops.import_scene.obj(filepath='{file_path}')
    imported_objects = [o for o in bpy.context.selected_objects]
elif file_ext == '.fbx':
    bpy.ops.import_scene.fbx(filepath='{file_path}')
    imported_objects = [o for o in bpy.context.selected_objects]
elif file_ext == '.stl':
    bpy.ops.import_mesh.stl(filepath='{file_path}')
    imported_objects = [o for o in bpy.context.selected_objects]
elif file_ext == '.glb' or file_ext == '.gltf':
    bpy.ops.import_scene.gltf(filepath='{file_path}')
    imported_objects = [o for o in bpy.context.selected_objects]
elif file_ext == '.dae':
    bpy.ops.wm.collada_import(filepath='{file_path}')
    imported_objects = [o for o in bpy.context.selected_objects]
elif file_ext == '.blend':
    with bpy.data.libraries.load('{file_path}') as (data_from, data_to):
        data_to.objects = data_from.objects
    
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            imported_objects.append(obj)
else:
    raise ValueError(f"Unsupported file format: {file_ext}")

# Print result as JSON
result = {{
    "imported_objects": [obj.name for obj in imported_objects],
    "file_path": '{file_path}'
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def export_model(self, file_path: str, export_format: str = "obj") -> Dict[str, Any]:
        """Export a 3D model from Blender."""
        script = f"""
import bpy
import json
import os

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Export based on format
if '{export_format}' == 'obj':
    bpy.ops.export_scene.obj(filepath='{file_path}', use_selection=True)
elif '{export_format}' == 'fbx':
    bpy.ops.export_scene.fbx(filepath='{file_path}', use_selection=True)
elif '{export_format}' == 'stl':
    bpy.ops.export_mesh.stl(filepath='{file_path}', use_selection=True)
elif '{export_format}' == 'glb' or '{export_format}' == 'gltf':
    bpy.ops.export_scene.gltf(filepath='{file_path}', use_selection=True)
elif '{export_format}' == 'dae':
    bpy.ops.wm.collada_export(filepath='{file_path}', use_selection=True)
else:
    raise ValueError(f"Unsupported export format: {export_format}")

# Print result as JSON
result = {{
    "file_path": '{file_path}',
    "format": '{export_format}',
    "exported_objects": [obj.name for obj in bpy.context.selected_objects]
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def create_animation(self, object_name: str, keyframes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an animation for an object."""
        keyframes_json = json.dumps(keyframes)
        script = f"""
import bpy
import json

# Get the object
obj = bpy.data.objects.get('{object_name}')
if not obj:
    raise ValueError(f"Object '{object_name}' not found")

# Parse keyframes
keyframes = json.loads('''{keyframes_json}''')

# Create animation
for keyframe in keyframes:
    frame = keyframe.get('frame', 1)
    location = keyframe.get('location')
    rotation = keyframe.get('rotation')
    scale = keyframe.get('scale')
    
    # Set current frame
    bpy.context.scene.frame_set(frame)
    
    # Set location
    if location:
        obj.location = (location[0], location[1], location[2])
        obj.keyframe_insert(data_path="location", frame=frame)
    
    # Set rotation
    if rotation:
        obj.rotation_euler = (rotation[0], rotation[1], rotation[2])
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    
    # Set scale
    if scale:
        obj.scale = (scale[0], scale[1], scale[2])
        obj.keyframe_insert(data_path="scale", frame=frame)

# Set animation length
bpy.context.scene.frame_end = max(kf['frame'] for kf in keyframes)

# Print result as JSON
result = {{
    "object": '{object_name}',
    "keyframe_count": len(keyframes),
    "animation_length": bpy.context.scene.frame_end
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def render_image(self, output_path: str, resolution: Tuple[int, int] = (1920, 1080), samples: int = 128) -> Dict[str, Any]:
        """Render an image in Blender."""
        script = f"""
import bpy
import json
import os

# Set render settings
bpy.context.scene.render.resolution_x = {resolution[0]}
bpy.context.scene.render.resolution_y = {resolution[1]}
bpy.context.scene.render.resolution_percentage = 100

# Set Cycles as renderer
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = {samples}

# Set output path
bpy.context.scene.render.filepath = '{output_path}'

# Render
bpy.ops.render.render(write_still=True)

# Print result as JSON
result = {{
    "output_path": '{output_path}',
    "resolution": [{resolution[0]}, {resolution[1]}],
    "samples": {samples}
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def create_procedural_terrain(self, size: int = 10, subdivision: int = 128, height: float = 5.0) -> Dict[str, Any]:
        """Create procedural terrain in Blender."""
        script = f"""
import bpy
import json
import numpy as np

# Delete default cube
if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'])

# Create a plane
bpy.ops.mesh.primitive_plane_add(size={size}, location=(0, 0, 0))
plane = bpy.context.active_object
plane.name = 'Terrain'

# Add subdivision surface modifier
subdiv = plane.modifiers.new(name="Subdivision", type='SUBSURF')
subdiv.levels = 6
subdiv.render_levels = 6

# Add displace modifier
displace = plane.modifiers.new(name="Displace", type='DISPLACE')

# Create texture for displacement
texture = bpy.data.textures.new(name="TerrainNoise", type='CLOUDS')
texture.noise_scale = 1.0

# Assign texture to displace modifier
displace.texture = texture
displace.strength = {height}

# Apply modifiers
bpy.ops.object.modifier_apply({"object": plane}, modifier=subdiv.name)
bpy.ops.object.modifier_apply({"object": plane}, modifier=displace.name)

# Add material
material = bpy.data.materials.new(name="TerrainMaterial")
material.use_nodes = True
nodes = material.node_tree.nodes

# Get the Principled BSDF node
principled = nodes.get("Principled BSDF")
if principled:
    # Set color
    principled.inputs["Base Color"].default_value = (0.1, 0.5, 0.1, 1.0)
    # Set roughness
    principled.inputs["Roughness"].default_value = 0.8

# Assign material to terrain
if plane.data.materials:
    plane.data.materials[0] = material
else:
    plane.data.materials.append(material)

# Print result as JSON
result = {{
    "name": plane.name,
    "vertices": len(plane.data.vertices),
    "size": {size},
    "height": {height}
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def create_character(self, character_type: str = "humanoid") -> Dict[str, Any]:
        """Create a character in Blender."""
        script = f"""
import bpy
import json

# Delete default cube
if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'])

# Create character based on type
character = None
if '{character_type}' == 'humanoid':
    # Create metarig (requires rigify addon)
    try:
        bpy.ops.object.armature_human_metarig_add()
        character = bpy.context.active_object
        character.name = 'HumanoidCharacter'
    except:
        # Fallback if rigify is not available
        bpy.ops.object.armature_add(location=(0, 0, 0))
        character = bpy.context.active_object
        character.name = 'HumanoidCharacter'
elif '{character_type}' == 'quadruped':
    # Create quadruped metarig (requires rigify addon)
    try:
        bpy.ops.object.armature_human_metarig_add()
        character = bpy.context.active_object
        character.name = 'QuadrupedCharacter'
        # Modify to make it quadruped-like
        # (simplified for example)
    except:
        # Fallback if rigify is not available
        bpy.ops.object.armature_add(location=(0, 0, 0))
        character = bpy.context.active_object
        character.name = 'QuadrupedCharacter'
else:
    raise ValueError(f"Unknown character type: {character_type}")

# Print result as JSON
result = {{
    "name": character.name,
    "type": '{character_type}',
    "bones": len(character.data.bones) if hasattr(character.data, 'bones') else 0
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def create_particle_system(self, object_name: str, particle_type: str = "fire") -> Dict[str, Any]:
        """Create a particle system for an object."""
        script = f"""
import bpy
import json

# Get the object
obj = bpy.data.objects.get('{object_name}')
if not obj:
    # Create a default object if not found
    bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = '{object_name}'

# Add particle system
particle_system = obj.modifiers.new(name="ParticleSystem", type='PARTICLE_SYSTEM')
settings = particle_system.particle_system.settings

# Configure based on type
if '{particle_type}' == 'fire':
    settings.count = 5000
    settings.lifetime = 1.0
    settings.emit_from = 'FACE'
    settings.physics_type = 'NEWTON'
    settings.render_type = 'HALO'
    settings.size_random = 0.5
    settings.effector_weights.gravity = 0.0
    settings.normal_factor = 0.5
    settings.use_dynamic_rotation = True
    settings.angular_velocity_factor = 0.2
elif '{particle_type}' == 'smoke':
    settings.count = 1000
    settings.lifetime = 2.0
    settings.emit_from = 'VOLUME'
    settings.physics_type = 'NEWTON'
    settings.render_type = 'HALO'
    settings.size_random = 0.3
    settings.effector_weights.gravity = -0.1
elif '{particle_type}' == 'water':
    settings.count = 10000
    settings.lifetime = 1.5
    settings.emit_from = 'VOLUME'
    settings.physics_type = 'NEWTON'
    settings.render_type = 'HALO'
    settings.size_random = 0.1
    settings.effector_weights.gravity = 1.0
else:
    raise ValueError(f"Unknown particle type: {particle_type}")

# Print result as JSON
result = {{
    "object": obj.name,
    "particle_system": particle_system.name,
    "type": '{particle_type}',
    "particle_count": settings.count
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def create_physics_simulation(self, object_name: str, physics_type: str = "rigid_body") -> Dict[str, Any]:
        """Add physics simulation to an object."""
        script = f"""
import bpy
import json

# Get the object
obj = bpy.data.objects.get('{object_name}')
if not obj:
    # Create a default object if not found
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 5))
    obj = bpy.context.active_object
    obj.name = '{object_name}'

# Add physics based on type
if '{physics_type}' == 'rigid_body':
    bpy.ops.rigidbody.object_add({{'object':obj.name}})
    obj.rigid_body.mass = 1.0
    obj.rigid_body.collision_shape = 'CONVEX_HULL'
elif '{physics_type}' == 'soft_body':
    bpy.ops.object.modifier_add(type='SOFT_BODY')
    soft_body = obj.modifiers[-1]
    soft_body.settings.mass = 1.0
    soft_body.settings.goal_spring = 0.5
    soft_body.settings.goal_friction = 0.5
elif '{physics_type}' == 'cloth':
    bpy.ops.object.modifier_add(type='CLOTH')
    cloth = obj.modifiers[-1]
    cloth.settings.quality = 5
    cloth.settings.mass = 0.3
    cloth.settings.tension_stiffness = 15
else:
    raise ValueError(f"Unknown physics type: {physics_type}")

# Create a ground plane for collision
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = 'Ground'

if '{physics_type}' == 'rigid_body':
    bpy.ops.rigidbody.object_add({{'object':ground.name}})
    ground.rigid_body.type = 'PASSIVE'

# Set up animation
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 250

# Print result as JSON
result = {{
    "object": obj.name,
    "physics_type": '{physics_type}',
    "animation_frames": bpy.context.scene.frame_end
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def run_ai_agent(self, agent_script: str) -> Dict[str, Any]:
        """Run an AI agent script in Blender."""
        # Escape any quotes in the script
        agent_script = agent_script.replace("'", "\\'").replace('"', '\\"')
        
        script = f"""
import bpy
import json
import sys
import importlib.util
import traceback

try:
    # Create a temporary module for the agent script
    spec = importlib.util.spec_from_loader('oni_agent', loader=None)
    agent_module = importlib.util.module_from_spec(spec)
    
    # Execute the agent script
    exec('''{agent_script}''', agent_module.__dict__)
    
    # Run the agent
    if hasattr(agent_module, 'run_agent'):
        result = agent_module.run_agent(bpy)
        
        # Print result as JSON
        print("PYTHON_RESULT_JSON_START")
        print(json.dumps({{"success": True, "result": result}}))
        print("PYTHON_RESULT_JSON_END")
    else:
        print("PYTHON_RESULT_JSON_START")
        print(json.dumps({{"success": False, "error": "Agent script does not contain a run_agent function"}}))
        print("PYTHON_RESULT_JSON_END")
        
except Exception as e:
    error_msg = str(e)
    traceback_str = traceback.format_exc()
    
    print("PYTHON_RESULT_JSON_START")
    print(json.dumps({{"success": False, "error": error_msg, "traceback": traceback_str}}))
    print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def create_procedural_model(self, model_type: str, complexity: float = 0.5) -> Dict[str, Any]:
        """Create a procedural model in Blender."""
        script = f"""
import bpy
import json
import math
import random

# Delete default cube
if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'])

# Create procedural model based on type
if '{model_type}' == 'tree':
    # Create tree using sapling add-on if available
    try:
        bpy.ops.curve.tree_add(do_update=True)
        tree = bpy.context.active_object
        tree.name = 'ProceduralTree'
        
        # Adjust parameters based on complexity
        if hasattr(bpy.context.active_operator, "levels"):
            bpy.context.active_operator.levels = int(3 + {complexity} * 2)
        if hasattr(bpy.context.active_operator, "length"):
            bpy.context.active_operator.length = [1.0, 0.3, 0.6, 0.45]
        if hasattr(bpy.context.active_operator, "branches"):
            bpy.context.active_operator.branches = [50, 30, 10, 1]
        
        bpy.ops.curve.tree_update()
        
    except:
        # Fallback if sapling add-on is not available
        bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=2, location=(0, 0, 1))
        trunk = bpy.context.active_object
        trunk.name = 'TreeTrunk'
        
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.5, location=(0, 0, 2.5))
        leaves = bpy.context.active_object
        leaves.name = 'TreeLeaves'
        
        # Join objects
        bpy.ops.object.select_all(action='DESELECT')
        trunk.select_set(True)
        leaves.select_set(True)
        bpy.context.view_layer.objects.active = trunk
        bpy.ops.object.join()
        
        tree = trunk
        tree.name = 'ProceduralTree'
        
elif '{model_type}' == 'terrain':
    # Create terrain mesh
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20, size=10)
    terrain = bpy.context.active_object
    terrain.name = 'ProceduralTerrain'
    
    # Add displacement modifier
    displace = terrain.modifiers.new(name="Displace", type='DISPLACE')
    
    # Create texture for displacement
    texture = bpy.data.textures.new(name="TerrainNoise", type='CLOUDS')
    texture.noise_scale = 0.5 + {complexity} * 1.5
    
    # Assign texture to displace modifier
    displace.texture = texture
    displace.strength = 1.0 + {complexity} * 4.0
    
    # Apply modifier
    bpy.ops.object.modifier_apply({"object": terrain}, modifier=displace.name)
    
elif '{model_type}' == 'building':
    # Create building base
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
    base = bpy.context.active_object
    base.name = 'BuildingBase'
    base.scale = (2, 2, 1)
    
    # Create floors based on complexity
    floors = max(1, int({complexity} * 10))
    building_objects = [base]
    
    for i in range(1, floors):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5 + i))
        floor = bpy.context.active_object
        floor.name = f'BuildingFloor_{i}'
        floor.scale = (2 - (i * 0.1), 2 - (i * 0.1), 0.9)
        building_objects.append(floor)
    
    # Join objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in building_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = building_objects[0]
    bpy.ops.object.join()
    
    building = building_objects[0]
    building.name = 'ProceduralBuilding'
    
else:
    raise ValueError(f"Unknown model type: {model_type}")

# Print result as JSON
result = {{
    "name": bpy.context.active_object.name,
    "type": '{model_type}',
    "complexity": {complexity},
    "vertices": len(bpy.context.active_object.data.vertices)
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def create_animation_from_text(self, text: str, duration: int = 5) -> Dict[str, Any]:
        """Create an animation based on text description."""
        script = f"""
import bpy
import json
import random
import math

# Delete default cube
if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'])

# Parse text to determine animation type
text = '''{text}'''
text_lower = text.lower()

# Set up scene
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = {duration} * 30  # 30 fps

# Create objects and animation based on text
created_objects = []
animation_type = ""

if "rotate" in text_lower or "rotation" in text_lower or "spinning" in text_lower:
    # Create a rotating object
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "RotatingObject"
    created_objects.append(obj)
    
    # Create rotation animation
    obj.rotation_euler = (0, 0, 0)
    obj.keyframe_insert(data_path="rotation_euler", frame=1)
    
    obj.rotation_euler = (0, 0, math.pi * 2)
    obj.keyframe_insert(data_path="rotation_euler", frame=bpy.context.scene.frame_end)
    
    animation_type = "rotation"
    
elif "bounce" in text_lower or "jumping" in text_lower:
    # Create a bouncing object
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, 3))
    obj = bpy.context.active_object
    obj.name = "BouncingObject"
    created_objects.append(obj)
    
    # Create bouncing animation
    frames_per_bounce = 15
    num_bounces = bpy.context.scene.frame_end // frames_per_bounce
    
    for i in range(num_bounces + 1):
        frame = i * frames_per_bounce + 1
        
        # Bottom of bounce
        if i % 2 == 0:
            obj.location = (0, 0, 0.5)
        # Top of bounce
        else:
            obj.location = (0, 0, 3)
            
        obj.keyframe_insert(data_path="location", frame=frame)
    
    animation_type = "bounce"
    
elif "grow" in text_lower or "scale" in text_lower:
    # Create a growing object
    bpy.ops.mesh.primitive_cube_add(size=0.1, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "GrowingObject"
    created_objects.append(obj)
    
    # Create growing animation
    obj.scale = (0.1, 0.1, 0.1)
    obj.keyframe_insert(data_path="scale", frame=1)
    
    obj.scale = (2.0, 2.0, 2.0)
    obj.keyframe_insert(data_path="scale", frame=bpy.context.scene.frame_end)
    
    animation_type = "scale"
    
else:
    # Default: create a moving object
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-5, 0, 0))
    obj = bpy.context.active_object
    obj.name = "MovingObject"
    created_objects.append(obj)
    
    # Create movement animation
    obj.location = (-5, 0, 0)
    obj.keyframe_insert(data_path="location", frame=1)
    
    obj.location = (5, 0, 0)
    obj.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_end)
    
    animation_type = "movement"

# Add a ground plane
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
created_objects.append(ground)

# Add a camera
bpy.ops.object.camera_add(location=(0, -10, 5), rotation=(math.radians(60), 0, 0))
camera = bpy.context.active_object
camera.name = "Camera"
bpy.context.scene.camera = camera
created_objects.append(camera)

# Add a light
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
light = bpy.context.active_object
light.name = "Sun"
created_objects.append(light)

# Print result as JSON
result = {{
    "animation_type": animation_type,
    "duration": {duration},
    "frames": bpy.context.scene.frame_end,
    "objects": [obj.name for obj in created_objects]
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)
    
    def render_animation(self, output_path: str, resolution: Tuple[int, int] = (1280, 720), samples: int = 64) -> Dict[str, Any]:
        """Render an animation in Blender."""
        script = f"""
import bpy
import json
import os

# Set render settings
bpy.context.scene.render.resolution_x = {resolution[0]}
bpy.context.scene.render.resolution_y = {resolution[1]}
bpy.context.scene.render.resolution_percentage = 100

# Set Cycles as renderer
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = {samples}
bpy.context.scene.cycles.device = 'GPU'

# Set output path
output_dir = os.path.dirname('{output_path}')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bpy.context.scene.render.filepath = '{output_path}'
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.ffmpeg.format = 'MPEG4'
bpy.context.scene.render.ffmpeg.codec = 'H264'
bpy.context.scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'

# Render animation
bpy.ops.render.render(animation=True)

# Print result as JSON
result = {{
    "output_path": '{output_path}',
    "resolution": [{resolution[0]}, {resolution[1]}],
    "samples": {samples},
    "frames": bpy.context.scene.frame_end
}}

print("PYTHON_RESULT_JSON_START")
print(json.dumps(result))
print("PYTHON_RESULT_JSON_END")
"""
        return self.execute_script(script)

# Example usage
if __name__ == "__main__":
    controller = BlenderController()
    controller.create_primitive("cube")
    controller.create_material("RedMaterial", (1, 0, 0))
    controller.render_image("output.png")