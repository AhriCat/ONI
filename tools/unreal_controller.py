import subprocess
import json
import time
import os
import socket
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class UnrealEngineController:
    """Controller for automating Unreal Engine through Python API and command line."""
    
    def __init__(self, unreal_path: str = None, project_path: str = None):
        """
        Initialize Unreal Engine controller.
        
        Args:
            unreal_path: Path to Unreal Engine installation
            project_path: Path to Unreal project
        """
        self.unreal_path = unreal_path or self._find_unreal_installation()
        self.project_path = project_path
        self.editor_process = None
        self.python_socket = None
        self.is_connected = False
        
    def _find_unreal_installation(self) -> str:
        """Find Unreal Engine installation automatically."""
        common_paths = [
            "C:/Program Files/Epic Games/UE_5.3/Engine/Binaries/Win64/UnrealEditor.exe",
            "C:/Program Files/Epic Games/UE_5.2/Engine/Binaries/Win64/UnrealEditor.exe",
            "C:/Program Files/Epic Games/UE_5.1/Engine/Binaries/Win64/UnrealEditor.exe",
            "/Applications/Epic Games/UE_5.3/Engine/Binaries/Mac/UnrealEditor.app",
            "/opt/UnrealEngine/Engine/Binaries/Linux/UnrealEditor"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Unreal Engine installation not found")
    
    def start_editor(self, headless: bool = False) -> bool:
        """Start Unreal Editor with Python support."""
        try:
            cmd = [self.unreal_path]
            
            if self.project_path:
                cmd.append(self.project_path)
            
            if headless:
                cmd.extend(["-unattended", "-nographics", "-nullrhi"])
            
            # Enable Python scripting
            cmd.extend(["-ExecutePythonScript=import unreal"])
            
            self.editor_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for editor to start
            time.sleep(10)
            
            logger.info("Unreal Editor started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Unreal Editor: {e}")
            return False
    
    def execute_python_script(self, script: str) -> Dict[str, Any]:
        """Execute Python script in Unreal Editor."""
        try:
            # Create temporary script file
            script_path = Path("temp_unreal_script.py")
            with open(script_path, 'w') as f:
                f.write(script)
            
            # Execute script through command line
            cmd = [
                self.unreal_path,
                "-ExecutePythonScript=" + str(script_path.absolute())
            ]
            
            if self.project_path:
                cmd.append(self.project_path)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Clean up
            script_path.unlink(missing_ok=True)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
            
        except Exception as e:
            logger.error(f"Failed to execute Python script: {e}")
            return {"success": False, "error": str(e)}
    
    def create_actor(self, actor_class: str, location: Tuple[float, float, float] = (0, 0, 0)) -> Dict[str, Any]:
        """Create a new actor in the level."""
        script = f"""
import unreal

# Get the editor world
world = unreal.EditorLevelLibrary.get_editor_world()

# Create actor
actor_class = unreal.load_class(None, '{actor_class}')
location = unreal.Vector({location[0]}, {location[1]}, {location[2]})

actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
    actor_class, 
    location
)

if actor:
    print(f"Created actor: {{actor.get_name()}}")
else:
    print("Failed to create actor")
"""
        return self.execute_python_script(script)
    
    def create_landscape(self, size: int = 1024, height_scale: float = 100.0) -> Dict[str, Any]:
        """Create a landscape in the level."""
        script = f"""
import unreal

# Create landscape
landscape_info = unreal.LandscapeImportLayerInfo()
landscape_info.layer_name = "Ground"

# Set up landscape creation parameters
landscape_settings = unreal.LandscapeEditorObject()
landscape_settings.new_landscape_preview_mode = unreal.NewLandscapePreviewMode.NEW_LANDSCAPE

# Create the landscape
world = unreal.EditorLevelLibrary.get_editor_world()
landscape = unreal.LandscapeEditorObject.create_landscape_for_world(
    world,
    unreal.Vector(0, 0, 0),
    {size},
    {size},
    1,
    1
)

print(f"Created landscape with size {size}x{size}")
"""
        return self.execute_python_script(script)
    
    def import_asset(self, file_path: str, destination_path: str = "/Game/") -> Dict[str, Any]:
        """Import an asset into the project."""
        script = f"""
import unreal

# Set up import task
task = unreal.AssetImportTask()
task.set_editor_property('automated', True)
task.set_editor_property('destination_path', '{destination_path}')
task.set_editor_property('filename', '{file_path}')
task.set_editor_property('replace_existing', True)
task.set_editor_property('save', True)

# Import the asset
unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

print(f"Imported asset: {file_path}")
"""
        return self.execute_python_script(script)
    
    def create_material(self, material_name: str, base_color: Tuple[float, float, float] = (1, 1, 1)) -> Dict[str, Any]:
        """Create a new material."""
        script = f"""
import unreal

# Create material
material_factory = unreal.MaterialFactoryNew()
material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
    '{material_name}',
    '/Game/Materials',
    unreal.Material,
    material_factory
)

# Set base color
base_color_node = unreal.MaterialEditingLibrary.create_material_expression(
    material,
    unreal.MaterialExpressionConstant3Vector,
    -300,
    0
)

base_color_node.set_editor_property('constant', unreal.LinearColor({base_color[0]}, {base_color[1]}, {base_color[2]}, 1.0))

# Connect to material
unreal.MaterialEditingLibrary.connect_material_property(
    base_color_node,
    '',
    unreal.MaterialProperty.MP_BASE_COLOR
)

# Compile and save
unreal.MaterialEditingLibrary.recompile_material(material)
unreal.EditorAssetLibrary.save_asset(material.get_path_name())

print(f"Created material: {material_name}")
"""
        return self.execute_python_script(script)
    
    def setup_lighting(self, lighting_type: str = "dynamic") -> Dict[str, Any]:
        """Set up lighting in the scene."""
        script = f"""
import unreal

world = unreal.EditorLevelLibrary.get_editor_world()

# Create directional light
light_class = unreal.load_class(None, '/Script/Engine.DirectionalLight')
light = unreal.EditorLevelLibrary.spawn_actor_from_class(
    light_class,
    unreal.Vector(0, 0, 1000)
)

# Configure light
light.set_editor_property('intensity', 10.0)
light.set_editor_property('temperature', 6500.0)

# Set up lighting mode
if '{lighting_type}' == 'dynamic':
    light.set_mobility(unreal.ComponentMobility.MOVABLE)
else:
    light.set_mobility(unreal.ComponentMobility.STATIC)

# Create sky atmosphere
sky_class = unreal.load_class(None, '/Script/Engine.SkyAtmosphere')
sky = unreal.EditorLevelLibrary.spawn_actor_from_class(
    sky_class,
    unreal.Vector(0, 0, 0)
)

print(f"Set up {lighting_type} lighting")
"""
        return self.execute_python_script(script)
    
    def create_blueprint(self, blueprint_name: str, parent_class: str = "Actor") -> Dict[str, Any]:
        """Create a new Blueprint class."""
        script = f"""
import unreal

# Create blueprint
factory = unreal.BlueprintFactory()
factory.set_editor_property('parent_class', unreal.load_class(None, '/Script/Engine.{parent_class}'))

blueprint = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
    '{blueprint_name}',
    '/Game/Blueprints',
    unreal.Blueprint,
    factory
)

# Save the blueprint
unreal.EditorAssetLibrary.save_asset(blueprint.get_path_name())

print(f"Created blueprint: {blueprint_name}")
"""
        return self.execute_python_script(script)
    
    def build_lighting(self) -> Dict[str, Any]:
        """Build lighting for the level."""
        script = """
import unreal

# Build lighting
unreal.EditorLevelLibrary.build_lighting()
print("Building lighting...")
"""
        return self.execute_python_script(script)
    
    def save_level(self, level_path: str = None) -> Dict[str, Any]:
        """Save the current level."""
        script = f"""
import unreal

# Save the current level
if '{level_path}':
    unreal.EditorLevelLibrary.save_current_level_as('{level_path}')
else:
    unreal.EditorLevelLibrary.save_current_level()

print("Level saved successfully")
"""
        return self.execute_python_script(script)
    
    def export_fbx(self, actor_name: str, output_path: str) -> Dict[str, Any]:
        """Export an actor to FBX format."""
        script = f"""
import unreal

# Find the actor
actor = None
for a in unreal.EditorLevelLibrary.get_all_level_actors():
    if a.get_name() == '{actor_name}':
        actor = a
        break

if not actor:
    print(f"Actor '{actor_name}' not found")
    exit(1)

# Set up export options
options = unreal.FbxExportOption()
options.set_editor_property('fbx_export_compatibility', unreal.FbxExportCompatibility.FBX_2020)
options.set_editor_property('vertex_color', True)
options.set_editor_property('collision', False)

# Export the actor
result = unreal.EditorStaticMeshLibrary.export_to_file(
    actor,
    '{output_path}',
    options
)

print(f"Exported actor to {output_path}")
"""
        return self.execute_python_script(script)
    
    def close_editor(self) -> bool:
        """Close Unreal Editor."""
        try:
            if self.editor_process:
                script = """
import unreal
unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
unreal.SystemLibrary.quit_editor()
"""
                self.execute_python_script(script)
                self.editor_process.terminate()
                self.editor_process = None
                logger.info("Unreal Editor closed successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to close Unreal Editor: {e}")
            return False
    
    def create_animation_sequence(self, skeleton_path: str, animation_name: str) -> Dict[str, Any]:
        """Create a new animation sequence."""
        script = f"""
import unreal

# Load the skeleton
skeleton = unreal.load_asset('{skeleton_path}')
if not skeleton:
    print(f"Skeleton not found: {skeleton_path}")
    exit(1)

# Create animation sequence
factory = unreal.AnimSequenceFactory()
factory.set_editor_property('skeleton', skeleton)
factory.set_editor_property('target_skeleton', skeleton)

animation = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
    '{animation_name}',
    '/Game/Animations',
    unreal.AnimSequence,
    factory
)

# Save the animation
unreal.EditorAssetLibrary.save_asset(animation.get_path_name())

print(f"Created animation sequence: {animation_name}")
"""
        return self.execute_python_script(script)
    
    def create_ai_behavior_tree(self, ai_name: str) -> Dict[str, Any]:
        """Create an AI behavior tree."""
        script = f"""
import unreal

# Create behavior tree
bt_factory = unreal.BehaviorTreeFactory()
behavior_tree = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
    '{ai_name}_BT',
    '/Game/AI',
    unreal.BehaviorTree,
    bt_factory
)

# Create blackboard
bb_factory = unreal.BlackboardDataFactory()
blackboard = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
    '{ai_name}_BB',
    '/Game/AI',
    unreal.BlackboardData,
    bb_factory
)

# Link blackboard to behavior tree
behavior_tree.set_editor_property('blackboard_asset', blackboard)

# Save assets
unreal.EditorAssetLibrary.save_asset(behavior_tree.get_path_name())
unreal.EditorAssetLibrary.save_asset(blackboard.get_path_name())

print(f"Created AI behavior tree and blackboard for {ai_name}")
"""
        return self.execute_python_script(script)
    
    def generate_procedural_level(self, size: int = 10, complexity: float = 0.5) -> Dict[str, Any]:
        """Generate a procedural level."""
        script = f"""
import unreal
import random

# Clear existing actors
unreal.EditorLevelLibrary.select_nothing()
for actor in unreal.EditorLevelLibrary.get_all_level_actors():
    if not actor.get_class().get_name() in ['WorldSettings', 'LevelBounds', 'SkyLight', 'DirectionalLight']:
        unreal.EditorLevelLibrary.select_actor(actor, True, False)
unreal.EditorLevelLibrary.delete_selected_actors()

# Load cube mesh
cube_mesh = unreal.load_asset('/Engine/BasicShapes/Cube')

# Generate grid
size = {size}
complexity = {complexity}
for x in range(size):
    for y in range(size):
        # Random height based on complexity
        if random.random() < complexity:
            height = random.randint(1, 5)
            
            # Create cube
            location = unreal.Vector(x * 400, y * 400, height * 100)
            actor = unreal.EditorLevelLibrary.spawn_actor_from_object(
                cube_mesh,
                location,
                unreal.Rotator(0, 0, 0)
            )
            
            # Scale cube
            actor.set_actor_scale3d(unreal.Vector(1.0, 1.0, height * 0.5))

print(f"Generated procedural level with size {size} and complexity {complexity}")
"""
        return self.execute_python_script(script)
    
    def create_cinematic_sequence(self, sequence_name: str) -> Dict[str, Any]:
        """Create a cinematic sequence."""
        script = f"""
import unreal

# Create level sequence
sequence_factory = unreal.LevelSequenceFactoryNew()
sequence = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
    '{sequence_name}',
    '/Game/Cinematics',
    unreal.LevelSequence,
    sequence_factory
)

# Add a camera track
camera_class = unreal.load_class(None, '/Script/Engine.CineCameraActor')
camera = unreal.EditorLevelLibrary.spawn_actor_from_class(
    camera_class,
    unreal.Vector(0, 0, 100)
)

# Add camera to sequence
binding = sequence.add_possessable(camera)
camera_track = binding.add_track(unreal.MovieScenePropertyTrack)
camera_track.set_property_name_and_path('Transform', 'Transform')

# Add keyframes
transform_section = camera_track.add_section()
transform_section.set_range(0, 300)  # 10 seconds at 30 fps

# Save sequence
unreal.EditorAssetLibrary.save_asset(sequence.get_path_name())

print(f"Created cinematic sequence: {sequence_name}")
"""
        return self.execute_python_script(script)

    def run_ai_agent(self, agent_script: str) -> Dict[str, Any]:
        """Run an AI agent script in Unreal Engine."""
        script = f"""
import unreal
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
        result = agent_module.run_agent(unreal)
        print(f"Agent executed successfully: {{result}}")
    else:
        print("Agent script does not contain a run_agent function")
        
except Exception as e:
    print(f"Error running agent: {{e}}")
    print(traceback.format_exc())
"""
        return self.execute_python_script(script)

# Example usage
if __name__ == "__main__":
    controller = UnrealEngineController()
    controller.start_editor()
    controller.create_actor("StaticMeshActor", (0, 0, 100))
    controller.create_material("NewMaterial", (1, 0, 0))
    controller.save_level()
    controller.close_editor()