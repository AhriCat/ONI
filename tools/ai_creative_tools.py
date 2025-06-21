import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

# Import controllers
from tools.unreal_controller import UnrealEngineController
from tools.unity_controller import UnityController
from tools.blender_controller import BlenderController
from tools.after_effects_controller import AfterEffectsController
from tools.photoshop_controller import PhotoshopController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AICreativeTools:
    """Main interface for AI to control creative tools."""
    
    def __init__(self):
        """Initialize AI creative tools."""
        self.unreal = None
        self.unity = None
        self.blender = None
        self.after_effects = None
        self.photoshop = None
        
        # Track initialized tools
        self.initialized_tools = set()
        
    def initialize_tool(self, tool_name: str, **kwargs) -> bool:
        """Initialize a specific creative tool."""
        try:
            if tool_name.lower() == "unreal":
                self.unreal = UnrealEngineController(**kwargs)
                self.initialized_tools.add("unreal")
                logger.info("Unreal Engine controller initialized")
                return True
            elif tool_name.lower() == "unity":
                self.unity = UnityController(**kwargs)
                self.initialized_tools.add("unity")
                logger.info("Unity controller initialized")
                return True
            elif tool_name.lower() == "blender":
                self.blender = BlenderController(**kwargs)
                self.initialized_tools.add("blender")
                logger.info("Blender controller initialized")
                return True
            elif tool_name.lower() == "after_effects" or tool_name.lower() == "aftereffects":
                self.after_effects = AfterEffectsController(**kwargs)
                self.initialized_tools.add("after_effects")
                logger.info("After Effects controller initialized")
                return True
            elif tool_name.lower() == "photoshop":
                self.photoshop = PhotoshopController(**kwargs)
                self.initialized_tools.add("photoshop")
                logger.info("Photoshop controller initialized")
                return True
            else:
                logger.error(f"Unknown tool: {tool_name}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize {tool_name}: {e}")
            return False
    
    def get_initialized_tools(self) -> List[str]:
        """Get list of initialized tools."""
        return list(self.initialized_tools)
    
    def run_unreal_agent(self, agent_script: str) -> Dict[str, Any]:
        """Run an AI agent script in Unreal Engine."""
        if "unreal" not in self.initialized_tools:
            return {"success": False, "error": "Unreal Engine controller not initialized"}
        
        return self.unreal.run_ai_agent(agent_script)
    
    def run_unity_agent(self, agent_script: str) -> Dict[str, Any]:
        """Run an AI agent script in Unity."""
        if "unity" not in self.initialized_tools:
            return {"success": False, "error": "Unity controller not initialized"}
        
        return self.unity.run_ai_agent(agent_script)
    
    def run_blender_agent(self, agent_script: str) -> Dict[str, Any]:
        """Run an AI agent script in Blender."""
        if "blender" not in self.initialized_tools:
            return {"success": False, "error": "Blender controller not initialized"}
        
        return self.blender.run_ai_agent(agent_script)
    
    def run_after_effects_agent(self, agent_script: str) -> Dict[str, Any]:
        """Run an AI agent script in After Effects."""
        if "after_effects" not in self.initialized_tools:
            return {"success": False, "error": "After Effects controller not initialized"}
        
        return self.after_effects.run_ai_agent(agent_script)
    
    def run_photoshop_agent(self, agent_script: str) -> Dict[str, Any]:
        """Run an AI agent script in Photoshop."""
        if "photoshop" not in self.initialized_tools:
            return {"success": False, "error": "Photoshop controller not initialized"}
        
        return self.photoshop.run_ai_agent(agent_script)
    
    def create_3d_scene_from_text(self, text: str, tool: str = "blender", output_path: str = None) -> Dict[str, Any]:
        """Create a 3D scene based on text description."""
        tool = tool.lower()
        
        if tool == "blender":
            if "blender" not in self.initialized_tools:
                self.initialize_tool("blender")
            
            return self.blender.create_animation_from_text(text, output_path)
        elif tool == "unreal":
            if "unreal" not in self.initialized_tools:
                self.initialize_tool("unreal")
            
            # Create a procedural level based on text
            return self.unreal.generate_procedural_level()
        elif tool == "unity":
            if "unity" not in self.initialized_tools:
                self.initialize_tool("unity")
            
            # Create a scene based on text
            # This is a simplified implementation
            return {"success": False, "error": "Not implemented yet"}
        else:
            return {"success": False, "error": f"Unsupported tool: {tool}"}
    
    def create_animation_from_text(self, text: str, tool: str = "after_effects", output_path: str = None) -> Dict[str, Any]:
        """Create an animation based on text description."""
        tool = tool.lower()
        
        if tool == "after_effects":
            if "after_effects" not in self.initialized_tools:
                self.initialize_tool("after_effects")
            
            return self.after_effects.create_animation_from_text(text, output_path)
        elif tool == "blender":
            if "blender" not in self.initialized_tools:
                self.initialize_tool("blender")
            
            return self.blender.create_animation_from_text(text, output_path)
        else:
            return {"success": False, "error": f"Unsupported tool: {tool}"}
    
    def create_image_from_text(self, text: str, tool: str = "photoshop", output_path: str = None) -> Dict[str, Any]:
        """Create an image based on text description."""
        tool = tool.lower()
        
        if tool == "photoshop":
            if "photoshop" not in self.initialized_tools:
                self.initialize_tool("photoshop")
            
            return self.photoshop.create_image_from_text(text, output_path)
        else:
            return {"success": False, "error": f"Unsupported tool: {tool}"}
    
    def create_game_character(self, description: str, tool: str = "unreal") -> Dict[str, Any]:
        """Create a game character based on description."""
        tool = tool.lower()
        
        if tool == "unreal":
            if "unreal" not in self.initialized_tools:
                self.initialize_tool("unreal")
            
            # Create a character blueprint
            return self.unreal.create_blueprint("Character", "Character")
        elif tool == "unity":
            if "unity" not in self.initialized_tools:
                self.initialize_tool("unity")
            
            # Create a character
            return self.unity.create_character("humanoid")
        elif tool == "blender":
            if "blender" not in self.initialized_tools:
                self.initialize_tool("blender")
            
            # Create a character
            return self.blender.create_character("humanoid")
        else:
            return {"success": False, "error": f"Unsupported tool: {tool}"}
    
    def create_visual_effect(self, description: str, tool: str = "unreal") -> Dict[str, Any]:
        """Create a visual effect based on description."""
        tool = tool.lower()
        
        if tool == "unreal":
            if "unreal" not in self.initialized_tools:
                self.initialize_tool("unreal")
            
            # Create a particle system
            # This is a simplified implementation
            return {"success": False, "error": "Not implemented yet"}
        elif tool == "unity":
            if "unity" not in self.initialized_tools:
                self.initialize_tool("unity")
            
            # Create a particle system
            return self.unity.create_particle_system("Effect", "fire")
        elif tool == "blender":
            if "blender" not in self.initialized_tools:
                self.initialize_tool("blender")
            
            # Create a particle system
            return self.blender.create_particle_system("Cube", "fire")
        elif tool == "after_effects":
            if "after_effects" not in self.initialized_tools:
                self.initialize_tool("after_effects")
            
            # Create a composition with effect
            comp_result = self.after_effects.create_composition("Visual Effect")
            
            if not comp_result.get("success"):
                return comp_result
            
            comp_id = comp_result["compId"]
            
            # Create a solid
            solid_result = self.after_effects.create_solid(comp_id, [0, 0, 0], "Effect Solid")
            
            if not solid_result.get("success"):
                return solid_result
            
            layer_index = solid_result["layerId"]
            
            # Add effect based on description
            description_lower = description.lower()
            
            effect_name = "CC Particle World"
            if "fire" in description_lower:
                effect_name = "CC Particle World"
            elif "glow" in description_lower:
                effect_name = "Glow"
            elif "blur" in description_lower:
                effect_name = "Gaussian Blur"
            
            effect_result = self.after_effects.add_effect(comp_id, layer_index, effect_name)
            
            return {
                "success": effect_result.get("success", False),
                "compId": comp_id,
                "layerId": layer_index,
                "effectName": effect_name
            }
        else:
            return {"success": False, "error": f"Unsupported tool: {tool}"}
    
    def create_ui_design(self, description: str, tool: str = "photoshop") -> Dict[str, Any]:
        """Create a UI design based on description."""
        tool = tool.lower()
        
        if tool == "photoshop":
            if "photoshop" not in self.initialized_tools:
                self.initialize_tool("photoshop")
            
            # Create a document
            doc_result = self.photoshop.create_document("UI Design", 1920, 1080)
            
            if not doc_result.get("success"):
                return doc_result
            
            # Parse description to determine UI type
            description_lower = description.lower()
            
            if "mobile" in description_lower:
                # Create mobile UI
                # Background
                bg_result = self.photoshop.add_shape("rectangle", [0, 0], [1080, 1920], [240, 240, 240])
                
                if not bg_result.get("success"):
                    return bg_result
                
                # Header
                header_result = self.photoshop.add_shape("rectangle", [0, 0], [1080, 200], [50, 50, 50])
                
                if not header_result.get("success"):
                    return header_result
                
                # Title
                title_text = "Mobile App"
                if "title" in description_lower:
                    # Extract title from description
                    title_match = re.search(r'title[:\s]+([^,\.]+)', description_lower)
                    if title_match:
                        title_text = title_match.group(1).strip()
                
                title_result = self.photoshop.add_text(title_text, [540, 100], 48, [255, 255, 255])
                
                if not title_result.get("success"):
                    return title_result
                
                # Content area
                for i in range(3):
                    item_result = self.photoshop.add_shape("rectangle", [50, 250 + i * 300], [980, 250], [255, 255, 255])
                    
                    if not item_result.get("success"):
                        return item_result
                
                # Navigation bar
                nav_result = self.photoshop.add_shape("rectangle", [0, 1720], [1080, 200], [50, 50, 50])
                
                if not nav_result.get("success"):
                    return nav_result
            
            elif "web" in description_lower:
                # Create web UI
                # Background
                bg_result = self.photoshop.add_shape("rectangle", [0, 0], [1920, 1080], [255, 255, 255])
                
                if not bg_result.get("success"):
                    return bg_result
                
                # Header
                header_result = self.photoshop.add_shape("rectangle", [0, 0], [1920, 100], [50, 50, 50])
                
                if not header_result.get("success"):
                    return header_result
                
                # Title
                title_text = "Website"
                if "title" in description_lower:
                    # Extract title from description
                    title_match = re.search(r'title[:\s]+([^,\.]+)', description_lower)
                    if title_match:
                        title_text = title_match.group(1).strip()
                
                title_result = self.photoshop.add_text(title_text, [100, 50], 36, [255, 255, 255])
                
                if not title_result.get("success"):
                    return title_result
                
                # Navigation
                for i in range(4):
                    nav_item_result = self.photoshop.add_text(f"Menu {i+1}", [800 + i * 200, 50], 24, [255, 255, 255])
                    
                    if not nav_item_result.get("success"):
                        return nav_item_result
                
                # Hero section
                hero_result = self.photoshop.add_shape("rectangle", [0, 100], [1920, 500], [200, 200, 200])
                
                if not hero_result.get("success"):
                    return hero_result
                
                # Hero text
                hero_text_result = self.photoshop.add_text("Welcome to our website", [960, 350], 72, [255, 255, 255])
                
                if not hero_text_result.get("success"):
                    return hero_text_result
                
                # Content sections
                for i in range(3):
                    section_result = self.photoshop.add_shape("rectangle", [50 + i * 640, 650], [590, 380], [240, 240, 240])
                    
                    if not section_result.get("success"):
                        return section_result
                    
                    section_title_result = self.photoshop.add_text(f"Section {i+1}", [345 + i * 640, 700], 36, [0, 0, 0])
                    
                    if not section_title_result.get("success"):
                        return section_title_result
            
            else:
                # Create generic UI
                # Background
                bg_result = self.photoshop.add_shape("rectangle", [0, 0], [1920, 1080], [240, 240, 240])
                
                if not bg_result.get("success"):
                    return bg_result
                
                # Title
                title_text = "UI Design"
                if "title" in description_lower:
                    # Extract title from description
                    title_match = re.search(r'title[:\s]+([^,\.]+)', description_lower)
                    if title_match:
                        title_text = title_match.group(1).strip()
                
                title_result = self.photoshop.add_text(title_text, [960, 100], 72, [0, 0, 0])
                
                if not title_result.get("success"):
                    return title_result
                
                # Content
                content_result = self.photoshop.add_shape("rectangle", [200, 200], [1520, 780], [255, 255, 255])
                
                if not content_result.get("success"):
                    return content_result
            
            return {
                "success": True,
                "description": description,
                "tool": tool
            }
        else:
            return {"success": False, "error": f"Unsupported tool: {tool}"}
    
    def create_game_level(self, description: str, tool: str = "unreal") -> Dict[str, Any]:
        """Create a game level based on description."""
        tool = tool.lower()
        
        if tool == "unreal":
            if "unreal" not in self.initialized_tools:
                self.initialize_tool("unreal")
            
            # Generate a procedural level
            return self.unreal.generate_procedural_level()
        elif tool == "unity":
            if "unity" not in self.initialized_tools:
                self.initialize_tool("unity")
            
            # Create a terrain
            return self.unity.create_terrain()
        else:
            return {"success": False, "error": f"Unsupported tool: {tool}"}
    
    def create_cinematic(self, description: str, tool: str = "after_effects") -> Dict[str, Any]:
        """Create a cinematic based on description."""
        tool = tool.lower()
        
        if tool == "after_effects":
            if "after_effects" not in self.initialized_tools:
                self.initialize_tool("after_effects")
            
            # Create a composition
            comp_result = self.after_effects.create_composition("Cinematic")
            
            if not comp_result.get("success"):
                return comp_result
            
            comp_id = comp_result["compId"]
            
            # Create a camera
            camera_result = self.after_effects.create_camera(comp_id)
            
            if not camera_result.get("success"):
                return camera_result
            
            # Create text layers based on description
            words = description.split()
            for i, word in enumerate(words[:5]):  # Limit to first 5 words
                text_result = self.after_effects.create_text_layer(comp_id, word, [960, 540 + (i - 2) * 100])
                
                if not text_result.get("success"):
                    return text_result
            
            return {
                "success": True,
                "compId": comp_id,
                "description": description,
                "tool": tool
            }
        elif tool == "unreal":
            if "unreal" not in self.initialized_tools:
                self.initialize_tool("unreal")
            
            # Create a cinematic sequence
            return self.unreal.create_cinematic_sequence("Cinematic")
        else:
            return {"success": False, "error": f"Unsupported tool: {tool}"}

# Example usage
if __name__ == "__main__":
    ai_tools = AICreativeTools()
    
    # Initialize tools
    ai_tools.initialize_tool("blender")
    
    # Create a 3D scene from text
    result = ai_tools.create_3d_scene_from_text("A rotating cube on a plane", "blender")
    print(result)