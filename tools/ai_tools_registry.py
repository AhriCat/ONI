import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import threading
import queue
import traceback

# Import ONI creative agent
from tools.oni_creative_agent import ONICreativeAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIToolsRegistry:
    """Registry for AI creative tools integration with ONI."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one registry instance."""
        if cls._instance is None:
            cls._instance = super(AIToolsRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize AI tools registry."""
        if self._initialized:
            return
        
        self.creative_agent = ONICreativeAgent()
        self.registered_tools = {}
        self.tool_capabilities = {
            "unreal": [
                "create_3d_scene",
                "create_game_character",
                "create_visual_effect",
                "create_game_level",
                "create_cinematic"
            ],
            "unity": [
                "create_3d_scene",
                "create_game_character",
                "create_visual_effect",
                "create_game_level"
            ],
            "blender": [
                "create_3d_scene",
                "create_animation",
                "create_game_character",
                "create_visual_effect"
            ],
            "after_effects": [
                "create_animation",
                "create_visual_effect",
                "create_cinematic"
            ],
            "photoshop": [
                "create_image",
                "create_ui_design"
            ]
        }
        
        self._initialized = True
        logger.info("AI Tools Registry initialized")
    
    def register_tool(self, tool_name: str, tool_path: str = None) -> bool:
        """Register a creative tool with the registry."""
        tool_name = tool_name.lower()
        
        if tool_name not in self.tool_capabilities:
            logger.error(f"Unknown tool: {tool_name}")
            return False
        
        try:
            # Initialize the tool
            task_id = None
            
            if tool_name == "unreal":
                task_id = self.creative_agent.tools_interface.initialize_unreal(tool_path)
            elif tool_name == "unity":
                task_id = self.creative_agent.tools_interface.initialize_unity(tool_path)
            elif tool_name == "blender":
                task_id = self.creative_agent.tools_interface.initialize_blender(tool_path)
            elif tool_name == "after_effects":
                task_id = self.creative_agent.tools_interface.initialize_after_effects(tool_path)
            elif tool_name == "photoshop":
                task_id = self.creative_agent.tools_interface.initialize_photoshop(tool_path)
            
            if task_id:
                result = self.creative_agent.tools_interface.wait_for_result(task_id, timeout=30)
                
                if result and result.get("result", {}).get("success", False):
                    self.registered_tools[tool_name] = {
                        "path": tool_path,
                        "capabilities": self.tool_capabilities[tool_name],
                        "registered_at": time.time()
                    }
                    logger.info(f"Tool registered: {tool_name}")
                    return True
            
            logger.error(f"Failed to register tool: {tool_name}")
            return False
            
        except Exception as e:
            logger.error(f"Error registering tool {tool_name}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a creative tool from the registry."""
        tool_name = tool_name.lower()
        
        if tool_name in self.registered_tools:
            del self.registered_tools[tool_name]
            logger.info(f"Tool unregistered: {tool_name}")
            return True
        else:
            logger.warning(f"Tool not registered: {tool_name}")
            return False
    
    def get_registered_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered tools."""
        return self.registered_tools
    
    def get_tool_capabilities(self, tool_name: str) -> List[str]:
        """Get capabilities of a specific tool."""
        tool_name = tool_name.lower()
        
        if tool_name in self.tool_capabilities:
            return self.tool_capabilities[tool_name]
        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return []
    
    def get_tools_for_capability(self, capability: str) -> List[str]:
        """Get all tools that support a specific capability."""
        capability = capability.lower()
        
        tools = []
        for tool, capabilities in self.tool_capabilities.items():
            if capability in capabilities:
                tools.append(tool)
        
        return tools
    
    def process_creative_request(self, request: str) -> Dict[str, Any]:
        """Process a creative request using the ONI creative agent."""
        return self.creative_agent.process_creative_request(request)
    
    def run_custom_agent(self, tool: str, agent_script: str) -> Dict[str, Any]:
        """Run a custom agent script in a creative tool."""
        return self.creative_agent.run_custom_agent(tool, agent_script)
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get history of processed tasks."""
        return self.creative_agent.get_task_history()

# Example usage
if __name__ == "__main__":
    registry = AIToolsRegistry()
    
    # Register tools
    registry.register_tool("blender")
    
    # Get registered tools
    tools = registry.get_registered_tools()
    print(f"Registered tools: {tools}")
    
    # Process a creative request
    result = registry.process_creative_request("Create a 3D scene with a rotating cube using Blender")
    print(f"Result: {result}")