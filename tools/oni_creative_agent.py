import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import threading
import queue
import traceback

# Import AI tools interface
from tools.ai_tools_interface import AIToolsInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONICreativeAgent:
    """ONI agent for creative tools integration."""
    
    def __init__(self):
        """Initialize ONI creative agent."""
        self.tools_interface = AIToolsInterface()
        self.tools_interface.start_worker()
        self.task_history = []
        
    def __del__(self):
        """Clean up resources."""
        self.tools_interface.stop_worker()
    
    def process_creative_request(self, request: str) -> Dict[str, Any]:
        """Process a creative request from ONI."""
        try:
            # Parse request to determine task type and parameters
            task_type, params = self._parse_request(request)
            
            # Submit task
            task_id = self.tools_interface.submit_task(task_type, params)
            
            # Wait for result
            result = self.tools_interface.wait_for_result(task_id, timeout=300)  # 5 minutes timeout
            
            if result:
                # Record task in history
                self.task_history.append({
                    "request": request,
                    "task_type": task_type,
                    "params": params,
                    "result": result,
                    "timestamp": time.time()
                })
                
                return {
                    "success": result.get("result", {}).get("success", False),
                    "message": f"Task {task_type} completed successfully" if result.get("result", {}).get("success", False) else f"Task {task_type} failed",
                    "details": result.get("result", {}),
                    "task_id": task_id
                }
            else:
                return {
                    "success": False,
                    "message": f"Task {task_type} timed out",
                    "task_id": task_id
                }
                
        except Exception as e:
            logger.error(f"Error processing creative request: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Error processing request: {str(e)}",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _parse_request(self, request: str) -> Tuple[str, Dict[str, Any]]:
        """Parse a creative request to determine task type and parameters."""
        request_lower = request.lower()
        
        # Check for tool initialization
        if "initialize" in request_lower or "setup" in request_lower or "start" in request_lower:
            if "unreal" in request_lower:
                return "initialize_tool", {"tool_name": "unreal"}
            elif "unity" in request_lower:
                return "initialize_tool", {"tool_name": "unity"}
            elif "blender" in request_lower:
                return "initialize_tool", {"tool_name": "blender"}
            elif "after effects" in request_lower or "aftereffects" in request_lower:
                return "initialize_tool", {"tool_name": "after_effects"}
            elif "photoshop" in request_lower:
                return "initialize_tool", {"tool_name": "photoshop"}
        
        # Check for 3D scene creation
        if "create" in request_lower and ("3d" in request_lower or "scene" in request_lower):
            tool = "blender"
            if "unreal" in request_lower:
                tool = "unreal"
            elif "unity" in request_lower:
                tool = "unity"
            
            return "create_3d_scene", {
                "description": request,
                "tool": tool
            }
        
        # Check for animation creation
        if "create" in request_lower and "animation" in request_lower:
            tool = "after_effects"
            if "blender" in request_lower:
                tool = "blender"
            
            return "create_animation", {
                "description": request,
                "tool": tool
            }
        
        # Check for image creation
        if "create" in request_lower and ("image" in request_lower or "picture" in request_lower or "photo" in request_lower):
            tool = "photoshop"
            
            return "create_image", {
                "description": request,
                "tool": tool
            }
        
        # Check for game character creation
        if "create" in request_lower and "character" in request_lower:
            tool = "unreal"
            if "unity" in request_lower:
                tool = "unity"
            elif "blender" in request_lower:
                tool = "blender"
            
            return "create_game_character", {
                "description": request,
                "tool": tool
            }
        
        # Check for visual effect creation
        if "create" in request_lower and ("effect" in request_lower or "vfx" in request_lower):
            tool = "unreal"
            if "unity" in request_lower:
                tool = "unity"
            elif "blender" in request_lower:
                tool = "blender"
            elif "after effects" in request_lower or "aftereffects" in request_lower:
                tool = "after_effects"
            
            return "create_visual_effect", {
                "description": request,
                "tool": tool
            }
        
        # Check for UI design creation
        if "create" in request_lower and ("ui" in request_lower or "interface" in request_lower or "design" in request_lower):
            tool = "photoshop"
            
            return "create_ui_design", {
                "description": request,
                "tool": tool
            }
        
        # Check for game level creation
        if "create" in request_lower and ("level" in request_lower or "map" in request_lower or "environment" in request_lower):
            tool = "unreal"
            if "unity" in request_lower:
                tool = "unity"
            
            return "create_game_level", {
                "description": request,
                "tool": tool
            }
        
        # Check for cinematic creation
        if "create" in request_lower and ("cinematic" in request_lower or "video" in request_lower or "movie" in request_lower):
            tool = "after_effects"
            if "unreal" in request_lower:
                tool = "unreal"
            
            return "create_cinematic", {
                "description": request,
                "tool": tool
            }
        
        # Default: create image
        return "create_image", {
            "description": request,
            "tool": "photoshop"
        }
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get history of processed tasks."""
        return self.task_history
    
    def get_initialized_tools(self) -> List[str]:
        """Get list of initialized tools."""
        task_id = self.tools_interface.submit_task("get_initialized_tools")
        result = self.tools_interface.wait_for_result(task_id, timeout=10)
        
        if result and result.get("result", {}).get("success", False):
            return result.get("result", {}).get("tools", [])
        else:
            return []
    
    def run_custom_agent(self, tool: str, agent_script: str) -> Dict[str, Any]:
        """Run a custom agent script in a creative tool."""
        task_id = self.tools_interface.run_agent(tool, agent_script)
        result = self.tools_interface.wait_for_result(task_id, timeout=300)  # 5 minutes timeout
        
        if result:
            return {
                "success": result.get("result", {}).get("success", False),
                "message": f"Agent script executed successfully" if result.get("result", {}).get("success", False) else f"Agent script execution failed",
                "details": result.get("result", {}),
                "task_id": task_id
            }
        else:
            return {
                "success": False,
                "message": f"Agent script execution timed out",
                "task_id": task_id
            }

# Example usage
if __name__ == "__main__":
    agent = ONICreativeAgent()
    
    # Process a creative request
    result = agent.process_creative_request("Create a 3D scene with a rotating cube using Blender")
    print(f"Result: {result}")
    
    # Get initialized tools
    tools = agent.get_initialized_tools()
    print(f"Initialized tools: {tools}")