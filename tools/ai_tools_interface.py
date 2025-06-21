import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import threading
import queue
import traceback

# Import AI creative tools
from tools.ai_creative_tools import AICreativeTools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIToolsInterface:
    """Interface for ONI to interact with creative tools."""
    
    def __init__(self):
        """Initialize AI tools interface."""
        self.ai_tools = AICreativeTools()
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
        
    def start_worker(self):
        """Start worker thread to process tasks."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._process_tasks)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            logger.info("Worker thread started")
    
    def stop_worker(self):
        """Stop worker thread."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
            logger.info("Worker thread stopped")
    
    def _process_tasks(self):
        """Process tasks from the queue."""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    continue
                
                task_id = task.get("id")
                task_type = task.get("type")
                task_params = task.get("params", {})
                
                logger.info(f"Processing task {task_id}: {task_type}")
                
                result = self._execute_task(task_type, task_params)
                
                self.result_queue.put({
                    "id": task_id,
                    "type": task_type,
                    "result": result,
                    "timestamp": time.time()
                })
                
                self.task_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                logger.error(traceback.format_exc())
    
    def _execute_task(self, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task based on its type."""
        try:
            if task_type == "initialize_tool":
                tool_name = params.get("tool_name")
                tool_params = params.get("tool_params", {})
                return {"success": self.ai_tools.initialize_tool(tool_name, **tool_params)}
            
            elif task_type == "get_initialized_tools":
                return {"success": True, "tools": self.ai_tools.get_initialized_tools()}
            
            elif task_type == "run_agent":
                tool = params.get("tool")
                agent_script = params.get("agent_script")
                
                if tool == "unreal":
                    return self.ai_tools.run_unreal_agent(agent_script)
                elif tool == "unity":
                    return self.ai_tools.run_unity_agent(agent_script)
                elif tool == "blender":
                    return self.ai_tools.run_blender_agent(agent_script)
                elif tool == "after_effects":
                    return self.ai_tools.run_after_effects_agent(agent_script)
                elif tool == "photoshop":
                    return self.ai_tools.run_photoshop_agent(agent_script)
                else:
                    return {"success": False, "error": f"Unsupported tool: {tool}"}
            
            elif task_type == "create_3d_scene":
                description = params.get("description")
                tool = params.get("tool", "blender")
                output_path = params.get("output_path")
                
                return self.ai_tools.create_3d_scene_from_text(description, tool, output_path)
            
            elif task_type == "create_animation":
                description = params.get("description")
                tool = params.get("tool", "after_effects")
                output_path = params.get("output_path")
                
                return self.ai_tools.create_animation_from_text(description, tool, output_path)
            
            elif task_type == "create_image":
                description = params.get("description")
                tool = params.get("tool", "photoshop")
                output_path = params.get("output_path")
                
                return self.ai_tools.create_image_from_text(description, tool, output_path)
            
            elif task_type == "create_game_character":
                description = params.get("description")
                tool = params.get("tool", "unreal")
                
                return self.ai_tools.create_game_character(description, tool)
            
            elif task_type == "create_visual_effect":
                description = params.get("description")
                tool = params.get("tool", "unreal")
                
                return self.ai_tools.create_visual_effect(description, tool)
            
            elif task_type == "create_ui_design":
                description = params.get("description")
                tool = params.get("tool", "photoshop")
                
                return self.ai_tools.create_ui_design(description, tool)
            
            elif task_type == "create_game_level":
                description = params.get("description")
                tool = params.get("tool", "unreal")
                
                return self.ai_tools.create_game_level(description, tool)
            
            elif task_type == "create_cinematic":
                description = params.get("description")
                tool = params.get("tool", "after_effects")
                
                return self.ai_tools.create_cinematic(description, tool)
            
            else:
                return {"success": False, "error": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error(f"Error executing task {task_type}: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def submit_task(self, task_type: str, params: Dict[str, Any] = None) -> str:
        """Submit a task to the queue."""
        task_id = f"task_{int(time.time())}_{hash(task_type) % 10000}"
        
        task = {
            "id": task_id,
            "type": task_type,
            "params": params or {},
            "timestamp": time.time()
        }
        
        self.task_queue.put(task)
        logger.info(f"Task {task_id} submitted: {task_type}")
        
        # Start worker if not already running
        self.start_worker()
        
        return task_id
    
    def get_result(self, task_id: str = None, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get result from the result queue."""
        try:
            if task_id:
                # Check all results in the queue
                results = []
                found_result = None
                
                # Get all available results
                while not self.result_queue.empty():
                    result = self.result_queue.get(timeout=timeout)
                    if result["id"] == task_id:
                        found_result = result
                    else:
                        results.append(result)
                
                # Put back results that don't match the task_id
                for result in results:
                    self.result_queue.put(result)
                
                return found_result
            else:
                # Get the next result
                return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def wait_for_result(self, task_id: str, timeout: float = None) -> Optional[Dict[str, Any]]:
        """Wait for a specific task result."""
        start_time = time.time()
        
        while timeout is None or time.time() - start_time < timeout:
            result = self.get_result(task_id, 0.1)
            if result:
                return result
            time.sleep(0.1)
        
        return None
    
    def initialize_unreal(self, unreal_path: str = None, project_path: str = None) -> str:
        """Initialize Unreal Engine controller."""
        return self.submit_task("initialize_tool", {
            "tool_name": "unreal",
            "tool_params": {
                "unreal_path": unreal_path,
                "project_path": project_path
            }
        })
    
    def initialize_unity(self, unity_path: str = None, project_path: str = None) -> str:
        """Initialize Unity controller."""
        return self.submit_task("initialize_tool", {
            "tool_name": "unity",
            "tool_params": {
                "unity_path": unity_path,
                "project_path": project_path
            }
        })
    
    def initialize_blender(self, blender_path: str = None) -> str:
        """Initialize Blender controller."""
        return self.submit_task("initialize_tool", {
            "tool_name": "blender",
            "tool_params": {
                "blender_path": blender_path
            }
        })
    
    def initialize_after_effects(self, after_effects_path: str = None) -> str:
        """Initialize After Effects controller."""
        return self.submit_task("initialize_tool", {
            "tool_name": "after_effects",
            "tool_params": {
                "after_effects_path": after_effects_path
            }
        })
    
    def initialize_photoshop(self, photoshop_path: str = None) -> str:
        """Initialize Photoshop controller."""
        return self.submit_task("initialize_tool", {
            "tool_name": "photoshop",
            "tool_params": {
                "photoshop_path": photoshop_path
            }
        })
    
    def create_3d_scene(self, description: str, tool: str = "blender", output_path: str = None) -> str:
        """Create a 3D scene based on text description."""
        return self.submit_task("create_3d_scene", {
            "description": description,
            "tool": tool,
            "output_path": output_path
        })
    
    def create_animation(self, description: str, tool: str = "after_effects", output_path: str = None) -> str:
        """Create an animation based on text description."""
        return self.submit_task("create_animation", {
            "description": description,
            "tool": tool,
            "output_path": output_path
        })
    
    def create_image(self, description: str, tool: str = "photoshop", output_path: str = None) -> str:
        """Create an image based on text description."""
        return self.submit_task("create_image", {
            "description": description,
            "tool": tool,
            "output_path": output_path
        })
    
    def create_game_character(self, description: str, tool: str = "unreal") -> str:
        """Create a game character based on description."""
        return self.submit_task("create_game_character", {
            "description": description,
            "tool": tool
        })
    
    def create_visual_effect(self, description: str, tool: str = "unreal") -> str:
        """Create a visual effect based on description."""
        return self.submit_task("create_visual_effect", {
            "description": description,
            "tool": tool
        })
    
    def create_ui_design(self, description: str, tool: str = "photoshop") -> str:
        """Create a UI design based on description."""
        return self.submit_task("create_ui_design", {
            "description": description,
            "tool": tool
        })
    
    def create_game_level(self, description: str, tool: str = "unreal") -> str:
        """Create a game level based on description."""
        return self.submit_task("create_game_level", {
            "description": description,
            "tool": tool
        })
    
    def create_cinematic(self, description: str, tool: str = "after_effects") -> str:
        """Create a cinematic based on description."""
        return self.submit_task("create_cinematic", {
            "description": description,
            "tool": tool
        })
    
    def run_agent(self, tool: str, agent_script: str) -> str:
        """Run an AI agent script in a creative tool."""
        return self.submit_task("run_agent", {
            "tool": tool,
            "agent_script": agent_script
        })

# Example usage
if __name__ == "__main__":
    interface = AIToolsInterface()
    
    # Initialize Blender
    task_id = interface.initialize_blender()
    result = interface.wait_for_result(task_id, timeout=10)
    print(f"Initialize Blender result: {result}")
    
    # Create a 3D scene
    task_id = interface.create_3d_scene("A rotating cube on a plane", "blender")
    result = interface.wait_for_result(task_id, timeout=30)
    print(f"Create 3D scene result: {result}")
    
    # Stop worker thread
    interface.stop_worker()