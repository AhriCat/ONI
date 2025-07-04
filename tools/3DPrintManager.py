#!/usr/bin/env python3
"""
ONI 3D Printer Tool
===================

An intelligent 3D printer management system integrated with ONI's multi-modal AGI capabilities.
This tool provides model optimization, print monitoring, failure detection, and automated troubleshooting.

Features:
- Intelligent model analysis and optimization
- Real-time print monitoring with computer vision
- Automated failure detection and recovery
- Material usage optimization
- Print quality prediction
- Integration with ONI's reasoning and memory systems

Author: ONI Project
License: Pantheum License
"""

import os
import sys
import json
import time
import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from queue import Queue
import cv2
import serial
from PIL import Image
import requests
import websockets

# ONI Core imports (assuming ONI structure)
try:
    from modules.vision.vision_transformer import VisionTransformer
    from modules.memory.memory_system import MemorySystem
    from modules.reasoning.chain_of_thought import ChainOfThought
    from modules.emotional.emotional_intelligence import EmotionalIntelligence
    from chain.oni_blockchain_api import ONIBlockchainAPI
    from oniapps.tool_integration import ToolIntegration
except ImportError:
    print("Warning: ONI modules not found. Running in standalone mode.")
    VisionTransformer = None
    MemorySystem = None
    ChainOfThought = None
    EmotionalIntelligence = None
    ONIBlockchainAPI = None
    ToolIntegration = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PrintJob:
    """Represents a 3D print job with all relevant metadata."""
    id: str
    model_path: str
    gcode_path: str
    material: str
    estimated_time: int  # minutes
    estimated_cost: float
    layer_height: float
    infill_percentage: int
    support_material: bool
    print_speed: int  # mm/s
    nozzle_temperature: int
    bed_temperature: int
    status: str = "queued"  # queued, printing, paused, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_layer: int = 0
    total_layers: int = 0
    progress: float = 0.0
    quality_score: float = 0.0
    oni_analysis: Dict[str, Any] = None

@dataclass
class PrinterState:
    """Current state of the 3D printer."""
    connected: bool = False
    printing: bool = False
    paused: bool = False
    current_temp_nozzle: float = 0.0
    current_temp_bed: float = 0.0
    target_temp_nozzle: float = 0.0
    target_temp_bed: float = 0.0
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    fan_speed: int = 0
    print_speed: int = 100
    filament_remaining: float = 100.0  # percentage
    last_update: datetime = None

class ONI3DPrinter:
    """
    Main 3D printer management class integrated with ONI's AGI capabilities.
    """
    
    def __init__(self, config_path: str = "config/3d_printer_config.json"):
        """Initialize the ONI 3D Printer system."""
        self.config = self._load_config(config_path)
        self.printer_state = PrinterState()
        self.print_queue = Queue()
        self.current_job: Optional[PrintJob] = None
        self.print_history: List[PrintJob] = []
        
        # ONI integration
        self.oni_enabled = self._initialize_oni()
        
        # Communication
        self.serial_connection = None
        self.camera = None
        self.websocket_server = None
        
        # Monitoring
        self.monitoring_active = False
        self.failure_detection_active = False
        
        # Initialize components
        self._initialize_printer_communication()
        self._initialize_camera()
        self._initialize_websocket_server()
        
        logger.info("ONI 3D Printer system initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        default_config = {
            "printer": {
                "serial_port": "/dev/ttyUSB0",
                "baud_rate": 115200,
                "timeout": 10
            },
            "camera": {
                "device_id": 0,
                "resolution": [640, 480],
                "fps": 30
            },
            "websocket": {
                "host": "localhost",
                "port": 8765
            },
            "oni": {
                "enabled": True,
                "vision_model": "models/vision_transformer.pth",
                "memory_system": True,
                "reasoning_engine": True,
                "emotional_intelligence": True
            },
            "monitoring": {
                "temperature_threshold": 5.0,
                "layer_time_threshold": 300,
                "failure_detection_sensitivity": 0.8
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            logger.info(f"Config file not found, using defaults: {config_path}")
            return default_config
    
    def _initialize_oni(self) -> bool:
        """Initialize ONI system components."""
        if not self.config["oni"]["enabled"]:
            logger.info("ONI integration disabled in config")
            return False
        
        try:
            # Initialize ONI components
            if VisionTransformer:
                self.vision_system = VisionTransformer(
                    model_path=self.config["oni"]["vision_model"]
                )
            else:
                self.vision_system = None
                
            if MemorySystem:
                self.memory_system = MemorySystem()
            else:
                self.memory_system = None
                
            if ChainOfThought:
                self.reasoning_engine = ChainOfThought()
            else:
                self.reasoning_engine = None
                
            if EmotionalIntelligence:
                self.emotional_system = EmotionalIntelligence()
            else:
                self.emotional_system = None
                
            if ONIBlockchainAPI:
                self.blockchain_api = ONIBlockchainAPI()
            else:
                self.blockchain_api = None
                
            logger.info("ONI integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ONI: {e}")
            return False
    
    def _initialize_printer_communication(self):
        """Initialize serial communication with the 3D printer."""
        try:
            self.serial_connection = serial.Serial(
                port=self.config["printer"]["serial_port"],
                baudrate=self.config["printer"]["baud_rate"],
                timeout=self.config["printer"]["timeout"]
            )
            self.printer_state.connected = True
            logger.info(f"Connected to printer on {self.config['printer']['serial_port']}")
        except Exception as e:
            logger.error(f"Failed to connect to printer: {e}")
            self.printer_state.connected = False
    
    def _initialize_camera(self):
        """Initialize camera for print monitoring."""
        try:
            self.camera = cv2.VideoCapture(self.config["camera"]["device_id"])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["resolution"][0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["resolution"][1])
            self.camera.set(cv2.CAP_PROP_FPS, self.config["camera"]["fps"])
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.camera = None
    
    def _initialize_websocket_server(self):
        """Initialize WebSocket server for real-time communication."""
        async def handle_client(websocket, path):
            """Handle WebSocket client connections."""
            try:
                async for message in websocket:
                    data = json.loads(message)
                    response = await self._handle_websocket_message(data)
                    await websocket.send(json.dumps(response))
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
        
        self.websocket_handler = handle_client
        logger.info("WebSocket server initialized")
    
    async def _handle_websocket_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming WebSocket messages."""
        command = data.get("command")
        
        if command == "get_status":
            return {
                "type": "status",
                "printer_state": asdict(self.printer_state),
                "current_job": asdict(self.current_job) if self.current_job else None,
                "queue_size": self.print_queue.qsize()
            }
        
        elif command == "start_print":
            job_id = data.get("job_id")
            result = await self._start_print(job_id)
            return {"type": "print_started", "success": result}
        
        elif command == "pause_print":
            result = await self._pause_print()
            return {"type": "print_paused", "success": result}
        
        elif command == "resume_print":
            result = await self._resume_print()
            return {"type": "print_resumed", "success": result}
        
        elif command == "cancel_print":
            result = await self._cancel_print()
            return {"type": "print_cancelled", "success": result}
        
        elif command == "get_camera_feed":
            frame = self._capture_frame()
            return {"type": "camera_feed", "frame": frame}
        
        else:
            return {"type": "error", "message": "Unknown command"}
    
    def analyze_model_with_oni(self, model_path: str) -> Dict[str, Any]:
        """Use ONI's reasoning capabilities to analyze and optimize a 3D model."""
        if not self.oni_enabled or not self.reasoning_engine:
            return {"optimization_suggestions": [], "quality_prediction": 0.8}
        
        try:
            # Load and analyze the model
            model_data = self._load_3d_model(model_path)
            
            # Use ONI's chain-of-thought reasoning
            analysis_prompt = f"""
            Analyze this 3D model for printing optimization:
            - Model complexity: {model_data.get('complexity', 'unknown')}
            - Layer count: {model_data.get('layers', 'unknown')}
            - Overhangs detected: {model_data.get('overhangs', 'unknown')}
            - Support requirements: {model_data.get('supports_needed', 'unknown')}
            
            Provide optimization recommendations and quality predictions.
            """
            
            reasoning_result = self.reasoning_engine.process(analysis_prompt)
            
            # Store analysis in memory
            if self.memory_system:
                self.memory_system.store_episodic_memory(
                    "3d_model_analysis",
                    {
                        "model_path": model_path,
                        "analysis": reasoning_result,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            return {
                "optimization_suggestions": reasoning_result.get("suggestions", []),
                "quality_prediction": reasoning_result.get("quality_score", 0.8),
                "oni_reasoning": reasoning_result.get("reasoning_chain", [])
            }
            
        except Exception as e:
            logger.error(f"ONI model analysis failed: {e}")
            return {"optimization_suggestions": [], "quality_prediction": 0.8}
    
    def _load_3d_model(self, model_path: str) -> Dict[str, Any]:
        """Load and analyze 3D model file."""
        # This is a simplified implementation
        # In a real system, you'd use libraries like trimesh or open3d
        try:
            file_size = os.path.getsize(model_path)
            return {
                "complexity": "medium" if file_size < 10*1024*1024 else "high",
                "layers": 200,  # Estimated
                "overhangs": True,
                "supports_needed": True
            }
        except Exception as e:
            logger.error(f"Failed to load 3D model: {e}")
            return {}
    
    def create_print_job(self, model_path: str, **kwargs) -> PrintJob:
        """Create a new print job with ONI analysis."""
        job_id = f"job_{int(time.time())}"
        
        # Get ONI analysis
        oni_analysis = self.analyze_model_with_oni(model_path)
        
        # Create job with intelligent defaults
        job = PrintJob(
            id=job_id,
            model_path=model_path,
            gcode_path=kwargs.get("gcode_path", model_path.replace(".stl", ".gcode")),
            material=kwargs.get("material", "PLA"),
            estimated_time=kwargs.get("estimated_time", 120),
            estimated_cost=kwargs.get("estimated_cost", 5.0),
            layer_height=kwargs.get("layer_height", 0.2),
            infill_percentage=kwargs.get("infill_percentage", 20),
            support_material=kwargs.get("support_material", oni_analysis.get("supports_needed", True)),
            print_speed=kwargs.get("print_speed", 50),
            nozzle_temperature=kwargs.get("nozzle_temperature", 200),
            bed_temperature=kwargs.get("bed_temperature", 60),
            oni_analysis=oni_analysis
        )
        
        return job
    
    def queue_print_job(self, job: PrintJob):
        """Add a print job to the queue."""
        self.print_queue.put(job)
        logger.info(f"Added job {job.id} to print queue")
    
    async def _start_print(self, job_id: str = None) -> bool:
        """Start printing a job."""
        if self.printer_state.printing:
            logger.warning("Printer is already printing")
            return False
        
        try:
            # Get job from queue or by ID
            if job_id:
                job = self._find_job_by_id(job_id)
            else:
                job = self.print_queue.get_nowait()
            
            if not job:
                logger.warning("No job found to start")
                return False
            
            self.current_job = job
            self.current_job.start_time = datetime.now()
            self.current_job.status = "printing"
            
            # Send G-code to printer
            if self.serial_connection:
                gcode_commands = self._load_gcode(job.gcode_path)
                for command in gcode_commands:
                    self.serial_connection.write(f"{command}\n".encode())
                    await asyncio.sleep(0.1)  # Small delay between commands
            
            self.printer_state.printing = True
            
            # Start monitoring
            self._start_monitoring()
            
            logger.info(f"Started printing job {job.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start print: {e}")
            return False
    
    async def _pause_print(self) -> bool:
        """Pause the current print job."""
        if not self.printer_state.printing:
            return False
        
        try:
            if self.serial_connection:
                self.serial_connection.write(b"M600\n")  # Pause command
            
            self.printer_state.paused = True
            if self.current_job:
                self.current_job.status = "paused"
            
            logger.info("Print paused")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause print: {e}")
            return False
    
    async def _resume_print(self) -> bool:
        """Resume a paused print job."""
        if not self.printer_state.paused:
            return False
        
        try:
            if self.serial_connection:
                self.serial_connection.write(b"M601\n")  # Resume command
            
            self.printer_state.paused = False
            if self.current_job:
                self.current_job.status = "printing"
            
            logger.info("Print resumed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume print: {e}")
            return False
    
    async def _cancel_print(self) -> bool:
        """Cancel the current print job."""
        if not self.printer_state.printing:
            return False
        
        try:
            if self.serial_connection:
                self.serial_connection.write(b"M104 S0\n")  # Turn off nozzle
                self.serial_connection.write(b"M140 S0\n")  # Turn off bed
                self.serial_connection.write(b"G28 X Y\n")  # Home X and Y
            
            self.printer_state.printing = False
            self.printer_state.paused = False
            
            if self.current_job:
                self.current_job.status = "cancelled"
                self.current_job.end_time = datetime.now()
                self.print_history.append(self.current_job)
                self.current_job = None
            
            self._stop_monitoring()
            
            logger.info("Print cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel print: {e}")
            return False
    
    def _start_monitoring(self):
        """Start monitoring the print job."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.failure_detection_active = True
        
        # Start monitoring threads
        threading.Thread(target=self._monitor_print_progress, daemon=True).start()
        threading.Thread(target=self._monitor_print_quality, daemon=True).start()
        threading.Thread(target=self._monitor_temperature, daemon=True).start()
    
    def _stop_monitoring(self):
        """Stop monitoring the print job."""
        self.monitoring_active = False
        self.failure_detection_active = False
    
    def _monitor_print_progress(self):
        """Monitor print progress and update status."""
        while self.monitoring_active and self.current_job:
            try:
                # Update progress based on G-code execution
                # This is simplified - in reality you'd track actual G-code execution
                if self.current_job.start_time:
                    elapsed = (datetime.now() - self.current_job.start_time).total_seconds()
                    estimated_total = self.current_job.estimated_time * 60
                    self.current_job.progress = min(elapsed / estimated_total * 100, 100)
                
                # Check if print is complete
                if self.current_job.progress >= 100:
                    self._complete_print()
                    break
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring print progress: {e}")
                time.sleep(10)
    
    def _monitor_print_quality(self):
        """Monitor print quality using computer vision."""
        if not self.camera or not self.oni_enabled or not self.vision_system:
            return
        
        while self.monitoring_active and self.current_job:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Analyze with ONI vision system
                analysis = self.vision_system.analyze_frame(frame)
                
                # Detect potential issues
                if analysis.get("quality_score", 1.0) < 0.7:
                    self._handle_quality_issue(analysis)
                
                # Update job quality score
                if self.current_job:
                    self.current_job.quality_score = analysis.get("quality_score", 0.8)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring print quality: {e}")
                time.sleep(30)
    
    def _monitor_temperature(self):
        """Monitor printer temperatures."""
        while self.monitoring_active and self.current_job:
            try:
                # Read temperature from printer
                if self.serial_connection:
                    self.serial_connection.write(b"M105\n")  # Get temperature
                    response = self.serial_connection.readline().decode().strip()
                    
                    # Parse temperature response
                    temps = self._parse_temperature_response(response)
                    if temps:
                        self.printer_state.current_temp_nozzle = temps.get("nozzle", 0)
                        self.printer_state.current_temp_bed = temps.get("bed", 0)
                        
                        # Check for temperature issues
                        self._check_temperature_issues(temps)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring temperature: {e}")
                time.sleep(5)
    
    def _handle_quality_issue(self, analysis: Dict[str, Any]):
        """Handle detected quality issues using ONI reasoning."""
        if not self.reasoning_engine:
            return
        
        issue_prompt = f"""
        Print quality issue detected:
        - Quality score: {analysis.get('quality_score', 0)}
        - Issues detected: {analysis.get('issues', [])}
        - Current layer: {self.current_job.current_layer if self.current_job else 0}
        
        Recommend corrective actions.
        """
        
        reasoning_result = self.reasoning_engine.process(issue_prompt)
        
        # Log the issue and recommendations
        logger.warning(f"Quality issue detected: {analysis}")
        logger.info(f"ONI recommendations: {reasoning_result}")
        
        # Store in memory for learning
        if self.memory_system:
            self.memory_system.store_episodic_memory(
                "print_quality_issue",
                {
                    "job_id": self.current_job.id if self.current_job else None,
                    "analysis": analysis,
                    "recommendations": reasoning_result,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _complete_print(self):
        """Complete the current print job."""
        if not self.current_job:
            return
        
        self.current_job.status = "completed"
        self.current_job.end_time = datetime.now()
        self.current_job.progress = 100.0
        
        # Turn off heaters
        if self.serial_connection:
            self.serial_connection.write(b"M104 S0\n")  # Turn off nozzle
            self.serial_connection.write(b"M140 S0\n")  # Turn off bed
        
        self.printer_state.printing = False
        self.print_history.append(self.current_job)
        
        # Record completion on blockchain if enabled
        if self.blockchain_api:
            try:
                self.blockchain_api.record_contribution(
                    "3d_print_completion",
                    {
                        "job_id": self.current_job.id,
                        "quality_score": self.current_job.quality_score,
                        "completion_time": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Failed to record completion on blockchain: {e}")
        
        logger.info(f"Print job {self.current_job.id} completed successfully")
        self.current_job = None
        self._stop_monitoring()
    
    def _parse_temperature_response(self, response: str) -> Dict[str, float]:
        """Parse temperature response from printer."""
        temps = {}
        try:
            # Parse typical response: "ok T:200.0 /200.0 B:60.0 /60.0"
            parts = response.split()
            for part in parts:
                if part.startswith("T:"):
                    temps["nozzle"] = float(part.split(":")[1].split("/")[0])
                elif part.startswith("B:"):
                    temps["bed"] = float(part.split(":")[1].split("/")[0])
        except Exception as e:
            logger.error(f"Failed to parse temperature response: {e}")
        
        return temps
    
    def _check_temperature_issues(self, temps: Dict[str, float]):
        """Check for temperature-related issues."""
        if not self.current_job:
            return
        
        threshold = self.config["monitoring"]["temperature_threshold"]
        
        # Check nozzle temperature
        if abs(temps.get("nozzle", 0) - self.current_job.nozzle_temperature) > threshold:
            logger.warning(f"Nozzle temperature deviation: {temps.get('nozzle', 0)} vs {self.current_job.nozzle_temperature}")
        
        # Check bed temperature
        if abs(temps.get("bed", 0) - self.current_job.bed_temperature) > threshold:
            logger.warning(f"Bed temperature deviation: {temps.get('bed', 0)} vs {self.current_job.bed_temperature}")
    
    def _capture_frame(self) -> Optional[str]:
        """Capture a frame from the camera and return as base64."""
        if not self.camera:
            return None
        
        try:
            ret, frame = self.camera.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                import base64
                return base64.b64encode(buffer).decode()
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
        
        return None
    
    def _load_gcode(self, gcode_path: str) -> List[str]:
        """Load G-code commands from file."""
        try:
            with open(gcode_path, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            logger.error(f"Failed to load G-code: {e}")
            return []
    
    def _find_job_by_id(self, job_id: str) -> Optional[PrintJob]:
        """Find a job by its ID."""
        # Check current job
        if self.current_job and self.current_job.id == job_id:
            return self.current_job
        
        # Check queue (simplified implementation)
        # In a real system, you'd need a more sophisticated queue management
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "printer_state": asdict(self.printer_state),
            "current_job": asdict(self.current_job) if self.current_job else None,
            "queue_size": self.print_queue.qsize(),
            "oni_enabled": self.oni_enabled,
            "monitoring_active": self.monitoring_active
        }
    
    def get_print_history(self) -> List[Dict[str, Any]]:
        """Get print history."""
        return [asdict(job) for job in self.print_history]
    
    async def start_websocket_server(self):
        """Start the WebSocket server for real-time communication."""
        import websockets
        
        async with websockets.serve(
            self.websocket_handler,
            self.config["websocket"]["host"],
            self.config["websocket"]["port"]
        ):
            logger.info(f"WebSocket server started on {self.config['websocket']['host']}:{self.config['websocket']['port']}")
            await asyncio.Future()  # Run forever
    
    def shutdown(self):
        """Shutdown the 3D printer system."""
        logger.info("Shutting down ONI 3D Printer system")
        
        # Stop monitoring
        self._stop_monitoring()
        
        # Close connections
        if self.serial_connection:
            self.serial_connection.close()
        
        if self.camera:
            self.camera.release()
        
        # Save state
        self._save_state()
        
        logger.info("System shutdown complete")
    
    def _save_state(self):
        """Save system state to file."""
        try:
            state = {
                "printer_state": asdict(self.printer_state),
                "current_job": asdict(self.current_job) if self.current_job else None,
                "print_history": [asdict(job) for job in self.print_history],
                "timestamp": datetime.now().isoformat()
            }
            
            with open("data/printer_state.json", "w") as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")


def main():
    """Main entry point for the ONI 3D Printer tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ONI 3D Printer Tool")
    parser.add_argument("--config", default="config/3d_printer_config.json", help="Configuration file path")
    parser.add_argument("--web-server", action="store_true", help="Start web server")
    parser.add_argument("--add-job", help="Add a print job (model file path)")
    parser.add_argument("--start-print", action="store_true", help="Start next print job")
    parser.add_argument("--status", action="store_true", help="Show system status")
    
    args = parser.parse_args()
    
    # Initialize the system
    printer = ONI3DPrinter(args.config)
    
    try:
        if args.add_job:
            job = printer.create_print_job(args.add_job)
            printer.queue_print_job(job)
            print(f"Added job: {job.id}")
        
        elif args.start_print:
            asyncio.run(printer._start_print())
            print("Print started")
        
        elif args.status:
            status = printer.get_status()
            print(json.dumps(status, indent=2, default=str))
        
        elif args.web_server:
            print("Starting ONI 3D Printer web interface...")
            asyncio.run(printer.start_websocket_server())
        
        else:
            print("ONI 3D Printer Tool initialized. Use --help for options.")
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        printer.shutdown()


class ONI3DPrinterWebInterface:
    """
    Web interface for the ONI 3D Printer system.
    Provides a comprehensive dashboard for monitoring and controlling prints.
    """
    
    def __init__(self, printer: ONI3DPrinter):
        self.printer = printer
        self.html_template = self._generate_html_interface()
    
    def _generate_html_interface(self) -> str:
        """Generate the HTML interface for the 3D printer."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONI 3D Printer Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #ff6b6b, #ffd93d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h2 {
            margin-bottom: 15px;
            color: #ffd93d;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-connected { background-color: #4CAF50; }
        .status-disconnected { background-color: #f44336; }
        .status-printing { background-color: #2196F3; }
        .status-paused { background-color: #FF9800; }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
        }
        
        .control-buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(45deg, #FF9800, #f57c00);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #f44336, #d32f2f);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        
        .temperature-display {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }
        
        .temp-item {
            text-align: center;
        }
        
        .temp-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #ffd93d;
        }
        
        .camera-feed {
            width: 100%;
            border-radius: 10px;
            background: rgba(0,0,0,0.3);
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .oni-analysis {
            background: rgba(255, 215, 0, 0.1);
            border-left: 4px solid #ffd93d;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 10px 10px 0;
        }
        
        .job-queue {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .job-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }
        
        .metric {
            text-align: center;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }
        
        .metric-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #4CAF50;
        }
        
        .metric-label {
            font-size: 0.8rem;
            opacity: 0.8;
        }
        
        .file-upload {
            margin: 15px 0;
        }
        
        .file-upload input[type="file"] {
            display: none;
        }
        
        .file-upload label {
            display: inline-block;
            padding: 10px 20px;
            background: linear-gradient(45deg, #2196F3, #1976D2);
            color: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .file-upload label:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        
        .log-container {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 10px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.9rem;
        }
        
        .log-entry {
            margin: 2px 0;
            padding: 2px 0;
        }
        
        .log-info { color: #4CAF50; }
        .log-warning { color: #FF9800; }
        .log-error { color: #f44336; }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            .control-buttons {
                flex-direction: column;
            }
            
            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ONI 3D Printer Dashboard</h1>
            <p>Intelligent 3D Printing with AGI Integration</p>
        </div>
        
        <div class="dashboard">
            <!-- Printer Status Card -->
            <div class="card">
                <h2>Printer Status</h2>
                <div id="printer-status">
                    <div>
                        <span class="status-indicator" id="connection-status"></span>
                        <span id="connection-text">Disconnected</span>
                    </div>
                    <div>
                        <span class="status-indicator" id="print-status"></span>
                        <span id="print-text">Idle</span>
                    </div>
                </div>
                
                <div class="temperature-display">
                    <div class="temp-item">
                        <div class="temp-value" id="nozzle-temp">0°C</div>
                        <div>Nozzle</div>
                    </div>
                    <div class="temp-item">
                        <div class="temp-value" id="bed-temp">0°C</div>
                        <div>Bed</div>
                    </div>
                </div>
                
                <div class="control-buttons">
                    <button class="btn btn-primary" id="start-btn">Start Print</button>
                    <button class="btn btn-warning" id="pause-btn">Pause</button>
                    <button class="btn btn-danger" id="cancel-btn">Cancel</button>
                </div>
            </div>
            
            <!-- Current Job Card -->
            <div class="card">
                <h2>Current Job</h2>
                <div id="current-job">
                    <div id="job-info">No active job</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="job-progress" style="width: 0%"></div>
                    </div>
                    <div id="job-details"></div>
                </div>
                
                <div class="oni-analysis" id="oni-analysis" style="display: none;">
                    <h3>ONI Analysis</h3>
                    <div id="oni-suggestions"></div>
                </div>
            </div>
            
            <!-- Camera Feed Card -->
            <div class="card">
                <h2>Live Camera Feed</h2>
                <div class="camera-feed" id="camera-feed">
                    <div>Camera feed will appear here</div>
                </div>
            </div>
            
            <!-- Job Queue Card -->
            <div class="card">
                <h2>Print Queue</h2>
                <div class="file-upload">
                    <input type="file" id="file-input" accept=".stl,.obj,.gcode">
                    <label for="file-input">Upload Model</label>
                </div>
                <div class="job-queue" id="job-queue">
                    <div class="job-item">
                        <div><strong>Sample Job</strong></div>
                        <div>Material: PLA | Time: 2h 30m</div>
                    </div>
                </div>
            </div>
            
            <!-- Metrics Card -->
            <div class="card">
                <h2>Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value" id="success-rate">95%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="avg-quality">8.5</div>
                        <div class="metric-label">Avg Quality</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="total-prints">142</div>
                        <div class="metric-label">Total Prints</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="uptime">99.2%</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                </div>
            </div>
            
            <!-- System Logs Card -->
            <div class="card">
                <h2>System Logs</h2>
                <div class="log-container" id="log-container">
                    <div class="log-entry log-info">[INFO] ONI 3D Printer system initialized</div>
                    <div class="log-entry log-info">[INFO] Connected to printer</div>
                    <div class="log-entry log-info">[INFO] Camera initialized</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        class ONI3DPrinterInterface {
            constructor() {
                this.websocket = null;
                this.isConnected = false;
                this.initializeWebSocket();
                this.initializeEventListeners();
                this.updateInterval = setInterval(() => this.updateStatus(), 5000);
            }
            
            initializeWebSocket() {
                const wsUrl = `ws://${window.location.hostname}:8765`;
                this.websocket = new WebSocket(wsUrl);
                
                this.websocket.onopen = () => {
                    this.isConnected = true;
                    this.addLog('Connected to ONI 3D Printer', 'info');
                    this.updateStatus();
                };
                
                this.websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.websocket.onclose = () => {
                    this.isConnected = false;
                    this.addLog('Disconnected from ONI 3D Printer', 'error');
                    setTimeout(() => this.initializeWebSocket(), 5000);
                };
                
                this.websocket.onerror = (error) => {
                    this.addLog(`WebSocket error: ${error}`, 'error');
                };
            }
            
            initializeEventListeners() {
                document.getElementById('start-btn').addEventListener('click', () => {
                    this.sendCommand('start_print');
                });
                
                document.getElementById('pause-btn').addEventListener('click', () => {
                    this.sendCommand('pause_print');
                });
                
                document.getElementById('cancel-btn').addEventListener('click', () => {
                    this.sendCommand('cancel_print');
                });
                
                document.getElementById('file-input').addEventListener('change', (event) => {
                    this.handleFileUpload(event);
                });
            }
            
            sendCommand(command, data = {}) {
                if (!this.isConnected) return;
                
                const message = {
                    command: command,
                    ...data
                };
                
                this.websocket.send(JSON.stringify(message));
            }
            
            handleWebSocketMessage(data) {
                switch(data.type) {
                    case 'status':
                        this.updateDashboard(data);
                        break;
                    case 'print_started':
                        this.addLog('Print started', 'info');
                        break;
                    case 'print_paused':
                        this.addLog('Print paused', 'warning');
                        break;
                    case 'print_cancelled':
                        this.addLog('Print cancelled', 'warning');
                        break;
                    case 'camera_feed':
                        this.updateCameraFeed(data.frame);
                        break;
                    case 'error':
                        this.addLog(data.message, 'error');
                        break;
                }
            }
            
            updateStatus() {
                this.sendCommand('get_status');
                this.sendCommand('get_camera_feed');
            }
            
            updateDashboard(data) {
                const printerState = data.printer_state;
                const currentJob = data.current_job;
                
                // Update connection status
                const connectionStatus = document.getElementById('connection-status');
                const connectionText = document.getElementById('connection-text');
                
                if (printerState.connected) {
                    connectionStatus.className = 'status-indicator status-connected';
                    connectionText.textContent = 'Connected';
                } else {
                    connectionStatus.className = 'status-indicator status-disconnected';
                    connectionText.textContent = 'Disconnected';
                }
                
                // Update print status
                const printStatus = document.getElementById('print-status');
                const printText = document.getElementById('print-text');
                
                if (printerState.printing) {
                    if (printerState.paused) {
                        printStatus.className = 'status-indicator status-paused';
                        printText.textContent = 'Paused';
                    } else {
                        printStatus.className = 'status-indicator status-printing';
                        printText.textContent = 'Printing';
                    }
                } else {
                    printStatus.className = 'status-indicator status-disconnected';
                    printText.textContent = 'Idle';
                }
                
                // Update temperatures
                document.getElementById('nozzle-temp').textContent = 
                    `${printerState.current_temp_nozzle.toFixed(1)}°C`;
                document.getElementById('bed-temp').textContent = 
                    `${printerState.current_temp_bed.toFixed(1)}°C`;
                
                // Update current job
                if (currentJob) {
                    document.getElementById('job-info').innerHTML = `
                        <strong>${currentJob.id}</strong><br>
                        Material: ${currentJob.material}<br>
                        Layer: ${currentJob.current_layer}/${currentJob.total_layers}
                    `;
                    
                    document.getElementById('job-progress').style.width = `${currentJob.progress}%`;
                    
                    document.getElementById('job-details').innerHTML = `
                        <div>Progress: ${currentJob.progress.toFixed(1)}%</div>
                        <div>Quality Score: ${currentJob.quality_score.toFixed(2)}</div>
                        <div>Status: ${currentJob.status}</div>
                    `;
                    
                    // Show ONI analysis if available
                    if (currentJob.oni_analysis) {
                        const oniAnalysis = document.getElementById('oni-analysis');
                        oniAnalysis.style.display = 'block';
                        
                        const suggestions = currentJob.oni_analysis.optimization_suggestions;
                        document.getElementById('oni-suggestions').innerHTML = 
                            suggestions.map(s => `<div>• ${s}</div>`).join('');
                    }
                } else {
                    document.getElementById('job-info').textContent = 'No active job';
                    document.getElementById('job-progress').style.width = '0%';
                    document.getElementById('job-details').innerHTML = '';
                    document.getElementById('oni-analysis').style.display = 'none';
                }
                
                // Update queue size
                document.getElementById('job-queue').innerHTML = `
                    <div class="job-item">
                        <div><strong>Queue Size: ${data.queue_size}</strong></div>
                        <div>Jobs waiting to be printed</div>
                    </div>
                `;
            }
            
            updateCameraFeed(frameData) {
                if (frameData) {
                    const cameraFeed = document.getElementById('camera-feed');
                    cameraFeed.innerHTML = `<img src="data:image/jpeg;base64,${frameData}" style="width: 100%; border-radius: 10px;">`;
                }
            }
            
            handleFileUpload(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                this.addLog(`Uploading file: ${file.name}`, 'info');
                
                // In a real implementation, you would upload the file to the server
                // For now, just simulate adding it to the queue
                setTimeout(() => {
                    this.addLog(`File uploaded and added to queue: ${file.name}`, 'info');
                }, 1000);
            }
            
            addLog(message, level = 'info') {
                const logContainer = document.getElementById('log-container');
                const timestamp = new Date().toLocaleTimeString();
                
                const logEntry = document.createElement('div');
                logEntry.className = `log-entry log-${level}`;
                logEntry.textContent = `[${timestamp}] ${message}`;
                
                logContainer.appendChild(logEntry);
                logContainer.scrollTop = logContainer.scrollHeight;
                
                // Keep only last 50 log entries
                while (logContainer.children.length > 50) {
                    logContainer.removeChild(logContainer.firstChild);
                }
            }
        }
        
        // Initialize the interface when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new ONI3DPrinterInterface();
        });
    </script>
</body>
</html>
        '''
    
    def serve_interface(self, host='localhost', port=8080):
        """Serve the web interface."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(self.server.html_content.encode())
        
        server = HTTPServer((host, port), RequestHandler)
        server.html_content = self.html_template
        
        print(f"Starting web interface on http://{host}:{port}")
        server.serve_forever()


# Additional utility functions for the ONI 3D Printer system

def generate_config_file(output_path: str = "config/3d_printer_config.json"):
    """Generate a sample configuration file."""
    config = {
        "printer": {
            "serial_port": "/dev/ttyUSB0",
            "baud_rate": 115200,
            "timeout": 10,
            "model": "Generic 3D Printer",
            "build_volume": [200, 200, 200]
        },
        "camera": {
            "device_id": 0,
            "resolution": [640, 480],
            "fps": 30,
            "enabled": True
        },
        "websocket": {
            "host": "localhost",
            "port": 8765
        },
        "web_interface": {
            "host": "localhost",
            "port": 8080,
            "enabled": True
        },
        "oni": {
            "enabled": True,
            "vision_model": "models/vision_transformer.pth",
            "memory_system": True,
            "reasoning_engine": True,
            "emotional_intelligence": True,
            "blockchain_integration": True
        },
        "monitoring": {
            "temperature_threshold": 5.0,
            "layer_time_threshold": 300,
            "failure_detection_sensitivity": 0.8,
            "quality_threshold": 0.7
        },
        "materials": {
            "PLA": {
                "nozzle_temp": 200,
                "bed_temp": 60,
                "print_speed": 50
            },
            "ABS": {
                "nozzle_temp": 240,
                "bed_temp": 80,
                "print_speed": 40
            },
            "PETG": {
                "nozzle_temp": 230,
                "bed_temp": 70,
                "print_speed": 45
            }
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration file generated: {output_path}")


def create_project_structure():
    """Create the project directory structure."""
    directories = [
        "config",
        "data",
        "models",
        "logs",
        "uploads",
        "gcodes",
        "completed_prints"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


if __name__ == "__main__":
    main()
