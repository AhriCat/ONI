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
import shutil
import base64

logger = logging.getLogger(__name__)

class UnityController:
    """Controller for automating Unity through Python API and command line."""
    
    def __init__(self, unity_path: str = None, project_path: str = None):
        """
        Initialize Unity controller.
        
        Args:
            unity_path: Path to Unity installation
            project_path: Path to Unity project
        """
        self.unity_path = unity_path or self._find_unity_installation()
        self.project_path = project_path
        self.editor_process = None
        self.tcp_port = 12345
        self.tcp_server = None
        self.is_connected = False
        self.temp_dir = Path(tempfile.mkdtemp(prefix="unity_controller_"))
        
    def __del__(self):
        """Clean up resources."""
        if self.editor_process:
            self.close_editor()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _find_unity_installation(self) -> str:
        """Find Unity installation automatically."""
        common_paths = [
            "C:/Program Files/Unity/Hub/Editor/2022.3.18f1/Editor/Unity.exe",
            "C:/Program Files/Unity/Hub/Editor/2021.3.30f1/Editor/Unity.exe",
            "C:/Program Files/Unity/Hub/Editor/2020.3.48f1/Editor/Unity.exe",
            "/Applications/Unity/Hub/Editor/2022.3.18f1/Unity.app/Contents/MacOS/Unity",
            "/opt/unity/Editor/Unity"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Try to find Unity through Unity Hub
        try:
            hub_output = subprocess.check_output(["Unity Hub", "--headless", "editors", "--installed"], text=True)
            lines = hub_output.strip().split('\n')
            if lines and len(lines) > 1:
                latest_version = lines[1].split()[0]
                if os.name == 'nt':  # Windows
                    return f"C:/Program Files/Unity/Hub/Editor/{latest_version}/Editor/Unity.exe"
                elif os.name == 'posix':  # macOS or Linux
                    if os.path.exists(f"/Applications/Unity/Hub/Editor/{latest_version}"):
                        return f"/Applications/Unity/Hub/Editor/{latest_version}/Unity.app/Contents/MacOS/Unity"
                    else:
                        return f"/opt/unity/Editor/{latest_version}/Editor/Unity"
        except:
            pass
        
        raise FileNotFoundError("Unity installation not found")
    
    def start_editor(self, headless: bool = False) -> bool:
        """Start Unity Editor with Python support."""
        try:
            cmd = [self.unity_path]
            
            if self.project_path:
                cmd.extend(["-projectPath", self.project_path])
            
            if headless:
                cmd.extend(["-batchmode", "-nographics"])
            
            # Add TCP server script
            self._create_tcp_server_script()
            cmd.extend([
                "-executeMethod", 
                "UnityPythonBridge.StartTCPServer",
                "-tcpPort", 
                str(self.tcp_port)
            ])
            
            self.editor_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for editor to start
            time.sleep(10)
            
            # Connect to TCP server
            self._connect_to_tcp_server()
            
            logger.info("Unity Editor started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Unity Editor: {e}")
            return False
    
    def _create_tcp_server_script(self):
        """Create C# script for TCP server in Unity."""
        script_dir = self.temp_dir / "Assets" / "Editor"
        script_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = script_dir / "UnityPythonBridge.cs"
        
        script_content = """
using UnityEngine;
using UnityEditor;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;

public class UnityPythonBridge : MonoBehaviour
{
    private static TcpListener server;
    private static Thread serverThread;
    private static int port = 12345;
    private static bool isRunning = false;

    [MenuItem("Tools/Start Python TCP Server")]
    public static void StartTCPServer()
    {
        // Get port from command line arguments
        string[] args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-tcpPort" && i + 1 < args.Length)
            {
                port = int.Parse(args[i + 1]);
            }
        }

        if (!isRunning)
        {
            serverThread = new Thread(new ThreadStart(ServerLoop));
            serverThread.IsBackground = true;
            serverThread.Start();
            Debug.Log($"TCP Server started on port {port}");
        }
    }

    private static void ServerLoop()
    {
        try
        {
            server = new TcpListener(IPAddress.Loopback, port);
            server.Start();
            isRunning = true;

            while (isRunning)
            {
                using (TcpClient client = server.AcceptTcpClient())
                using (NetworkStream stream = client.GetStream())
                {
                    byte[] lengthBytes = new byte[4];
                    stream.Read(lengthBytes, 0, 4);
                    int length = BitConverter.ToInt32(lengthBytes, 0);

                    byte[] buffer = new byte[length];
                    int bytesRead = 0;
                    while (bytesRead < length)
                    {
                        bytesRead += stream.Read(buffer, bytesRead, length - bytesRead);
                    }

                    string message = Encoding.UTF8.GetString(buffer);
                    Debug.Log($"Received: {message}");

                    // Process the command
                    string response = ProcessCommand(message);

                    // Send response
                    byte[] responseData = Encoding.UTF8.GetBytes(response);
                    byte[] responseLengthBytes = BitConverter.GetBytes(responseData.Length);
                    stream.Write(responseLengthBytes, 0, 4);
                    stream.Write(responseData, 0, responseData.Length);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"TCP Server error: {e.Message}");
        }
        finally
        {
            if (server != null)
            {
                server.Stop();
            }
            isRunning = false;
        }
    }

    private static string ProcessCommand(string command)
    {
        try
        {
            // Parse JSON command
            Dictionary<string, object> cmd = JsonUtility.FromJson<Dictionary<string, object>>(command);
            string action = cmd["action"] as string;
            
            switch (action)
            {
                case "execute_editor_method":
                    return ExecuteEditorMethod(cmd);
                case "create_game_object":
                    return CreateGameObject(cmd);
                case "create_material":
                    return CreateMaterial(cmd);
                case "import_asset":
                    return ImportAsset(cmd);
                case "build_project":
                    return BuildProject(cmd);
                default:
                    return JsonUtility.ToJson(new { success = false, error = "Unknown command" });
            }
        }
        catch (Exception e)
        {
            return JsonUtility.ToJson(new { success = false, error = e.Message });
        }
    }

    private static string ExecuteEditorMethod(Dictionary<string, object> cmd)
    {
        string methodName = cmd["method"] as string;
        
        // Execute method using reflection
        // This is a simplified example - in a real implementation, you would use reflection
        // to find and invoke the method with the provided parameters
        
        return JsonUtility.ToJson(new { success = true, message = $"Executed method: {methodName}" });
    }

    private static string CreateGameObject(Dictionary<string, object> cmd)
    {
        string name = cmd["name"] as string;
        
        // Create game object
        GameObject go = new GameObject(name);
        
        return JsonUtility.ToJson(new { success = true, id = go.GetInstanceID() });
    }

    private static string CreateMaterial(Dictionary<string, object> cmd)
    {
        string name = cmd["name"] as string;
        
        // Create material
        Material material = new Material(Shader.Find("Standard"));
        AssetDatabase.CreateAsset(material, $"Assets/Materials/{name}.mat");
        
        return JsonUtility.ToJson(new { success = true, path = $"Assets/Materials/{name}.mat" });
    }

    private static string ImportAsset(Dictionary<string, object> cmd)
    {
        string path = cmd["path"] as string;
        
        // Import asset
        AssetDatabase.ImportAsset(path);
        
        return JsonUtility.ToJson(new { success = true, path = path });
    }

    private static string BuildProject(Dictionary<string, object> cmd)
    {
        string target = cmd["target"] as string;
        
        // Build project
        BuildTarget buildTarget = (BuildTarget)Enum.Parse(typeof(BuildTarget), target);
        BuildPipeline.BuildPlayer(EditorBuildSettings.scenes, "Builds/" + target, buildTarget, BuildOptions.None);
        
        return JsonUtility.ToJson(new { success = true, message = $"Built project for {target}" });
    }
}
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Create a manifest file to ensure the script is compiled
        manifest_path = script_dir / "UnityPythonBridge.asmdef"
        manifest_content = """
{
    "name": "UnityPythonBridge",
    "references": [],
    "includePlatforms": [
        "Editor"
    ],
    "excludePlatforms": [],
    "allowUnsafeCode": false,
    "overrideReferences": false,
    "precompiledReferences": [],
    "autoReferenced": true,
    "defineConstraints": [],
    "versionDefines": [],
    "noEngineReferences": false
}
"""
        with open(manifest_path, 'w') as f:
            f.write(manifest_content)
    
    def _connect_to_tcp_server(self) -> bool:
        """Connect to TCP server in Unity."""
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect(('localhost', self.tcp_port))
            self.is_connected = True
            logger.info(f"Connected to Unity TCP server on port {self.tcp_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Unity TCP server: {e}")
            self.is_connected = False
            return False
    
    def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to Unity TCP server."""
        if not self.is_connected:
            if not self._connect_to_tcp_server():
                return {"success": False, "error": "Not connected to Unity"}
        
        try:
            # Convert command to JSON
            command_json = json.dumps(command)
            command_bytes = command_json.encode('utf-8')
            
            # Send command length
            length_bytes = len(command_bytes).to_bytes(4, byteorder='little')
            self.tcp_socket.sendall(length_bytes)
            
            # Send command
            self.tcp_socket.sendall(command_bytes)
            
            # Receive response length
            response_length_bytes = self.tcp_socket.recv(4)
            response_length = int.from_bytes(response_length_bytes, byteorder='little')
            
            # Receive response
            response_bytes = b''
            while len(response_bytes) < response_length:
                chunk = self.tcp_socket.recv(response_length - len(response_bytes))
                if not chunk:
                    break
                response_bytes += chunk
            
            # Parse response
            response_json = response_bytes.decode('utf-8')
            return json.loads(response_json)
            
        except Exception as e:
            logger.error(f"Failed to send command to Unity: {e}")
            self.is_connected = False
            return {"success": False, "error": str(e)}
    
    def create_game_object(self, name: str) -> Dict[str, Any]:
        """Create a new game object in the scene."""
        command = {
            "action": "create_game_object",
            "name": name
        }
        return self._send_command(command)
    
    def create_material(self, name: str, color: Tuple[float, float, float] = (1, 1, 1)) -> Dict[str, Any]:
        """Create a new material."""
        command = {
            "action": "create_material",
            "name": name,
            "color": color
        }
        return self._send_command(command)
    
    def import_asset(self, file_path: str) -> Dict[str, Any]:
        """Import an asset into the project."""
        command = {
            "action": "import_asset",
            "path": file_path
        }
        return self._send_command(command)
    
    def build_project(self, target: str = "StandaloneWindows64") -> Dict[str, Any]:
        """Build the project for a specific platform."""
        command = {
            "action": "build_project",
            "target": target
        }
        return self._send_command(command)
    
    def execute_editor_method(self, method_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a method in the Unity Editor."""
        command = {
            "action": "execute_editor_method",
            "method": method_name,
            "parameters": parameters or {}
        }
        return self._send_command(command)
    
    def create_terrain(self, size: int = 1000, height: int = 600) -> Dict[str, Any]:
        """Create a terrain in the scene."""
        command = {
            "action": "execute_editor_method",
            "method": "CreateTerrain",
            "parameters": {
                "size": size,
                "height": height
            }
        }
        return self._send_command(command)
    
    def create_character(self, character_type: str = "humanoid") -> Dict[str, Any]:
        """Create a character in the scene."""
        command = {
            "action": "execute_editor_method",
            "method": "CreateCharacter",
            "parameters": {
                "type": character_type
            }
        }
        return self._send_command(command)
    
    def create_animation_controller(self, name: str) -> Dict[str, Any]:
        """Create an animation controller."""
        command = {
            "action": "execute_editor_method",
            "method": "CreateAnimationController",
            "parameters": {
                "name": name
            }
        }
        return self._send_command(command)
    
    def create_particle_system(self, name: str, type: str = "fire") -> Dict[str, Any]:
        """Create a particle system."""
        command = {
            "action": "execute_editor_method",
            "method": "CreateParticleSystem",
            "parameters": {
                "name": name,
                "type": type
            }
        }
        return self._send_command(command)
    
    def create_ui_element(self, element_type: str, parent: str = "Canvas") -> Dict[str, Any]:
        """Create a UI element."""
        command = {
            "action": "execute_editor_method",
            "method": "CreateUIElement",
            "parameters": {
                "type": element_type,
                "parent": parent
            }
        }
        return self._send_command(command)
    
    def create_script(self, script_name: str, script_content: str) -> Dict[str, Any]:
        """Create a C# script."""
        command = {
            "action": "execute_editor_method",
            "method": "CreateScript",
            "parameters": {
                "name": script_name,
                "content": script_content
            }
        }
        return self._send_command(command)
    
    def run_ai_agent(self, agent_script: str) -> Dict[str, Any]:
        """Run an AI agent script in Unity."""
        command = {
            "action": "execute_editor_method",
            "method": "RunAIAgent",
            "parameters": {
                "script": agent_script
            }
        }
        return self._send_command(command)
    
    def close_editor(self) -> bool:
        """Close Unity Editor."""
        try:
            if self.editor_process:
                command = {
                    "action": "execute_editor_method",
                    "method": "QuitEditor",
                    "parameters": {}
                }
                self._send_command(command)
                
                # Wait for editor to close
                self.editor_process.terminate()
                self.editor_process.wait(timeout=10)
                self.editor_process = None
                
                if self.tcp_socket:
                    self.tcp_socket.close()
                    self.tcp_socket = None
                    self.is_connected = False
                
                logger.info("Unity Editor closed successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to close Unity Editor: {e}")
            return False

# Example usage
if __name__ == "__main__":
    controller = UnityController()
    controller.start_editor()
    controller.create_game_object("TestObject")
    controller.create_material("TestMaterial", (1, 0, 0))
    controller.close_editor()