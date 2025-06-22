import os
import sys
import subprocess
import tempfile
import shutil
import logging
import json
import time
import signal
import resource
import threading
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
import importlib.util
import inspect
import builtins
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecuritySandbox:
    """
    Security sandboxing module for safe execution of code and tools.
    
    This module provides:
    1. Isolated execution environments for untrusted code
    2. Resource limits and timeouts
    3. Permission controls for file and network access
    4. Audit logging of all operations
    """
    
    def __init__(self, sandbox_dir: Optional[str] = None, 
                enable_network: bool = False,
                enable_file_io: bool = False,
                resource_limits: Optional[Dict[str, int]] = None,
                allowed_modules: Optional[List[str]] = None):
        """
        Initialize security sandbox.
        
        Args:
            sandbox_dir: Directory for sandbox operations (optional)
            enable_network: Whether to allow network access
            enable_file_io: Whether to allow file I/O
            resource_limits: Resource limits (optional)
            allowed_modules: List of allowed Python modules (optional)
        """
        self.sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="oni_sandbox_")
        self.enable_network = enable_network
        self.enable_file_io = enable_file_io
        self.resource_limits = resource_limits or {
            'cpu_time': 30,  # seconds
            'memory': 512 * 1024 * 1024,  # 512 MB
            'file_size': 10 * 1024 * 1024,  # 10 MB
            'processes': 5
        }
        self.allowed_modules = allowed_modules or [
            'math', 're', 'json', 'datetime', 'collections', 'itertools',
            'functools', 'operator', 'random', 'statistics', 'uuid'
        ]
        
        # Create sandbox directory
        os.makedirs(self.sandbox_dir, exist_ok=True)
        
        # Audit log
        self.audit_log = []
        
        # Active sandboxes
        self.active_sandboxes = {}
        
        logger.info(f"Initialized security sandbox in {self.sandbox_dir}")
    
    def __del__(self):
        """Clean up resources."""
        try:
            # Terminate any active sandboxes
            for sandbox_id, process in self.active_sandboxes.items():
                try:
                    process.terminate()
                    logger.info(f"Terminated sandbox {sandbox_id}")
                except:
                    pass
            
            # Remove sandbox directory
            if os.path.exists(self.sandbox_dir):
                shutil.rmtree(self.sandbox_dir)
                logger.info(f"Removed sandbox directory {self.sandbox_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up sandbox: {e}")
    
    def execute_python_code(self, code: str, inputs: Optional[Dict[str, Any]] = None, 
                          timeout: int = 30) -> Dict[str, Any]:
        """
        Execute Python code in a sandbox.
        
        Args:
            code: Python code to execute
            inputs: Input variables (optional)
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        # Create a unique ID for this execution
        execution_id = str(uuid.uuid4())
        
        # Create a temporary file for the code
        code_file = os.path.join(self.sandbox_dir, f"{execution_id}.py")
        
        try:
            # Prepare code with sandbox wrapper
            wrapped_code = self._wrap_python_code(code, inputs)
            
            # Write code to file
            with open(code_file, 'w') as f:
                f.write(wrapped_code)
            
            # Log execution
            self._log_audit_event(
                event_type="execute_python",
                details={
                    "execution_id": execution_id,
                    "code_length": len(code),
                    "timeout": timeout,
                    "enable_network": self.enable_network,
                    "enable_file_io": self.enable_file_io
                }
            )
            
            # Execute code in subprocess
            start_time = time.time()
            
            # Prepare command
            cmd = [sys.executable, code_file]
            
            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = self.sandbox_dir
            
            if not self.enable_network:
                # Block network access (Unix-like systems)
                if os.name == 'posix':
                    env["ONI_SANDBOX_BLOCK_NETWORK"] = "1"
            
            # Execute process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # Store active sandbox
            self.active_sandboxes[execution_id] = process
            
            try:
                # Wait for process to complete with timeout
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Remove from active sandboxes
                del self.active_sandboxes[execution_id]
                
                # Check for errors
                if process.returncode != 0:
                    logger.error(f"Code execution failed: {stderr}")
                    return {
                        'success': False,
                        'error': stderr,
                        'execution_time': time.time() - start_time
                    }
                
                # Parse output
                try:
                    # Look for JSON output marker
                    output_start = stdout.find("__ONI_SANDBOX_OUTPUT_START__")
                    output_end = stdout.find("__ONI_SANDBOX_OUTPUT_END__")
                    
                    if output_start >= 0 and output_end >= 0:
                        output_json = stdout[output_start + len("__ONI_SANDBOX_OUTPUT_START__"):output_end].strip()
                        output = json.loads(output_json)
                    else:
                        output = {'stdout': stdout}
                except json.JSONDecodeError:
                    output = {'stdout': stdout}
                
                return {
                    'success': True,
                    'output': output,
                    'execution_time': time.time() - start_time
                }
                
            except subprocess.TimeoutExpired:
                # Terminate process
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                # Remove from active sandboxes
                del self.active_sandboxes[execution_id]
                
                logger.error(f"Code execution timed out after {timeout}s")
                return {
                    'success': False,
                    'error': f"Execution timed out after {timeout} seconds",
                    'execution_time': timeout
                }
                
        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        finally:
            # Clean up
            if os.path.exists(code_file):
                os.remove(code_file)
    
    def execute_shell_command(self, command: str, timeout: int = 30, 
                            working_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a shell command in a sandbox.
        
        Args:
            command: Shell command to execute
            timeout: Execution timeout in seconds
            working_dir: Working directory (optional)
            
        Returns:
            Dictionary with execution results
        """
        # Create a unique ID for this execution
        execution_id = str(uuid.uuid4())
        
        # Set working directory
        if working_dir:
            if not os.path.isabs(working_dir):
                working_dir = os.path.join(self.sandbox_dir, working_dir)
            os.makedirs(working_dir, exist_ok=True)
        else:
            working_dir = self.sandbox_dir
        
        try:
            # Check if command is allowed
            if not self._is_command_allowed(command):
                return {
                    'success': False,
                    'error': f"Command not allowed: {command}"
                }
            
            # Log execution
            self._log_audit_event(
                event_type="execute_shell",
                details={
                    "execution_id": execution_id,
                    "command": command,
                    "timeout": timeout,
                    "working_dir": working_dir
                }
            )
            
            # Execute command
            start_time = time.time()
            
            # Set up environment
            env = os.environ.copy()
            
            if not self.enable_network:
                # Block network access (Unix-like systems)
                if os.name == 'posix':
                    env["ONI_SANDBOX_BLOCK_NETWORK"] = "1"
            
            # Execute process
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                env=env,
                text=True
            )
            
            # Store active sandbox
            self.active_sandboxes[execution_id] = process
            
            try:
                # Wait for process to complete with timeout
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Remove from active sandboxes
                del self.active_sandboxes[execution_id]
                
                # Check for errors
                if process.returncode != 0:
                    logger.error(f"Command execution failed: {stderr}")
                    return {
                        'success': False,
                        'error': stderr,
                        'execution_time': time.time() - start_time
                    }
                
                return {
                    'success': True,
                    'stdout': stdout,
                    'stderr': stderr,
                    'execution_time': time.time() - start_time
                }
                
            except subprocess.TimeoutExpired:
                # Terminate process
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                # Remove from active sandboxes
                del self.active_sandboxes[execution_id]
                
                logger.error(f"Command execution timed out after {timeout}s")
                return {
                    'success': False,
                    'error': f"Execution timed out after {timeout} seconds",
                    'execution_time': timeout
                }
                
        except Exception as e:
            logger.error(f"Error executing shell command: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def execute_function_in_sandbox(self, func: Callable, args: tuple = (), 
                                  kwargs: Dict[str, Any] = None,
                                  timeout: int = 30) -> Dict[str, Any]:
        """
        Execute a function in a sandbox.
        
        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        # Get function source code
        try:
            source = inspect.getsource(func)
        except (TypeError, OSError):
            return {
                'success': False,
                'error': "Could not get function source code"
            }
        
        # Get function name
        func_name = func.__name__
        
        # Create wrapper code
        wrapper_code = f"""
{source}

# Execute function with arguments
import json
import sys
import traceback

try:
    result = {func_name}(*{args}, **{kwargs or {}})
    
    # Print output marker
    print("__ONI_SANDBOX_OUTPUT_START__")
    print(json.dumps({{"result": result}}))
    print("__ONI_SANDBOX_OUTPUT_END__")
    
except Exception as e:
    print("__ONI_SANDBOX_OUTPUT_START__")
    print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}))
    print("__ONI_SANDBOX_OUTPUT_END__")
    sys.exit(1)
"""
        
        # Execute the wrapper code
        return self.execute_python_code(wrapper_code, timeout=timeout)
    
    def create_secure_module(self, module_name: str, code: str) -> Dict[str, Any]:
        """
        Create a secure Python module that can be imported.
        
        Args:
            module_name: Name for the module
            code: Python code for the module
            
        Returns:
            Dictionary with module creation results
        """
        # Create a unique ID for this module
        module_id = hashlib.md5(code.encode()).hexdigest()[:8]
        safe_module_name = f"oni_secure_{module_name}_{module_id}"
        
        # Create module file
        module_file = os.path.join(self.sandbox_dir, f"{safe_module_name}.py")
        
        try:
            # Write code to file
            with open(module_file, 'w') as f:
                f.write(code)
            
            # Log module creation
            self._log_audit_event(
                event_type="create_module",
                details={
                    "module_name": safe_module_name,
                    "code_length": len(code),
                    "module_id": module_id
                }
            )
            
            # Add sandbox directory to Python path
            if self.sandbox_dir not in sys.path:
                sys.path.append(self.sandbox_dir)
            
            # Import the module
            spec = importlib.util.spec_from_file_location(safe_module_name, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return {
                'success': True,
                'module_name': safe_module_name,
                'module': module,
                'module_file': module_file
            }
            
        except Exception as e:
            logger.error(f"Error creating secure module: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def create_virtual_environment(self, env_name: str, 
                                 packages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a virtual environment for isolated package installation.
        
        Args:
            env_name: Name for the virtual environment
            packages: List of packages to install (optional)
            
        Returns:
            Dictionary with virtual environment creation results
        """
        # Create a unique ID for this environment
        env_id = str(uuid.uuid4())[:8]
        safe_env_name = f"oni_env_{env_name}_{env_id}"
        
        # Create environment directory
        env_dir = os.path.join(self.sandbox_dir, safe_env_name)
        
        try:
            # Log environment creation
            self._log_audit_event(
                event_type="create_venv",
                details={
                    "env_name": safe_env_name,
                    "packages": packages,
                    "env_id": env_id
                }
            )
            
            # Create virtual environment
            venv_cmd = [sys.executable, "-m", "venv", env_dir]
            subprocess.run(venv_cmd, check=True)
            
            # Install packages if provided
            if packages:
                # Get pip path
                if os.name == 'nt':  # Windows
                    pip_path = os.path.join(env_dir, "Scripts", "pip")
                else:  # Unix-like
                    pip_path = os.path.join(env_dir, "bin", "pip")
                
                # Install each package
                for package in packages:
                    # Check if package is allowed
                    if not self._is_package_allowed(package):
                        logger.warning(f"Package not allowed: {package}")
                        continue
                        
                    pip_cmd = [pip_path, "install", package]
                    subprocess.run(pip_cmd, check=True)
            
            return {
                'success': True,
                'env_name': safe_env_name,
                'env_dir': env_dir,
                'python_path': os.path.join(env_dir, "bin", "python") if os.name != 'nt' else os.path.join(env_dir, "Scripts", "python")
            }
            
        except Exception as e:
            logger.error(f"Error creating virtual environment: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def execute_in_virtual_environment(self, env_name: str, code: str, 
                                     timeout: int = 60) -> Dict[str, Any]:
        """
        Execute Python code in a virtual environment.
        
        Args:
            env_name: Name of the virtual environment
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        # Find environment directory
        env_dir = None
        for dirname in os.listdir(self.sandbox_dir):
            if dirname.startswith(f"oni_env_{env_name}_"):
                env_dir = os.path.join(self.sandbox_dir, dirname)
                break
        
        if not env_dir:
            return {
                'success': False,
                'error': f"Virtual environment '{env_name}' not found"
            }
        
        # Get Python path
        if os.name == 'nt':  # Windows
            python_path = os.path.join(env_dir, "Scripts", "python")
        else:  # Unix-like
            python_path = os.path.join(env_dir, "bin", "python")
        
        # Create a temporary file for the code
        execution_id = str(uuid.uuid4())
        code_file = os.path.join(self.sandbox_dir, f"{execution_id}.py")
        
        try:
            # Write code to file
            with open(code_file, 'w') as f:
                f.write(code)
            
            # Log execution
            self._log_audit_event(
                event_type="execute_in_venv",
                details={
                    "execution_id": execution_id,
                    "env_name": env_name,
                    "code_length": len(code),
                    "timeout": timeout
                }
            )
            
            # Execute code
            start_time = time.time()
            
            # Set up environment
            env = os.environ.copy()
            
            if not self.enable_network:
                # Block network access (Unix-like systems)
                if os.name == 'posix':
                    env["ONI_SANDBOX_BLOCK_NETWORK"] = "1"
            
            # Execute process
            process = subprocess.Popen(
                [python_path, code_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # Store active sandbox
            self.active_sandboxes[execution_id] = process
            
            try:
                # Wait for process to complete with timeout
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Remove from active sandboxes
                del self.active_sandboxes[execution_id]
                
                # Check for errors
                if process.returncode != 0:
                    logger.error(f"Code execution in venv failed: {stderr}")
                    return {
                        'success': False,
                        'error': stderr,
                        'execution_time': time.time() - start_time
                    }
                
                return {
                    'success': True,
                    'stdout': stdout,
                    'stderr': stderr,
                    'execution_time': time.time() - start_time
                }
                
            except subprocess.TimeoutExpired:
                # Terminate process
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                # Remove from active sandboxes
                del self.active_sandboxes[execution_id]
                
                logger.error(f"Code execution in venv timed out after {timeout}s")
                return {
                    'success': False,
                    'error': f"Execution timed out after {timeout} seconds",
                    'execution_time': timeout
                }
                
        except Exception as e:
            logger.error(f"Error executing code in virtual environment: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        finally:
            # Clean up
            if os.path.exists(code_file):
                os.remove(code_file)
    
    def execute_in_docker(self, image: str, command: str, 
                        volumes: Optional[Dict[str, str]] = None,
                        environment: Optional[Dict[str, str]] = None,
                        timeout: int = 60) -> Dict[str, Any]:
        """
        Execute a command in a Docker container.
        
        Args:
            image: Docker image to use
            command: Command to execute
            volumes: Volume mappings (optional)
            environment: Environment variables (optional)
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Check if Docker is available
            try:
                subprocess.run(["docker", "--version"], check=True, capture_output=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                return {
                    'success': False,
                    'error': "Docker is not available"
                }
            
            # Create a unique ID for this execution
            execution_id = str(uuid.uuid4())
            container_name = f"oni_sandbox_{execution_id}"
            
            # Prepare Docker command
            docker_cmd = ["docker", "run", "--name", container_name, "--rm"]
            
            # Add resource limits
            docker_cmd.extend([
                "--memory", f"{self.resource_limits['memory']}b",
                "--cpus", "1"
            ])
            
            # Add volumes
            if volumes:
                for host_path, container_path in volumes.items():
                    docker_cmd.extend(["-v", f"{host_path}:{container_path}"])
            
            # Add environment variables
            if environment:
                for key, value in environment.items():
                    docker_cmd.extend(["-e", f"{key}={value}"])
            
            # Add network configuration
            if not self.enable_network:
                docker_cmd.append("--network=none")
            
            # Add image and command
            docker_cmd.append(image)
            docker_cmd.extend(command.split())
            
            # Log execution
            self._log_audit_event(
                event_type="execute_docker",
                details={
                    "execution_id": execution_id,
                    "container_name": container_name,
                    "image": image,
                    "command": command,
                    "timeout": timeout
                }
            )
            
            # Execute Docker command
            start_time = time.time()
            
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store active sandbox
            self.active_sandboxes[execution_id] = process
            
            try:
                # Wait for process to complete with timeout
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Remove from active sandboxes
                del self.active_sandboxes[execution_id]
                
                # Check for errors
                if process.returncode != 0:
                    logger.error(f"Docker execution failed: {stderr}")
                    return {
                        'success': False,
                        'error': stderr,
                        'execution_time': time.time() - start_time
                    }
                
                return {
                    'success': True,
                    'stdout': stdout,
                    'stderr': stderr,
                    'execution_time': time.time() - start_time
                }
                
            except subprocess.TimeoutExpired:
                # Stop container
                subprocess.run(["docker", "stop", container_name], check=False)
                
                # Remove from active sandboxes
                del self.active_sandboxes[execution_id]
                
                logger.error(f"Docker execution timed out after {timeout}s")
                return {
                    'success': False,
                    'error': f"Execution timed out after {timeout} seconds",
                    'execution_time': timeout
                }
                
        except Exception as e:
            logger.error(f"Error executing in Docker: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _wrap_python_code(self, code: str, inputs: Optional[Dict[str, Any]] = None) -> str:
        """Wrap Python code with sandbox restrictions."""
        # Prepare inputs JSON
        inputs_json = json.dumps(inputs or {})
        
        # Create wrapper code
        wrapper = f"""
import os
import sys
import json
import time
import traceback
import resource
import builtins
import importlib

# Set resource limits
resource.setrlimit(resource.RLIMIT_CPU, ({self.resource_limits['cpu_time']}, {self.resource_limits['cpu_time']}))
resource.setrlimit(resource.RLIMIT_AS, ({self.resource_limits['memory']}, {self.resource_limits['memory']}))
resource.setrlimit(resource.RLIMIT_FSIZE, ({self.resource_limits['file_size']}, {self.resource_limits['file_size']}))
resource.setrlimit(resource.RLIMIT_NPROC, ({self.resource_limits['processes']}, {self.resource_limits['processes']}))

# Block network access if disabled
if os.environ.get('ONI_SANDBOX_BLOCK_NETWORK') == '1':
    # Monkey patch socket module
    import socket
    _original_socket = socket.socket
    
    def _blocked_socket(*args, **kwargs):
        raise PermissionError("Network access is disabled in this sandbox")
    
    socket.socket = _blocked_socket
    
    # Block urllib and requests
    sys.modules['urllib'] = None
    sys.modules['urllib.request'] = None
    sys.modules['requests'] = None

# Block file I/O if disabled
if not {self.enable_file_io}:
    # Monkey patch built-in open function
    _original_open = builtins.open
    
    def _restricted_open(file, mode='r', *args, **kwargs):
        # Allow reading from specific sandbox directories
        if mode in ('r', 'rb') and (
            file.startswith('{self.sandbox_dir}') or 
            file.startswith('/tmp/') or
            file.startswith(os.path.join(os.getcwd(), 'sandbox'))
        ):
            return _original_open(file, mode, *args, **kwargs)
        else:
            raise PermissionError(f"File access denied: {{file}}")
    
    builtins.open = _restricted_open
    
    # Block os.remove, os.unlink, etc.
    os.remove = lambda path: PermissionError(f"File removal denied: {{path}}")
    os.unlink = lambda path: PermissionError(f"File removal denied: {{path}}")
    os.rmdir = lambda path: PermissionError(f"Directory removal denied: {{path}}")

# Restricted import function
_original_import = builtins.__import__

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Check if module is allowed
    allowed_modules = {', '.join(repr(m) for m in self.allowed_modules)}
    
    if name in allowed_modules or name.split('.')[0] in allowed_modules:
        return _original_import(name, globals, locals, fromlist, level)
    else:
        raise ImportError(f"Import of '{{name}}' is not allowed in the sandbox")

builtins.__import__ = _restricted_import

# Load inputs
inputs = json.loads('''{inputs_json}''')
for key, value in inputs.items():
    globals()[key] = value

# Record output
class OutputCollector:
    def __init__(self):
        self.stdout = []
        self.values = {{"stdout": ""}}
    
    def write(self, text):
        self.stdout.append(text)
        self.values["stdout"] = "".join(self.stdout)
        sys.__stdout__.write(text)
    
    def flush(self):
        sys.__stdout__.flush()

output_collector = OutputCollector()
sys.stdout = output_collector

# Execute user code
try:
    # Start execution timer
    start_time = time.time()
    
    # Execute code
{self._indent_code(code)}
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Add execution time to output
    output_collector.values["execution_time"] = execution_time
    
    # Print output as JSON
    print("__ONI_SANDBOX_OUTPUT_START__")
    print(json.dumps(output_collector.values))
    print("__ONI_SANDBOX_OUTPUT_END__")
    
except Exception as e:
    # Print error as JSON
    print("__ONI_SANDBOX_OUTPUT_START__")
    print(json.dumps({{
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc(),
        "execution_time": time.time() - start_time
    }}))
    print("__ONI_SANDBOX_OUTPUT_END__")
    sys.exit(1)
"""
        
        return wrapper
    
    def _indent_code(self, code: str, indent: int = 4) -> str:
        """Indent code for inclusion in wrapper."""
        lines = code.split('\n')
        indented_lines = [' ' * indent + line for line in lines]
        return '\n'.join(indented_lines)
    
    def _is_command_allowed(self, command: str) -> bool:
        """Check if a shell command is allowed."""
        # List of disallowed commands
        disallowed_commands = [
            'rm -rf', 'rm -r', 'rmdir', 'mkfs',
            'dd', 'shred', 'wget', 'curl',
            'ssh', 'telnet', 'nc', 'netcat',
            'sudo', 'su', 'chown', 'chmod',
            'iptables', 'ifconfig', 'route',
            'mount', 'umount', 'fdisk',
            'mkfs', 'fsck', 'systemctl',
            'service', 'init', 'reboot',
            'shutdown', 'halt', 'poweroff'
        ]
        
        # Check for disallowed commands
        command_lower = command.lower()
        for disallowed in disallowed_commands:
            if disallowed in command_lower:
                return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            '>', '>>', '|', ';', '&&', '||',  # Command chaining
            '$(', '`', '${',  # Command substitution
            'eval', 'exec', 'source',  # Command execution
            '/etc/', '/var/', '/root/', '/home/',  # System directories
            '/dev/', '/proc/', '/sys/'  # System directories
        ]
        
        if not self.enable_file_io:
            # Block file operations if file I/O is disabled
            for pattern in suspicious_patterns:
                if pattern in command:
                    return False
        
        return True
    
    def _is_package_allowed(self, package: str) -> bool:
        """Check if a Python package is allowed."""
        # List of disallowed packages
        disallowed_packages = [
            'subprocess', 'os.system', 'eval', 'exec',
            'socket', 'requests', 'urllib', 'http.client',
            'ftplib', 'paramiko', 'telnetlib',
            'smtplib', 'poplib', 'imaplib',
            'pexpect', 'pty', 'cryptography'
        ]
        
        # Check for disallowed packages
        package_lower = package.lower()
        for disallowed in disallowed_packages:
            if disallowed in package_lower:
                return False
        
        return True
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details
        }
        
        self.audit_log.append(event)
        logger.debug(f"Audit: {event_type} - {json.dumps(details)}")
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Get the audit log.
        
        Returns:
            List of audit events
        """
        return self.audit_log
    
    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self.audit_log = []
        logger.info("Audit log cleared")
    
    def terminate_sandbox(self, sandbox_id: str) -> bool:
        """
        Terminate a running sandbox.
        
        Args:
            sandbox_id: ID of the sandbox to terminate
            
        Returns:
            True if the sandbox was terminated, False otherwise
        """
        if sandbox_id in self.active_sandboxes:
            process = self.active_sandboxes[sandbox_id]
            try:
                process.terminate()
                process.wait(timeout=5)
                del self.active_sandboxes[sandbox_id]
                logger.info(f"Terminated sandbox {sandbox_id}")
                return True
            except:
                try:
                    process.kill()
                    process.wait(timeout=5)
                    del self.active_sandboxes[sandbox_id]
                    logger.info(f"Killed sandbox {sandbox_id}")
                    return True
                except:
                    logger.error(f"Failed to terminate sandbox {sandbox_id}")
                    return False
        else:
            logger.warning(f"Sandbox {sandbox_id} not found")
            return False
    
    def get_active_sandboxes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about active sandboxes.
        
        Returns:
            Dictionary mapping sandbox IDs to sandbox information
        """
        result = {}
        
        for sandbox_id, process in self.active_sandboxes.items():
            try:
                result[sandbox_id] = {
                    'pid': process.pid,
                    'running': process.poll() is None,
                    'returncode': process.returncode if process.poll() is not None else None
                }
            except:
                result[sandbox_id] = {
                    'pid': None,
                    'running': False,
                    'returncode': None,
                    'error': "Failed to get process information"
                }
        
        return result
    
    def create_secure_environment(self, name: str) -> Dict[str, Any]:
        """
        Create a secure execution environment.
        
        Args:
            name: Name for the environment
            
        Returns:
            Dictionary with environment information
        """
        # Create a unique ID for this environment
        env_id = str(uuid.uuid4())[:8]
        safe_env_name = f"oni_env_{name}_{env_id}"
        
        # Create environment directory
        env_dir = os.path.join(self.sandbox_dir, safe_env_name)
        os.makedirs(env_dir, exist_ok=True)
        
        # Create Python virtual environment
        venv_result = self.create_virtual_environment(safe_env_name)
        
        if not venv_result['success']:
            return venv_result
        
        # Create environment information
        env_info = {
            'name': safe_env_name,
            'id': env_id,
            'dir': env_dir,
            'python_path': venv_result['python_path'],
            'created_at': time.time()
        }
        
        # Log environment creation
        self._log_audit_event(
            event_type="create_environment",
            details=env_info
        )
        
        return {
            'success': True,
            'environment': env_info
        }
    
    def execute_in_environment(self, env_name: str, code: str, 
                             inputs: Optional[Dict[str, Any]] = None,
                             timeout: int = 30) -> Dict[str, Any]:
        """
        Execute code in a secure environment.
        
        Args:
            env_name: Name of the environment
            code: Python code to execute
            inputs: Input variables (optional)
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        # Find environment directory
        env_dir = None
        for dirname in os.listdir(self.sandbox_dir):
            if dirname == env_name or dirname.startswith(f"oni_env_{env_name}_"):
                env_dir = os.path.join(self.sandbox_dir, dirname)
                break
        
        if not env_dir:
            return {
                'success': False,
                'error': f"Environment '{env_name}' not found"
            }
        
        # Get Python path
        if os.name == 'nt':  # Windows
            python_path = os.path.join(env_dir, "Scripts", "python")
        else:  # Unix-like
            python_path = os.path.join(env_dir, "bin", "python")
        
        # Create a temporary file for the code
        execution_id = str(uuid.uuid4())
        code_file = os.path.join(env_dir, f"{execution_id}.py")
        
        try:
            # Prepare code with sandbox wrapper
            wrapped_code = self._wrap_python_code(code, inputs)
            
            # Write code to file
            with open(code_file, 'w') as f:
                f.write(wrapped_code)
            
            # Log execution
            self._log_audit_event(
                event_type="execute_in_environment",
                details={
                    "execution_id": execution_id,
                    "environment": env_name,
                    "code_length": len(code),
                    "timeout": timeout
                }
            )
            
            # Execute code
            start_time = time.time()
            
            # Set up environment
            env = os.environ.copy()
            
            if not self.enable_network:
                # Block network access (Unix-like systems)
                if os.name == 'posix':
                    env["ONI_SANDBOX_BLOCK_NETWORK"] = "1"
            
            # Execute process
            process = subprocess.Popen(
                [python_path, code_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # Store active sandbox
            self.active_sandboxes[execution_id] = process
            
            try:
                # Wait for process to complete with timeout
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Remove from active sandboxes
                del self.active_sandboxes[execution_id]
                
                # Check for errors
                if process.returncode != 0:
                    logger.error(f"Code execution in environment failed: {stderr}")
                    return {
                        'success': False,
                        'error': stderr,
                        'execution_time': time.time() - start_time
                    }
                
                # Parse output
                try:
                    # Look for JSON output marker
                    output_start = stdout.find("__ONI_SANDBOX_OUTPUT_START__")
                    output_end = stdout.find("__ONI_SANDBOX_OUTPUT_END__")
                    
                    if output_start >= 0 and output_end >= 0:
                        output_json = stdout[output_start + len("__ONI_SANDBOX_OUTPUT_START__"):output_end].strip()
                        output = json.loads(output_json)
                    else:
                        output = {'stdout': stdout}
                except json.JSONDecodeError:
                    output = {'stdout': stdout}
                
                return {
                    'success': True,
                    'output': output,
                    'execution_time': time.time() - start_time
                }
                
            except subprocess.TimeoutExpired:
                # Terminate process
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                # Remove from active sandboxes
                del self.active_sandboxes[execution_id]
                
                logger.error(f"Code execution in environment timed out after {timeout}s")
                return {
                    'success': False,
                    'error': f"Execution timed out after {timeout} seconds",
                    'execution_time': timeout
                }
                
        except Exception as e:
            logger.error(f"Error executing in environment: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        finally:
            # Clean up
            if os.path.exists(code_file):
                os.remove(code_file)
