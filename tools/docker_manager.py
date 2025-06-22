import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import tempfile
import shutil
import subprocess
import threading
import queue
import traceback

# Import Docker library
try:
    import docker
    from docker.errors import DockerException, ImageNotFound, APIError
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DockerManager:
    """Docker integration for ONI to manage containers and environments."""
    
    def __init__(self, work_dir: Optional[str] = None):
        """
        Initialize Docker manager.
        
        Args:
            work_dir: Working directory for Docker operations (default: temporary directory)
        """
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="oni_docker_")
        
        # Check if Docker is available
        self.docker_available = HAS_DOCKER
        self.client = None
        
        if self.docker_available:
            try:
                self.client = docker.from_env()
                self.docker_available = True
                logger.info("Docker client initialized successfully")
            except DockerException as e:
                logger.error(f"Failed to initialize Docker client: {e}")
                self.docker_available = False
        else:
            logger.warning("Docker library not available. Install docker-py to use Docker integration.")
        
        # Create work directory if it doesn't exist
        os.makedirs(self.work_dir, exist_ok=True)
        
        logger.info(f"Docker manager initialized with work directory: {self.work_dir}")
    
    def __del__(self):
        """Clean up resources."""
        if self.work_dir.startswith(tempfile.gettempdir()):
            try:
                shutil.rmtree(self.work_dir)
                logger.info(f"Removed temporary directory: {self.work_dir}")
            except Exception as e:
                logger.error(f"Failed to remove temporary directory: {e}")
    
    def is_docker_available(self) -> bool:
        """Check if Docker is available."""
        if not self.docker_available:
            return False
        
        try:
            # Try to ping Docker daemon
            self.client.ping()
            return True
        except:
            return False
    
    def list_images(self) -> Dict[str, Any]:
        """
        List Docker images.
        
        Returns:
            Dictionary with list of images
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            images = self.client.images.list()
            
            # Format image information
            formatted_images = []
            for image in images:
                tags = image.tags
                image_id = image.id
                created = image.attrs.get('Created')
                size = image.attrs.get('Size')
                
                formatted_images.append({
                    "id": image_id,
                    "tags": tags,
                    "created": created,
                    "size": size
                })
            
            logger.info(f"Listed {len(formatted_images)} Docker images")
            return {
                "success": True,
                "images": formatted_images
            }
            
        except Exception as e:
            logger.error(f"Failed to list Docker images: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_containers(self, all_containers: bool = False) -> Dict[str, Any]:
        """
        List Docker containers.
        
        Args:
            all_containers: Whether to list all containers (including stopped ones)
            
        Returns:
            Dictionary with list of containers
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            containers = self.client.containers.list(all=all_containers)
            
            # Format container information
            formatted_containers = []
            for container in containers:
                container_id = container.id
                name = container.name
                image = container.image.tags[0] if container.image.tags else container.image.id
                status = container.status
                ports = container.ports
                
                formatted_containers.append({
                    "id": container_id,
                    "name": name,
                    "image": image,
                    "status": status,
                    "ports": ports
                })
            
            logger.info(f"Listed {len(formatted_containers)} Docker containers")
            return {
                "success": True,
                "containers": formatted_containers
            }
            
        except Exception as e:
            logger.error(f"Failed to list Docker containers: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def pull_image(self, image_name: str) -> Dict[str, Any]:
        """
        Pull a Docker image.
        
        Args:
            image_name: Name of the image to pull
            
        Returns:
            Dictionary with pull result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            logger.info(f"Pulling Docker image: {image_name}")
            image = self.client.images.pull(image_name)
            
            logger.info(f"Pulled Docker image: {image_name}")
            return {
                "success": True,
                "image": image_name,
                "id": image.id
            }
            
        except Exception as e:
            logger.error(f"Failed to pull Docker image: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def build_image(self, dockerfile_content: str, tag: str, 
                   build_args: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Build a Docker image from a Dockerfile.
        
        Args:
            dockerfile_content: Content of the Dockerfile
            tag: Tag for the built image
            build_args: Build arguments (optional)
            
        Returns:
            Dictionary with build result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Create a temporary directory for the build
            build_dir = os.path.join(self.work_dir, f"build_{int(time.time())}")
            os.makedirs(build_dir, exist_ok=True)
            
            # Write Dockerfile
            dockerfile_path = os.path.join(build_dir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
            
            logger.info(f"Building Docker image: {tag}")
            
            # Build image
            image, logs = self.client.images.build(
                path=build_dir,
                tag=tag,
                buildargs=build_args,
                rm=True
            )
            
            # Collect build logs
            build_logs = []
            for log in logs:
                if 'stream' in log:
                    log_line = log['stream'].strip()
                    if log_line:
                        build_logs.append(log_line)
            
            logger.info(f"Built Docker image: {tag}")
            return {
                "success": True,
                "image": tag,
                "id": image.id,
                "logs": build_logs
            }
            
        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # Clean up build directory
            if 'build_dir' in locals():
                shutil.rmtree(build_dir)
    
    def run_container(self, image: str, command: Optional[str] = None, 
                     name: Optional[str] = None, ports: Optional[Dict[str, str]] = None,
                     volumes: Optional[Dict[str, Dict[str, str]]] = None,
                     environment: Optional[Dict[str, str]] = None,
                     detach: bool = True) -> Dict[str, Any]:
        """
        Run a Docker container.
        
        Args:
            image: Image to run
            command: Command to run in the container (optional)
            name: Name for the container (optional)
            ports: Port mappings (optional)
            volumes: Volume mappings (optional)
            environment: Environment variables (optional)
            detach: Whether to run the container in detached mode
            
        Returns:
            Dictionary with run result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            logger.info(f"Running Docker container from image: {image}")
            
            # Run container
            container = self.client.containers.run(
                image=image,
                command=command,
                name=name,
                ports=ports,
                volumes=volumes,
                environment=environment,
                detach=detach
            )
            
            logger.info(f"Started Docker container: {container.name}")
            return {
                "success": True,
                "container_id": container.id,
                "container_name": container.name,
                "image": image
            }
            
        except Exception as e:
            logger.error(f"Failed to run Docker container: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def stop_container(self, container_id: str) -> Dict[str, Any]:
        """
        Stop a Docker container.
        
        Args:
            container_id: ID or name of the container to stop
            
        Returns:
            Dictionary with stop result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Get container
            container = self.client.containers.get(container_id)
            
            # Stop container
            container.stop()
            
            logger.info(f"Stopped Docker container: {container_id}")
            return {
                "success": True,
                "container_id": container.id,
                "container_name": container.name
            }
            
        except Exception as e:
            logger.error(f"Failed to stop Docker container: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def remove_container(self, container_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Remove a Docker container.
        
        Args:
            container_id: ID or name of the container to remove
            force: Whether to force removal
            
        Returns:
            Dictionary with removal result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Get container
            container = self.client.containers.get(container_id)
            
            # Remove container
            container.remove(force=force)
            
            logger.info(f"Removed Docker container: {container_id}")
            return {
                "success": True,
                "container_id": container_id
            }
            
        except Exception as e:
            logger.error(f"Failed to remove Docker container: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_container_logs(self, container_id: str, tail: int = 100) -> Dict[str, Any]:
        """
        Get logs from a Docker container.
        
        Args:
            container_id: ID or name of the container
            tail: Number of lines to get from the end of the logs
            
        Returns:
            Dictionary with container logs
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Get container
            container = self.client.containers.get(container_id)
            
            # Get logs
            logs = container.logs(tail=tail).decode('utf-8')
            
            logger.info(f"Got logs from Docker container: {container_id}")
            return {
                "success": True,
                "container_id": container.id,
                "container_name": container.name,
                "logs": logs
            }
            
        except Exception as e:
            logger.error(f"Failed to get Docker container logs: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def exec_in_container(self, container_id: str, command: str) -> Dict[str, Any]:
        """
        Execute a command in a running container.
        
        Args:
            container_id: ID or name of the container
            command: Command to execute
            
        Returns:
            Dictionary with execution result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Get container
            container = self.client.containers.get(container_id)
            
            # Execute command
            exec_result = container.exec_run(command)
            
            # Get output
            exit_code = exec_result.exit_code
            output = exec_result.output.decode('utf-8')
            
            logger.info(f"Executed command in Docker container: {container_id}")
            return {
                "success": exit_code == 0,
                "container_id": container.id,
                "container_name": container.name,
                "exit_code": exit_code,
                "output": output
            }
            
        except Exception as e:
            logger.error(f"Failed to execute command in Docker container: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_network(self, name: str, driver: str = "bridge") -> Dict[str, Any]:
        """
        Create a Docker network.
        
        Args:
            name: Name for the network
            driver: Network driver
            
        Returns:
            Dictionary with network creation result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Create network
            network = self.client.networks.create(name=name, driver=driver)
            
            logger.info(f"Created Docker network: {name}")
            return {
                "success": True,
                "network_id": network.id,
                "network_name": network.name,
                "driver": driver
            }
            
        except Exception as e:
            logger.error(f"Failed to create Docker network: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_volume(self, name: str) -> Dict[str, Any]:
        """
        Create a Docker volume.
        
        Args:
            name: Name for the volume
            
        Returns:
            Dictionary with volume creation result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Create volume
            volume = self.client.volumes.create(name=name)
            
            logger.info(f"Created Docker volume: {name}")
            return {
                "success": True,
                "volume_name": volume.name
            }
            
        except Exception as e:
            logger.error(f"Failed to create Docker volume: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_docker_compose(self, compose_content: str, project_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run Docker Compose.
        
        Args:
            compose_content: Content of the docker-compose.yml file
            project_name: Name for the Docker Compose project (optional)
            
        Returns:
            Dictionary with Docker Compose result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Create a temporary directory for Docker Compose
            compose_dir = os.path.join(self.work_dir, f"compose_{int(time.time())}")
            os.makedirs(compose_dir, exist_ok=True)
            
            # Write docker-compose.yml
            compose_path = os.path.join(compose_dir, "docker-compose.yml")
            with open(compose_path, "w") as f:
                f.write(compose_content)
            
            # Build command
            cmd = ["docker-compose"]
            
            if project_name:
                cmd.extend(["-p", project_name])
            
            cmd.extend(["-f", compose_path, "up", "-d"])
            
            # Run Docker Compose
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker Compose failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr
                }
            
            logger.info(f"Started Docker Compose project: {project_name or 'default'}")
            return {
                "success": True,
                "project_name": project_name or "default",
                "compose_dir": compose_dir,
                "output": result.stdout
            }
            
        except Exception as e:
            logger.error(f"Failed to run Docker Compose: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_dockerfile(self, base_image: str, commands: List[str], 
                        expose_ports: Optional[List[int]] = None,
                        environment: Optional[Dict[str, str]] = None,
                        volumes: Optional[List[str]] = None,
                        entrypoint: Optional[str] = None,
                        cmd: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a Dockerfile with specified parameters.
        
        Args:
            base_image: Base image for the Dockerfile
            commands: List of RUN commands
            expose_ports: List of ports to expose (optional)
            environment: Environment variables (optional)
            volumes: Volumes to create (optional)
            entrypoint: Container entrypoint (optional)
            cmd: Container command (optional)
            
        Returns:
            Dictionary with Dockerfile content
        """
        try:
            # Start with base image
            dockerfile = f"FROM {base_image}\n\n"
            
            # Add environment variables
            if environment:
                for key, value in environment.items():
                    dockerfile += f"ENV {key}={value}\n"
                dockerfile += "\n"
            
            # Add commands
            for command in commands:
                dockerfile += f"RUN {command}\n"
            dockerfile += "\n"
            
            # Add volumes
            if volumes:
                for volume in volumes:
                    dockerfile += f"VOLUME {volume}\n"
                dockerfile += "\n"
            
            # Add exposed ports
            if expose_ports:
                for port in expose_ports:
                    dockerfile += f"EXPOSE {port}\n"
                dockerfile += "\n"
            
            # Add entrypoint
            if entrypoint:
                dockerfile += f"ENTRYPOINT {entrypoint}\n"
            
            # Add command
            if cmd:
                dockerfile += f"CMD {cmd}\n"
            
            logger.info(f"Created Dockerfile based on {base_image}")
            return {
                "success": True,
                "dockerfile": dockerfile,
                "base_image": base_image
            }
            
        except Exception as e:
            logger.error(f"Failed to create Dockerfile: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_docker_compose(self, services: Dict[str, Dict[str, Any]], 
                            networks: Optional[Dict[str, Dict[str, Any]]] = None,
                            volumes: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Create a Docker Compose file.
        
        Args:
            services: Dictionary of services
            networks: Dictionary of networks (optional)
            volumes: Dictionary of volumes (optional)
            
        Returns:
            Dictionary with Docker Compose content
        """
        try:
            # Create Docker Compose structure
            compose = {
                "version": "3",
                "services": services
            }
            
            # Add networks if provided
            if networks:
                compose["networks"] = networks
            
            # Add volumes if provided
            if volumes:
                compose["volumes"] = volumes
            
            # Convert to YAML
            import yaml
            compose_yaml = yaml.dump(compose, default_flow_style=False)
            
            logger.info(f"Created Docker Compose file with {len(services)} services")
            return {
                "success": True,
                "compose": compose_yaml,
                "services": list(services.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to create Docker Compose file: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_container_with_model(self, model_path: str, framework: str = "pytorch", 
                               gpu: bool = False, ports: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run a container with a machine learning model.
        
        Args:
            model_path: Path to the model
            framework: ML framework ('pytorch', 'tensorflow', 'onnx')
            gpu: Whether to use GPU
            ports: Port mappings (optional)
            
        Returns:
            Dictionary with container run result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Determine base image based on framework and GPU
            if framework == "pytorch":
                base_image = "pytorch/pytorch:latest-gpu" if gpu else "pytorch/pytorch:latest"
            elif framework == "tensorflow":
                base_image = "tensorflow/tensorflow:latest-gpu" if gpu else "tensorflow/tensorflow:latest"
            elif framework == "onnx":
                base_image = "mcr.microsoft.com/azureml/onnxruntime:latest-cuda" if gpu else "mcr.microsoft.com/azureml/onnxruntime:latest"
            else:
                return {"success": False, "error": f"Unsupported framework: {framework}"}
            
            # Create a temporary directory for the model
            model_dir = os.path.join(self.work_dir, f"model_{int(time.time())}")
            os.makedirs(model_dir, exist_ok=True)
            
            # Copy model to the directory
            if os.path.isfile(model_path):
                shutil.copy(model_path, os.path.join(model_dir, os.path.basename(model_path)))
            elif os.path.isdir(model_path):
                shutil.copytree(model_path, os.path.join(model_dir, os.path.basename(model_path)))
            
            # Set up volumes
            volumes = {
                model_dir: {'bind': '/model', 'mode': 'ro'}
            }
            
            # Set up environment variables
            environment = {
                "MODEL_PATH": "/model",
                "FRAMEWORK": framework,
                "USE_GPU": "1" if gpu else "0"
            }
            
            # Run container
            result = self.run_container(
                image=base_image,
                ports=ports,
                volumes=volumes,
                environment=environment,
                detach=True
            )
            
            if not result.get("success", False):
                return result
            
            logger.info(f"Started container with {framework} model")
            return {
                "success": True,
                "container_id": result.get("container_id"),
                "container_name": result.get("container_name"),
                "framework": framework,
                "gpu": gpu,
                "model_dir": model_dir
            }
            
        except Exception as e:
            logger.error(f"Failed to run container with model: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_development_environment(self, language: str, project_name: str, 
                                     dependencies: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a development environment container.
        
        Args:
            language: Programming language ('python', 'node', 'go', etc.)
            project_name: Name for the project
            dependencies: List of dependencies to install (optional)
            
        Returns:
            Dictionary with environment creation result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Determine base image and setup commands based on language
            if language == "python":
                base_image = "python:3.10"
                setup_commands = ["pip install --no-cache-dir -U pip"]
                if dependencies:
                    setup_commands.append(f"pip install --no-cache-dir {' '.join(dependencies)}")
                
                # Create requirements.txt if dependencies provided
                if dependencies:
                    requirements_path = os.path.join(self.work_dir, "requirements.txt")
                    with open(requirements_path, "w") as f:
                        f.write("\n".join(dependencies))
            
            elif language == "node":
                base_image = "node:16"
                setup_commands = ["npm install -g npm@latest"]
                if dependencies:
                    setup_commands.append(f"npm install -g {' '.join(dependencies)}")
                
                # Create package.json if dependencies provided
                if dependencies:
                    package_json = {
                        "name": project_name,
                        "version": "1.0.0",
                        "description": f"{project_name} project",
                        "dependencies": {dep: "latest" for dep in dependencies}
                    }
                    package_path = os.path.join(self.work_dir, "package.json")
                    with open(package_path, "w") as f:
                        json.dump(package_json, f, indent=2)
            
            elif language == "go":
                base_image = "golang:1.18"
                setup_commands = []
                if dependencies:
                    for dep in dependencies:
                        setup_commands.append(f"go get {dep}")
            
            else:
                return {"success": False, "error": f"Unsupported language: {language}"}
            
            # Create Dockerfile
            dockerfile_result = self.create_dockerfile(
                base_image=base_image,
                commands=setup_commands,
                volumes=["/app"],
                entrypoint='["sh", "-c"]',
                cmd='["cd /app && bash"]'
            )
            
            if not dockerfile_result.get("success", False):
                return dockerfile_result
            
            # Create project directory
            project_dir = os.path.join(self.work_dir, project_name)
            os.makedirs(project_dir, exist_ok=True)
            
            # Write Dockerfile
            dockerfile_path = os.path.join(project_dir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_result.get("dockerfile", ""))
            
            # Build image
            image_tag = f"{project_name.lower()}-dev"
            
            # Build command
            cmd = ["docker", "build", "-t", image_tag, project_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker build failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr
                }
            
            # Run container
            container_name = f"{project_name.lower()}-dev"
            
            # Run command
            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "-v", f"{os.path.abspath(project_dir)}:/app",
                "-w", "/app",
                "-it",
                image_tag,
                "tail", "-f", "/dev/null"  # Keep container running
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker run failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr
                }
            
            logger.info(f"Created development environment for {language}")
            return {
                "success": True,
                "language": language,
                "project_name": project_name,
                "project_dir": project_dir,
                "image_tag": image_tag,
                "container_name": container_name
            }
            
        except Exception as e:
            logger.error(f"Failed to create development environment: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_jupyter_notebook(self, notebook_dir: str, port: int = 8888) -> Dict[str, Any]:
        """
        Run a Jupyter Notebook server in a container.
        
        Args:
            notebook_dir: Directory to mount for notebooks
            port: Port to expose Jupyter on
            
        Returns:
            Dictionary with Jupyter container result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Pull Jupyter image
            pull_result = self.pull_image("jupyter/datascience-notebook")
            
            if not pull_result.get("success", False):
                return pull_result
            
            # Create absolute path for notebook directory
            notebook_dir_abs = os.path.abspath(notebook_dir)
            
            # Create directory if it doesn't exist
            os.makedirs(notebook_dir_abs, exist_ok=True)
            
            # Set up volumes
            volumes = {
                notebook_dir_abs: {'bind': '/home/jovyan/work', 'mode': 'rw'}
            }
            
            # Set up ports
            ports = {
                '8888/tcp': str(port)
            }
            
            # Run container
            result = self.run_container(
                image="jupyter/datascience-notebook",
                command="start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''",
                ports=ports,
                volumes=volumes,
                detach=True
            )
            
            if not result.get("success", False):
                return result
            
            logger.info(f"Started Jupyter Notebook server on port {port}")
            return {
                "success": True,
                "container_id": result.get("container_id"),
                "container_name": result.get("container_name"),
                "url": f"http://localhost:{port}",
                "notebook_dir": notebook_dir_abs
            }
            
        except Exception as e:
            logger.error(f"Failed to run Jupyter Notebook: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_database(self, db_type: str, name: str, port: int, 
                    password: str, volume_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a database container.
        
        Args:
            db_type: Type of database ('postgres', 'mysql', 'mongodb', 'redis')
            name: Name for the database
            port: Port to expose
            password: Root/admin password
            volume_name: Name for the data volume (optional)
            
        Returns:
            Dictionary with database container result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Determine image and environment based on database type
            if db_type == "postgres":
                image = "postgres:13"
                environment = {
                    "POSTGRES_PASSWORD": password,
                    "POSTGRES_DB": name
                }
                ports = {
                    '5432/tcp': str(port)
                }
                volume_path = "/var/lib/postgresql/data"
            
            elif db_type == "mysql":
                image = "mysql:8"
                environment = {
                    "MYSQL_ROOT_PASSWORD": password,
                    "MYSQL_DATABASE": name
                }
                ports = {
                    '3306/tcp': str(port)
                }
                volume_path = "/var/lib/mysql"
            
            elif db_type == "mongodb":
                image = "mongo:5"
                environment = {
                    "MONGO_INITDB_ROOT_USERNAME": "root",
                    "MONGO_INITDB_ROOT_PASSWORD": password,
                    "MONGO_INITDB_DATABASE": name
                }
                ports = {
                    '27017/tcp': str(port)
                }
                volume_path = "/data/db"
            
            elif db_type == "redis":
                image = "redis:6"
                environment = {}
                if password:
                    environment["REDIS_PASSWORD"] = password
                ports = {
                    '6379/tcp': str(port)
                }
                volume_path = "/data"
            
            else:
                return {"success": False, "error": f"Unsupported database type: {db_type}"}
            
            # Pull image
            pull_result = self.pull_image(image)
            
            if not pull_result.get("success", False):
                return pull_result
            
            # Create volume if name provided
            volumes = {}
            if volume_name:
                volume_result = self.create_volume(volume_name)
                
                if not volume_result.get("success", False):
                    return volume_result
                
                volumes = {
                    volume_name: {'bind': volume_path, 'mode': 'rw'}
                }
            
            # Run container
            container_name = f"{db_type}-{name}"
            result = self.run_container(
                image=image,
                name=container_name,
                ports=ports,
                volumes=volumes,
                environment=environment,
                detach=True
            )
            
            if not result.get("success", False):
                return result
            
            logger.info(f"Started {db_type} database on port {port}")
            return {
                "success": True,
                "container_id": result.get("container_id"),
                "container_name": result.get("container_name"),
                "database_type": db_type,
                "database_name": name,
                "port": port,
                "volume": volume_name
            }
            
        except Exception as e:
            logger.error(f"Failed to run database: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_experiment_container(self, experiment_name: str, script_content: str, 
                               framework: str = "pytorch", gpu: bool = False,
                               dependencies: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run an experiment in a container.
        
        Args:
            experiment_name: Name for the experiment
            script_content: Content of the experiment script
            framework: ML framework ('pytorch', 'tensorflow', 'sklearn')
            gpu: Whether to use GPU
            dependencies: Additional dependencies to install
            
        Returns:
            Dictionary with experiment container result
        """
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # Determine base image based on framework and GPU
            if framework == "pytorch":
                base_image = "pytorch/pytorch:latest-gpu" if gpu else "pytorch/pytorch:latest"
            elif framework == "tensorflow":
                base_image = "tensorflow/tensorflow:latest-gpu" if gpu else "tensorflow/tensorflow:latest"
            elif framework == "sklearn":
                base_image = "python:3.9"  # scikit-learn doesn't have an official image
            else:
                return {"success": False, "error": f"Unsupported framework: {framework}"}
            
            # Create a temporary directory for the experiment
            experiment_dir = os.path.join(self.work_dir, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Write script
            script_path = os.path.join(experiment_dir, "experiment.py")
            with open(script_path, "w") as f:
                f.write(script_content)
            
            # Create requirements.txt if dependencies provided
            if dependencies:
                requirements_path = os.path.join(experiment_dir, "requirements.txt")
                with open(requirements_path, "w") as f:
                    f.write("\n".join(dependencies))
            
            # Create Dockerfile
            commands = []
            
            # Install dependencies
            if framework == "sklearn":
                commands.append("pip install --no-cache-dir scikit-learn numpy pandas matplotlib")
            
            if dependencies:
                commands.append("pip install --no-cache-dir -r requirements.txt")
            
            dockerfile_result = self.create_dockerfile(
                base_image=base_image,
                commands=commands,
                volumes=["/experiment"],
                cmd='["python", "/experiment/experiment.py"]'
            )
            
            if not dockerfile_result.get("success", False):
                return dockerfile_result
            
            # Write Dockerfile
            dockerfile_path = os.path.join(experiment_dir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_result.get("dockerfile", ""))
            
            # Build image
            image_tag = f"{experiment_name.lower()}-experiment"
            
            # Build command
            cmd = ["docker", "build", "-t", image_tag, experiment_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker build failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr
                }
            
            # Run container
            container_name = f"{experiment_name.lower()}-experiment"
            
            # Set up volumes
            volumes = {
                experiment_dir: {'bind': '/experiment', 'mode': 'rw'}
            }
            
            # Run command
            gpu_args = ["--gpus", "all"] if gpu else []
            cmd = [
                "docker", "run", "-d",
                "--name", container_name
            ] + gpu_args + [
                "-v", f"{os.path.abspath(experiment_dir)}:/experiment",
                image_tag
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker run failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr
                }
            
            logger.info(f"Started experiment container: {container_name}")
            return {
                "success": True,
                "experiment_name": experiment_name,
                "framework": framework,
                "gpu": gpu,
                "experiment_dir": experiment_dir,
                "image_tag": image_tag,
                "container_name": container_name
            }
            
        except Exception as e:
            logger.error(f"Failed to run experiment container: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    docker_manager = DockerManager()
    
    if docker_manager.is_docker_available():
        # List images
        images = docker_manager.list_images()
        print(f"Docker images: {images}")
        
        # Run a simple container
        container = docker_manager.run_container("hello-world")
        print(f"Container: {container}")
    else:
        print("Docker is not available")