import unittest
import os
import tempfile
import shutil
from tools.docker_manager import DockerManager

class TestDockerManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.docker_manager = DockerManager(work_dir=self.test_dir)
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test that DockerManager initializes correctly."""
        self.assertEqual(self.docker_manager.work_dir, self.test_dir)
        
    def test_is_docker_available(self):
        """Test checking if Docker is available."""
        # This test will pass or fail depending on whether Docker is installed
        # and running on the test machine
        result = self.docker_manager.is_docker_available()
        self.assertIsInstance(result, bool)
        
    def test_create_dockerfile(self):
        """Test creating a Dockerfile."""
        result = self.docker_manager.create_dockerfile(
            base_image="python:3.9",
            commands=["pip install numpy", "pip install pandas"],
            expose_ports=[8000, 8080],
            environment={"DEBUG": "true", "LOG_LEVEL": "info"},
            volumes=["/data", "/app"],
            entrypoint='["python"]',
            cmd='["app.py"]'
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["base_image"], "python:3.9")
        
        # Check Dockerfile content
        dockerfile = result["dockerfile"]
        self.assertIn("FROM python:3.9", dockerfile)
        self.assertIn("RUN pip install numpy", dockerfile)
        self.assertIn("RUN pip install pandas", dockerfile)
        self.assertIn("EXPOSE 8000", dockerfile)
        self.assertIn("EXPOSE 8080", dockerfile)
        self.assertIn("ENV DEBUG=true", dockerfile)
        self.assertIn("ENV LOG_LEVEL=info", dockerfile)
        self.assertIn("VOLUME /data", dockerfile)
        self.assertIn("VOLUME /app", dockerfile)
        self.assertIn("ENTRYPOINT [\"python\"]", dockerfile)
        self.assertIn("CMD [\"app.py\"]", dockerfile)
        
    def test_create_docker_compose(self):
        """Test creating a Docker Compose file."""
        services = {
            "web": {
                "image": "nginx:latest",
                "ports": ["80:80"],
                "volumes": ["./html:/usr/share/nginx/html"]
            },
            "db": {
                "image": "postgres:13",
                "environment": {
                    "POSTGRES_PASSWORD": "example"
                },
                "volumes": ["postgres_data:/var/lib/postgresql/data"]
            }
        }
        
        networks = {
            "frontend": {"driver": "bridge"},
            "backend": {"driver": "bridge"}
        }
        
        volumes = {
            "postgres_data": {"driver": "local"}
        }
        
        result = self.docker_manager.create_docker_compose(
            services=services,
            networks=networks,
            volumes=volumes
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["services"], ["web", "db"])
        
        # Check Docker Compose content
        compose = result["compose"]
        self.assertIn("version: '3'", compose)
        self.assertIn("services:", compose)
        self.assertIn("web:", compose)
        self.assertIn("db:", compose)
        self.assertIn("image: nginx:latest", compose)
        self.assertIn("image: postgres:13", compose)
        self.assertIn("networks:", compose)
        self.assertIn("volumes:", compose)
        
    def test_create_development_environment(self):
        """Test creating a development environment."""
        # Skip if Docker is not available
        if not self.docker_manager.is_docker_available():
            self.skipTest("Docker is not available")
            
        result = self.docker_manager.create_development_environment(
            language="python",
            project_name="test-project",
            dependencies=["numpy", "pandas"]
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["language"], "python")
        self.assertEqual(result["project_name"], "test-project")
        
        # Check that project directory was created
        project_dir = os.path.join(self.test_dir, "test-project")
        self.assertTrue(os.path.exists(project_dir))
        
        # Check that Dockerfile was created
        dockerfile_path = os.path.join(project_dir, "Dockerfile")
        self.assertTrue(os.path.exists(dockerfile_path))
        
        # Check Dockerfile content
        with open(dockerfile_path, "r") as f:
            dockerfile = f.read()
            self.assertIn("FROM python:3.10", dockerfile)
            self.assertIn("pip install --no-cache-dir numpy pandas", dockerfile)
            
    def test_run_experiment_container(self):
        """Test running an experiment container."""
        # Skip if Docker is not available
        if not self.docker_manager.is_docker_available():
            self.skipTest("Docker is not available")
            
        script_content = """
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.savefig('/experiment/sine_wave.png')
print('Experiment completed successfully!')
"""
        
        result = self.docker_manager.run_experiment_container(
            experiment_name="sine-wave",
            script_content=script_content,
            framework="sklearn",
            dependencies=["matplotlib"]
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["experiment_name"], "sine-wave")
        self.assertEqual(result["framework"], "sklearn")
        
        # Check that experiment directory was created
        experiment_dir = os.path.join(self.test_dir, "sine-wave")
        self.assertTrue(os.path.exists(experiment_dir))
        
        # Check that script was created
        script_path = os.path.join(experiment_dir, "experiment.py")
        self.assertTrue(os.path.exists(script_path))
        
        # Check script content
        with open(script_path, "r") as f:
            script = f.read()
            self.assertIn("import numpy as np", script)
            self.assertIn("import matplotlib.pyplot as plt", script)
            
    def test_run_database(self):
        """Test running a database container."""
        # Skip if Docker is not available
        if not self.docker_manager.is_docker_available():
            self.skipTest("Docker is not available")
            
        result = self.docker_manager.run_database(
            db_type="postgres",
            name="test-db",
            port=5432,
            password="test-password"
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["database_type"], "postgres")
        self.assertEqual(result["database_name"], "test-db")
        self.assertEqual(result["port"], 5432)
        
        # Clean up container
        if "container_id" in result:
            self.docker_manager.stop_container(result["container_id"])
            self.docker_manager.remove_container(result["container_id"], force=True)

if __name__ == '__main__':
    unittest.main()