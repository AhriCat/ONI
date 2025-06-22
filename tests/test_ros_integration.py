import unittest
import os
import tempfile
import shutil
from tools.ros_integration import ROSIntegration

class TestROSIntegration(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.ros_integration = ROSIntegration(node_name="oni_test_node")
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
        
        # Stop ROS node if running
        if hasattr(self, 'ros_integration') and self.ros_integration.is_running:
            self.ros_integration.stop()
        
    def test_initialization(self):
        """Test that ROSIntegration initializes correctly."""
        self.assertEqual(self.ros_integration.node_name, "oni_test_node")
        self.assertFalse(self.ros_integration.is_initialized)
        self.assertFalse(self.ros_integration.is_running)
        
    def test_create_ros_package(self):
        """Test creating a ROS package."""
        result = self.ros_integration.create_ros_package(
            package_name="test_package",
            dependencies=["std_msgs", "sensor_msgs"]
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["package_name"], "test_package")
        
        # Check that package directory was created
        package_dir = result["package_dir"]
        self.assertTrue(os.path.exists(package_dir))
        
        # Check that package files were created
        self.assertTrue(os.path.exists(os.path.join(package_dir, "package.xml")))
        self.assertTrue(os.path.exists(os.path.join(package_dir, "CMakeLists.txt")))
        self.assertTrue(os.path.exists(os.path.join(package_dir, "src")))
        self.assertTrue(os.path.exists(os.path.join(package_dir, "include", "test_package")))
        self.assertTrue(os.path.exists(os.path.join(package_dir, "launch")))
        
        # Check package.xml content
        with open(os.path.join(package_dir, "package.xml"), "r") as f:
            package_xml = f.read()
            self.assertIn("<name>test_package</name>", package_xml)
            self.assertIn("<depend>std_msgs</depend>", package_xml)
            self.assertIn("<depend>sensor_msgs</depend>", package_xml)
        
    def test_create_ros_launch_file(self):
        """Test creating a ROS launch file."""
        # Create a package first
        package_result = self.ros_integration.create_ros_package(
            package_name="test_package"
        )
        
        # Create launch file
        nodes = [
            {
                "name": "talker",
                "package": "test_package",
                "type": "talker_node.py",
                "output": "screen",
                "respawn": True,
                "params": {
                    "rate": 10,
                    "message": "Hello, ROS!"
                }
            },
            {
                "name": "listener",
                "package": "test_package",
                "type": "listener_node.py",
                "output": "screen",
                "respawn": False,
                "remappings": {
                    "input": "talker/output"
                }
            }
        ]
        
        result = self.ros_integration.create_ros_launch_file(
            package_name="test_package",
            launch_name="test_launch",
            nodes=nodes
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["package_name"], "test_package")
        self.assertEqual(result["launch_name"], "test_launch")
        
        # Check that launch file was created
        launch_file_path = result["launch_file_path"]
        self.assertTrue(os.path.exists(launch_file_path))
        
        # Check launch file content
        with open(launch_file_path, "r") as f:
            launch_file = f.read()
            self.assertIn("<launch>", launch_file)
            self.assertIn("<node name=\"talker\"", launch_file)
            self.assertIn("<node name=\"listener\"", launch_file)
            self.assertIn("<param name=\"rate\" value=10 />", launch_file)
            self.assertIn("<param name=\"message\" value=\"Hello, ROS!\" />", launch_file)
            self.assertIn("<remap from=\"input\" to=\"talker/output\" />", launch_file)
            
    def test_initialize_and_stop(self):
        """Test initializing and stopping the ROS node."""
        # Skip if ROS is not available
        if not self.ros_integration.ros_available:
            self.skipTest("ROS is not available")
            
        # Initialize ROS node
        result = self.ros_integration.initialize()
        
        # This might fail if ROS is not properly set up in the test environment
        if not result:
            self.skipTest("Failed to initialize ROS node")
            
        self.assertTrue(self.ros_integration.is_initialized)
        
        # Start ROS node
        start_result = self.ros_integration.start()
        self.assertTrue(start_result["success"])
        self.assertTrue(self.ros_integration.is_running)
        
        # Stop ROS node
        stop_result = self.ros_integration.stop()
        self.assertTrue(stop_result["success"])
        self.assertFalse(self.ros_integration.is_running)
        
    def test_create_robot_controller(self):
        """Test creating a robot controller."""
        # Skip if ROS is not available
        if not self.ros_integration.ros_available:
            self.skipTest("ROS is not available")
            
        # Initialize ROS node
        if not self.ros_integration.initialize():
            self.skipTest("Failed to initialize ROS node")
            
        # Create TurtleBot controller
        result = self.ros_integration.create_robot_controller("turtlebot")
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["robot_type"], "turtlebot")
        self.assertIn("/cmd_vel", result["publishers"])
        self.assertIn("/scan", result["subscribers"])
        
        # Check that publishers and subscribers were created
        self.assertIn("/cmd_vel", self.ros_integration.publishers)
        self.assertIn("/scan", self.ros_integration.subscribers)
        
    def test_move_robot(self):
        """Test sending movement commands to a robot."""
        # Skip if ROS is not available
        if not self.ros_integration.ros_available:
            self.skipTest("ROS is not available")
            
        # Initialize ROS node
        if not self.ros_integration.initialize():
            self.skipTest("Failed to initialize ROS node")
            
        # Create TurtleBot controller
        controller_result = self.ros_integration.create_robot_controller("turtlebot")
        if not controller_result["success"]:
            self.skipTest("Failed to create robot controller")
            
        # Send movement command
        command = {
            "linear_x": 0.5,
            "angular_z": 0.2
        }
        
        result = self.ros_integration.move_robot("turtlebot", command)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["robot_type"], "turtlebot")
        self.assertEqual(result["command"], command)

if __name__ == '__main__':
    unittest.main()