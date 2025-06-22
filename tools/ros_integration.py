import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import threading
import queue
import traceback
import subprocess
import tempfile
import shutil

# Import ROS library if available
try:
    import rospy
    import rosnodejs
    from std_msgs.msg import String, Float32, Int32, Bool
    from geometry_msgs.msg import Twist, Pose, Point, Quaternion
    from sensor_msgs.msg import Image, LaserScan, PointCloud2, JointState
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ROSIntegration:
    """Robot Operating System (ROS) integration for ONI."""
    
    def __init__(self, node_name: str = "oni_ros_node", use_rosnodejs: bool = False):
        """
        Initialize ROS integration.
        
        Args:
            node_name: Name for the ROS node
            use_rosnodejs: Whether to use rosnodejs (for JavaScript) instead of rospy
        """
        self.node_name = node_name
        self.use_rosnodejs = use_rosnodejs
        self.ros_available = HAS_ROS
        
        # Track publishers and subscribers
        self.publishers = {}
        self.subscribers = {}
        self.subscriber_callbacks = {}
        
        # Message queue for received messages
        self.message_queue = queue.Queue()
        
        # Track initialization state
        self.is_initialized = False
        self.is_running = False
        self.ros_thread = None
        
        logger.info(f"ROS integration initialized with node name: {node_name}")
    
    def initialize(self) -> bool:
        """Initialize ROS node."""
        if not self.ros_available:
            logger.error("ROS is not available. Please install ROS and the required Python packages.")
            return False
        
        try:
            if self.use_rosnodejs:
                # Initialize rosnodejs
                rosnodejs.initNode(self.node_name)
                self.nh = rosnodejs.nh
            else:
                # Initialize rospy
                rospy.init_node(self.node_name, anonymous=True, disable_signals=True)
            
            self.is_initialized = True
            logger.info(f"ROS node '{self.node_name}' initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ROS node: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def create_publisher(self, topic: str, msg_type: str, queue_size: int = 10) -> Dict[str, Any]:
        """
        Create a ROS publisher.
        
        Args:
            topic: ROS topic to publish to
            msg_type: Message type (e.g., 'std_msgs/String', 'geometry_msgs/Twist')
            queue_size: Publisher queue size
            
        Returns:
            Dictionary with publisher creation result
        """
        if not self.is_initialized:
            return {"success": False, "error": "ROS node not initialized"}
        
        try:
            # Get message class based on type
            msg_class = self._get_message_class(msg_type)
            
            if not msg_class:
                return {"success": False, "error": f"Unsupported message type: {msg_type}"}
            
            # Create publisher
            if self.use_rosnodejs:
                publisher = self.nh.advertise(topic, msg_class._type, {'queueSize': queue_size})
            else:
                publisher = rospy.Publisher(topic, msg_class, queue_size=queue_size)
            
            # Store publisher
            self.publishers[topic] = {
                "publisher": publisher,
                "msg_type": msg_type,
                "msg_class": msg_class
            }
            
            logger.info(f"Created publisher for topic: {topic} with type: {msg_type}")
            return {
                "success": True,
                "topic": topic,
                "msg_type": msg_type
            }
            
        except Exception as e:
            logger.error(f"Failed to create publisher: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_subscriber(self, topic: str, msg_type: str, queue_size: int = 10) -> Dict[str, Any]:
        """
        Create a ROS subscriber.
        
        Args:
            topic: ROS topic to subscribe to
            msg_type: Message type (e.g., 'std_msgs/String', 'sensor_msgs/LaserScan')
            queue_size: Subscriber queue size
            
        Returns:
            Dictionary with subscriber creation result
        """
        if not self.is_initialized:
            return {"success": False, "error": "ROS node not initialized"}
        
        try:
            # Get message class based on type
            msg_class = self._get_message_class(msg_type)
            
            if not msg_class:
                return {"success": False, "error": f"Unsupported message type: {msg_type}"}
            
            # Create callback function
            def callback(msg):
                # Convert message to dictionary
                msg_dict = self._message_to_dict(msg)
                
                # Add to message queue
                self.message_queue.put({
                    "topic": topic,
                    "msg_type": msg_type,
                    "data": msg_dict,
                    "timestamp": time.time()
                })
                
                # Call custom callback if registered
                if topic in self.subscriber_callbacks:
                    self.subscriber_callbacks[topic](msg_dict)
            
            # Create subscriber
            if self.use_rosnodejs:
                subscriber = self.nh.subscribe(topic, msg_class._type, callback, {'queueSize': queue_size})
            else:
                subscriber = rospy.Subscriber(topic, msg_class, callback, queue_size=queue_size)
            
            # Store subscriber
            self.subscribers[topic] = {
                "subscriber": subscriber,
                "msg_type": msg_type,
                "msg_class": msg_class
            }
            
            logger.info(f"Created subscriber for topic: {topic} with type: {msg_type}")
            return {
                "success": True,
                "topic": topic,
                "msg_type": msg_type
            }
            
        except Exception as e:
            logger.error(f"Failed to create subscriber: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def register_callback(self, topic: str, callback) -> Dict[str, Any]:
        """
        Register a callback function for a subscribed topic.
        
        Args:
            topic: ROS topic
            callback: Callback function that takes a message dictionary
            
        Returns:
            Dictionary with registration result
        """
        if topic not in self.subscribers:
            return {"success": False, "error": f"Not subscribed to topic: {topic}"}
        
        try:
            # Register callback
            self.subscriber_callbacks[topic] = callback
            
            logger.info(f"Registered callback for topic: {topic}")
            return {
                "success": True,
                "topic": topic
            }
            
        except Exception as e:
            logger.error(f"Failed to register callback: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def publish_message(self, topic: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish a message to a ROS topic.
        
        Args:
            topic: ROS topic to publish to
            data: Message data as a dictionary
            
        Returns:
            Dictionary with publish result
        """
        if not self.is_initialized:
            return {"success": False, "error": "ROS node not initialized"}
        
        if topic not in self.publishers:
            return {"success": False, "error": f"No publisher for topic: {topic}"}
        
        try:
            # Get publisher info
            publisher_info = self.publishers[topic]
            publisher = publisher_info["publisher"]
            msg_class = publisher_info["msg_class"]
            
            # Create message
            msg = self._dict_to_message(data, msg_class)
            
            # Publish message
            publisher.publish(msg)
            
            logger.info(f"Published message to topic: {topic}")
            return {
                "success": True,
                "topic": topic,
                "data": data
            }
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_latest_message(self, topic: Optional[str] = None, timeout: float = 0.1) -> Dict[str, Any]:
        """
        Get the latest message from a topic.
        
        Args:
            topic: ROS topic (if None, get any message)
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with message data
        """
        try:
            # Check if we're subscribed to the topic
            if topic and topic not in self.subscribers:
                return {"success": False, "error": f"Not subscribed to topic: {topic}"}
            
            # Get message from queue
            try:
                message = self.message_queue.get(timeout=timeout)
                
                # Check if message is for the requested topic
                if topic and message["topic"] != topic:
                    return {"success": False, "error": f"No message for topic: {topic}"}
                
                return {
                    "success": True,
                    "topic": message["topic"],
                    "msg_type": message["msg_type"],
                    "data": message["data"],
                    "timestamp": message["timestamp"]
                }
                
            except queue.Empty:
                return {"success": False, "error": "No message available"}
                
        except Exception as e:
            logger.error(f"Failed to get latest message: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def wait_for_message(self, topic: str, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Wait for a message on a specific topic.
        
        Args:
            topic: ROS topic
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with message data
        """
        if not self.is_initialized:
            return {"success": False, "error": "ROS node not initialized"}
        
        if topic not in self.subscribers:
            return {"success": False, "error": f"Not subscribed to topic: {topic}"}
        
        try:
            # Wait for message
            if self.use_rosnodejs:
                # For rosnodejs, we need to use our own queue
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        message = self.message_queue.get(timeout=0.1)
                        if message["topic"] == topic:
                            return {
                                "success": True,
                                "topic": message["topic"],
                                "msg_type": message["msg_type"],
                                "data": message["data"],
                                "timestamp": message["timestamp"]
                            }
                    except queue.Empty:
                        pass
                
                return {"success": False, "error": f"Timeout waiting for message on topic: {topic}"}
            else:
                # For rospy, we can use the built-in wait_for_message
                msg_class = self.subscribers[topic]["msg_class"]
                msg = rospy.wait_for_message(topic, msg_class, timeout=timeout)
                
                # Convert message to dictionary
                msg_dict = self._message_to_dict(msg)
                
                return {
                    "success": True,
                    "topic": topic,
                    "msg_type": self.subscribers[topic]["msg_type"],
                    "data": msg_dict,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"Failed to wait for message: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_topic_list(self) -> Dict[str, Any]:
        """
        Get list of available ROS topics.
        
        Returns:
            Dictionary with topic list
        """
        if not self.is_initialized:
            return {"success": False, "error": "ROS node not initialized"}
        
        try:
            if self.use_rosnodejs:
                # For rosnodejs, we need to use the ROS master API
                topics = rosnodejs.getTopicsAndTypes()
                
                formatted_topics = []
                for topic in topics:
                    formatted_topics.append({
                        "name": topic.name,
                        "type": topic.type
                    })
            else:
                # For rospy, we can use get_published_topics
                topics = rospy.get_published_topics()
                
                formatted_topics = []
                for topic_name, topic_type in topics:
                    formatted_topics.append({
                        "name": topic_name,
                        "type": topic_type
                    })
            
            logger.info(f"Got {len(formatted_topics)} ROS topics")
            return {
                "success": True,
                "topics": formatted_topics
            }
            
        except Exception as e:
            logger.error(f"Failed to get topic list: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_param(self, param_name: str, default: Any = None) -> Dict[str, Any]:
        """
        Get a ROS parameter.
        
        Args:
            param_name: Parameter name
            default: Default value if parameter doesn't exist
            
        Returns:
            Dictionary with parameter value
        """
        if not self.is_initialized:
            return {"success": False, "error": "ROS node not initialized"}
        
        try:
            if self.use_rosnodejs:
                # For rosnodejs, we need to use the parameter server
                if rosnodejs.hasParam(param_name):
                    value = rosnodejs.getParam(param_name)
                else:
                    value = default
            else:
                # For rospy, we can use get_param
                value = rospy.get_param(param_name, default)
            
            logger.info(f"Got ROS parameter: {param_name}")
            return {
                "success": True,
                "param_name": param_name,
                "value": value
            }
            
        except Exception as e:
            logger.error(f"Failed to get parameter: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def set_param(self, param_name: str, value: Any) -> Dict[str, Any]:
        """
        Set a ROS parameter.
        
        Args:
            param_name: Parameter name
            value: Parameter value
            
        Returns:
            Dictionary with operation result
        """
        if not self.is_initialized:
            return {"success": False, "error": "ROS node not initialized"}
        
        try:
            if self.use_rosnodejs:
                # For rosnodejs, we need to use the parameter server
                rosnodejs.setParam(param_name, value)
            else:
                # For rospy, we can use set_param
                rospy.set_param(param_name, value)
            
            logger.info(f"Set ROS parameter: {param_name}")
            return {
                "success": True,
                "param_name": param_name,
                "value": value
            }
            
        except Exception as e:
            logger.error(f"Failed to set parameter: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def start(self) -> Dict[str, Any]:
        """
        Start ROS node in a separate thread.
        
        Returns:
            Dictionary with start result
        """
        if not self.is_initialized:
            init_result = self.initialize()
            if not init_result:
                return {"success": False, "error": "Failed to initialize ROS node"}
        
        if self.is_running:
            return {"success": True, "message": "ROS node already running"}
        
        try:
            # Start ROS node in a separate thread
            self.is_running = True
            self.ros_thread = threading.Thread(target=self._ros_thread_func)
            self.ros_thread.daemon = True
            self.ros_thread.start()
            
            logger.info(f"Started ROS node: {self.node_name}")
            return {
                "success": True,
                "node_name": self.node_name
            }
            
        except Exception as e:
            logger.error(f"Failed to start ROS node: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop ROS node.
        
        Returns:
            Dictionary with stop result
        """
        if not self.is_running:
            return {"success": True, "message": "ROS node not running"}
        
        try:
            # Stop ROS node
            self.is_running = False
            
            if self.ros_thread and self.ros_thread.is_alive():
                self.ros_thread.join(timeout=1.0)
            
            # Unregister all publishers and subscribers
            for topic, publisher_info in self.publishers.items():
                if self.use_rosnodejs:
                    publisher_info["publisher"].shutdown()
                else:
                    publisher_info["publisher"].unregister()
            
            for topic, subscriber_info in self.subscribers.items():
                if self.use_rosnodejs:
                    subscriber_info["subscriber"].shutdown()
                else:
                    subscriber_info["subscriber"].unregister()
            
            # Clear collections
            self.publishers = {}
            self.subscribers = {}
            self.subscriber_callbacks = {}
            
            # Clear message queue
            while not self.message_queue.empty():
                self.message_queue.get_nowait()
            
            logger.info(f"Stopped ROS node: {self.node_name}")
            return {
                "success": True,
                "node_name": self.node_name
            }
            
        except Exception as e:
            logger.error(f"Failed to stop ROS node: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def _ros_thread_func(self) -> None:
        """Thread function for ROS node."""
        try:
            if self.use_rosnodejs:
                # For rosnodejs, we need to use the event loop
                rosnodejs.spin()
            else:
                # For rospy, we can use the built-in spin
                while self.is_running and not rospy.is_shutdown():
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in ROS thread: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.is_running = False
    
    def _get_message_class(self, msg_type: str):
        """Get message class based on type string."""
        if not self.ros_available:
            return None
        
        # Parse message type
        parts = msg_type.split('/')
        if len(parts) != 2:
            logger.error(f"Invalid message type format: {msg_type}")
            return None
        
        package, msg_name = parts
        
        try:
            if self.use_rosnodejs:
                # For rosnodejs, we need to use the message registry
                return rosnodejs.require(package).msg[msg_name]
            else:
                # For rospy, we need to import the message class
                if package == "std_msgs":
                    if msg_name == "String":
                        return String
                    elif msg_name == "Float32":
                        return Float32
                    elif msg_name == "Int32":
                        return Int32
                    elif msg_name == "Bool":
                        return Bool
                elif package == "geometry_msgs":
                    if msg_name == "Twist":
                        return Twist
                    elif msg_name == "Pose":
                        return Pose
                    elif msg_name == "Point":
                        return Point
                    elif msg_name == "Quaternion":
                        return Quaternion
                elif package == "sensor_msgs":
                    if msg_name == "Image":
                        return Image
                    elif msg_name == "LaserScan":
                        return LaserScan
                    elif msg_name == "PointCloud2":
                        return PointCloud2
                    elif msg_name == "JointState":
                        return JointState
                
                # If we get here, the message type is not supported
                logger.error(f"Unsupported message type: {msg_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get message class for {msg_type}: {e}")
            return None
    
    def _message_to_dict(self, msg) -> Dict[str, Any]:
        """Convert a ROS message to a dictionary."""
        if not self.ros_available:
            return {}
        
        try:
            if self.use_rosnodejs:
                # For rosnodejs, messages are already JavaScript objects
                return msg
            else:
                # For rospy, we need to convert the message to a dictionary
                if hasattr(msg, "_slot_types"):
                    # Standard ROS message
                    result = {}
                    for slot, slot_type in zip(msg.__slots__, msg._slot_types):
                        value = getattr(msg, slot)
                        
                        # Handle nested messages
                        if hasattr(value, "_slot_types"):
                            result[slot] = self._message_to_dict(value)
                        # Handle arrays
                        elif isinstance(value, (list, tuple)):
                            if len(value) > 0 and hasattr(value[0], "_slot_types"):
                                result[slot] = [self._message_to_dict(item) for item in value]
                            else:
                                result[slot] = list(value)
                        # Handle basic types
                        else:
                            result[slot] = value
                    
                    return result
                else:
                    # Not a standard ROS message
                    return {"value": str(msg)}
                
        except Exception as e:
            logger.error(f"Failed to convert message to dictionary: {e}")
            return {"error": str(e)}
    
    def _dict_to_message(self, data: Dict[str, Any], msg_class) -> Any:
        """Convert a dictionary to a ROS message."""
        if not self.ros_available:
            return None
        
        try:
            if self.use_rosnodejs:
                # For rosnodejs, we can create a message directly from an object
                return new msg_class(data)
            else:
                # For rospy, we need to create a message and set its fields
                msg = msg_class()
                
                for key, value in data.items():
                    if hasattr(msg, key):
                        # Get the attribute
                        attr = getattr(msg, key)
                        
                        # Handle nested messages
                        if hasattr(attr, "_slot_types"):
                            if isinstance(value, dict):
                                setattr(msg, key, self._dict_to_message(value, type(attr)))
                        # Handle arrays
                        elif isinstance(attr, (list, tuple)):
                            if len(attr) > 0 and hasattr(attr[0], "_slot_types"):
                                # Array of messages
                                if isinstance(value, list):
                                    setattr(msg, key, [self._dict_to_message(item, type(attr[0])) for item in value])
                            else:
                                # Array of basic types
                                setattr(msg, key, value)
                        # Handle basic types
                        else:
                            setattr(msg, key, value)
                
                return msg
                
        except Exception as e:
            logger.error(f"Failed to convert dictionary to message: {e}")
            return None
    
    def create_robot_controller(self, robot_type: str) -> Dict[str, Any]:
        """
        Create a controller for a specific type of robot.
        
        Args:
            robot_type: Type of robot ('turtlebot', 'ur5', 'custom')
            
        Returns:
            Dictionary with controller creation result
        """
        if not self.is_initialized:
            return {"success": False, "error": "ROS node not initialized"}
        
        try:
            if robot_type == "turtlebot":
                # Create publisher for velocity commands
                vel_pub_result = self.create_publisher("/cmd_vel", "geometry_msgs/Twist")
                
                if not vel_pub_result.get("success", False):
                    return vel_pub_result
                
                # Create subscribers for sensor data
                scan_sub_result = self.create_subscriber("/scan", "sensor_msgs/LaserScan")
                
                if not scan_sub_result.get("success", False):
                    return scan_sub_result
                
                odom_sub_result = self.create_subscriber("/odom", "nav_msgs/Odometry")
                
                logger.info(f"Created TurtleBot controller")
                return {
                    "success": True,
                    "robot_type": "turtlebot",
                    "publishers": ["/cmd_vel"],
                    "subscribers": ["/scan", "/odom"]
                }
                
            elif robot_type == "ur5":
                # Create publisher for joint trajectory
                joint_pub_result = self.create_publisher("/joint_trajectory", "trajectory_msgs/JointTrajectory")
                
                if not joint_pub_result.get("success", False):
                    return joint_pub_result
                
                # Create subscribers for joint states
                joint_sub_result = self.create_subscriber("/joint_states", "sensor_msgs/JointState")
                
                if not joint_sub_result.get("success", False):
                    return joint_sub_result
                
                logger.info(f"Created UR5 controller")
                return {
                    "success": True,
                    "robot_type": "ur5",
                    "publishers": ["/joint_trajectory"],
                    "subscribers": ["/joint_states"]
                }
                
            elif robot_type == "custom":
                # For custom robots, just return success
                logger.info(f"Created custom robot controller")
                return {
                    "success": True,
                    "robot_type": "custom",
                    "message": "Custom robot controller created. Add publishers and subscribers as needed."
                }
                
            else:
                return {"success": False, "error": f"Unsupported robot type: {robot_type}"}
                
        except Exception as e:
            logger.error(f"Failed to create robot controller: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def move_robot(self, robot_type: str, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send movement commands to a robot.
        
        Args:
            robot_type: Type of robot ('turtlebot', 'ur5', 'custom')
            command: Movement command parameters
            
        Returns:
            Dictionary with movement result
        """
        if not self.is_initialized:
            return {"success": False, "error": "ROS node not initialized"}
        
        try:
            if robot_type == "turtlebot":
                # Check if we have the required publisher
                if "/cmd_vel" not in self.publishers:
                    return {"success": False, "error": "TurtleBot controller not initialized"}
                
                # Create Twist message
                linear_x = command.get("linear_x", 0.0)
                linear_y = command.get("linear_y", 0.0)
                linear_z = command.get("linear_z", 0.0)
                angular_x = command.get("angular_x", 0.0)
                angular_y = command.get("angular_y", 0.0)
                angular_z = command.get("angular_z", 0.0)
                
                twist_data = {
                    "linear": {
                        "x": linear_x,
                        "y": linear_y,
                        "z": linear_z
                    },
                    "angular": {
                        "x": angular_x,
                        "y": angular_y,
                        "z": angular_z
                    }
                }
                
                # Publish message
                result = self.publish_message("/cmd_vel", twist_data)
                
                if not result.get("success", False):
                    return result
                
                logger.info(f"Sent movement command to TurtleBot")
                return {
                    "success": True,
                    "robot_type": "turtlebot",
                    "command": command
                }
                
            elif robot_type == "ur5":
                # Check if we have the required publisher
                if "/joint_trajectory" not in self.publishers:
                    return {"success": False, "error": "UR5 controller not initialized"}
                
                # Create JointTrajectory message
                joint_positions = command.get("joint_positions", [])
                
                if not joint_positions or len(joint_positions) != 6:
                    return {"success": False, "error": "Invalid joint positions for UR5"}
                
                trajectory_data = {
                    "header": {
                        "stamp": {
                            "secs": int(time.time()),
                            "nsecs": int((time.time() % 1) * 1e9)
                        },
                        "frame_id": ""
                    },
                    "joint_names": [
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint"
                    ],
                    "points": [
                        {
                            "positions": joint_positions,
                            "velocities": [0.0] * 6,
                            "accelerations": [0.0] * 6,
                            "effort": [0.0] * 6,
                            "time_from_start": {
                                "secs": 1,
                                "nsecs": 0
                            }
                        }
                    ]
                }
                
                # Publish message
                result = self.publish_message("/joint_trajectory", trajectory_data)
                
                if not result.get("success", False):
                    return result
                
                logger.info(f"Sent movement command to UR5")
                return {
                    "success": True,
                    "robot_type": "ur5",
                    "command": command
                }
                
            elif robot_type == "custom":
                # For custom robots, check if the specified topic exists
                topic = command.get("topic")
                data = command.get("data", {})
                
                if not topic:
                    return {"success": False, "error": "No topic specified for custom robot"}
                
                if topic not in self.publishers:
                    return {"success": False, "error": f"No publisher for topic: {topic}"}
                
                # Publish message
                result = self.publish_message(topic, data)
                
                if not result.get("success", False):
                    return result
                
                logger.info(f"Sent command to custom robot on topic: {topic}")
                return {
                    "success": True,
                    "robot_type": "custom",
                    "topic": topic,
                    "command": command
                }
                
            else:
                return {"success": False, "error": f"Unsupported robot type: {robot_type}"}
                
        except Exception as e:
            logger.error(f"Failed to move robot: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_robot_state(self, robot_type: str) -> Dict[str, Any]:
        """
        Get the current state of a robot.
        
        Args:
            robot_type: Type of robot ('turtlebot', 'ur5', 'custom')
            
        Returns:
            Dictionary with robot state
        """
        if not self.is_initialized:
            return {"success": False, "error": "ROS node not initialized"}
        
        try:
            if robot_type == "turtlebot":
                # Check if we have the required subscribers
                if "/scan" not in self.subscribers or "/odom" not in self.subscribers:
                    return {"success": False, "error": "TurtleBot controller not initialized"}
                
                # Get latest scan message
                scan_result = self.wait_for_message("/scan", timeout=1.0)
                
                # Get latest odometry message
                odom_result = self.wait_for_message("/odom", timeout=1.0)
                
                # Combine results
                return {
                    "success": True,
                    "robot_type": "turtlebot",
                    "scan": scan_result.get("data") if scan_result.get("success", False) else None,
                    "odometry": odom_result.get("data") if odom_result.get("success", False) else None
                }
                
            elif robot_type == "ur5":
                # Check if we have the required subscriber
                if "/joint_states" not in self.subscribers:
                    return {"success": False, "error": "UR5 controller not initialized"}
                
                # Get latest joint states message
                joint_result = self.wait_for_message("/joint_states", timeout=1.0)
                
                return {
                    "success": True,
                    "robot_type": "ur5",
                    "joint_states": joint_result.get("data") if joint_result.get("success", False) else None
                }
                
            elif robot_type == "custom":
                # For custom robots, return all received messages
                messages = {}
                
                # Get latest message from each subscribed topic
                for topic in self.subscribers:
                    result = self.get_latest_message(topic)
                    if result.get("success", False):
                        messages[topic] = result.get("data")
                
                return {
                    "success": True,
                    "robot_type": "custom",
                    "messages": messages
                }
                
            else:
                return {"success": False, "error": f"Unsupported robot type: {robot_type}"}
                
        except Exception as e:
            logger.error(f"Failed to get robot state: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_ros_package(self, package_name: str, dependencies: List[str] = None) -> Dict[str, Any]:
        """
        Create a new ROS package.
        
        Args:
            package_name: Name for the package
            dependencies: List of package dependencies
            
        Returns:
            Dictionary with package creation result
        """
        try:
            # Create a temporary directory for the package
            package_dir = os.path.join(self.work_dir, package_name)
            os.makedirs(package_dir, exist_ok=True)
            
            # Create package.xml
            package_xml = f"""<?xml version="1.0"?>
<package format="2">
  <name>{package_name}</name>
  <version>0.0.1</version>
  <description>
    A package created by ONI
  </description>
  <maintainer email="oni@example.com">ONI</maintainer>
  <license>MIT</license>
  
  <buildtool_depend>catkin</buildtool_depend>
"""
            
            # Add dependencies
            if dependencies:
                for dep in dependencies:
                    package_xml += f"  <depend>{dep}</depend>\n"
            
            package_xml += """
  <export>
  </export>
</package>
"""
            
            # Write package.xml
            with open(os.path.join(package_dir, "package.xml"), "w") as f:
                f.write(package_xml)
            
            # Create CMakeLists.txt
            cmake_lists = f"""cmake_minimum_required(VERSION 3.0.2)
project({package_name})

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
"""
            
            # Add dependencies
            if dependencies:
                for dep in dependencies:
                    cmake_lists += f"  {dep}\n"
            
            cmake_lists += """)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Add executables and libraries here
"""
            
            # Write CMakeLists.txt
            with open(os.path.join(package_dir, "CMakeLists.txt"), "w") as f:
                f.write(cmake_lists)
            
            # Create src directory
            os.makedirs(os.path.join(package_dir, "src"), exist_ok=True)
            
            # Create include directory
            os.makedirs(os.path.join(package_dir, "include", package_name), exist_ok=True)
            
            # Create launch directory
            os.makedirs(os.path.join(package_dir, "launch"), exist_ok=True)
            
            # Create a simple launch file
            launch_file = f"""<launch>
  <!-- Launch file for {package_name} -->
  
  <!-- Parameters -->
  <param name="use_sim_time" value="false" />
  
  <!-- Nodes -->
  <!-- Add your nodes here -->
  
</launch>
"""
            
            # Write launch file
            with open(os.path.join(package_dir, "launch", f"{package_name}.launch"), "w") as f:
                f.write(launch_file)
            
            # Create a simple Python node
            python_node = f"""#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def {package_name}_node():
    # Initialize node
    rospy.init_node('{package_name}_node', anonymous=True)
    
    # Create publisher
    pub = rospy.Publisher('output', String, queue_size=10)
    
    # Set rate
    rate = rospy.Rate(1)  # 1 Hz
    
    # Main loop
    while not rospy.is_shutdown():
        # Create message
        msg = String()
        msg.data = "Hello from {package_name}"
        
        # Publish message
        pub.publish(msg)
        
        # Sleep to maintain rate
        rate.sleep()

if __name__ == '__main__':
    try:
        {package_name}_node()
    except rospy.ROSInterruptException:
        pass
"""
            
            # Write Python node
            python_node_path = os.path.join(package_dir, "src", f"{package_name}_node.py")
            with open(python_node_path, "w") as f:
                f.write(python_node)
            
            # Make Python node executable
            os.chmod(python_node_path, 0o755)
            
            logger.info(f"Created ROS package: {package_name}")
            return {
                "success": True,
                "package_name": package_name,
                "package_dir": package_dir
            }
            
        except Exception as e:
            logger.error(f"Failed to create ROS package: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_ros_launch_file(self, package_name: str, launch_name: str, 
                             nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a ROS launch file.
        
        Args:
            package_name: Name of the package
            launch_name: Name for the launch file
            nodes: List of node configurations
            
        Returns:
            Dictionary with launch file creation result
        """
        try:
            # Create package directory if it doesn't exist
            package_dir = os.path.join(self.work_dir, package_name)
            os.makedirs(package_dir, exist_ok=True)
            
            # Create launch directory if it doesn't exist
            launch_dir = os.path.join(package_dir, "launch")
            os.makedirs(launch_dir, exist_ok=True)
            
            # Create launch file content
            launch_file = f"""<launch>
  <!-- Launch file for {launch_name} -->
  
  <!-- Parameters -->
  <param name="use_sim_time" value="false" />
  
  <!-- Nodes -->
"""
            
            # Add nodes
            for node in nodes:
                node_type = node.get("type", "")
                node_name = node.get("name", "")
                node_pkg = node.get("package", "")
                node_output = node.get("output", "screen")
                node_respawn = str(node.get("respawn", "false")).lower()
                
                launch_file += f"""  <node name="{node_name}" pkg="{node_pkg}" type="{node_type}" output="{node_output}" respawn="{node_respawn}">
"""
                
                # Add parameters
                params = node.get("params", {})
                for param_name, param_value in params.items():
                    if isinstance(param_value, bool):
                        param_value = str(param_value).lower()
                    elif isinstance(param_value, (int, float)):
                        param_value = str(param_value)
                    elif isinstance(param_value, str):
                        param_value = f'"{param_value}"'
                    
                    launch_file += f"""    <param name="{param_name}" value={param_value} />
"""
                
                # Add remappings
                remappings = node.get("remappings", {})
                for from_topic, to_topic in remappings.items():
                    launch_file += f"""    <remap from="{from_topic}" to="{to_topic}" />
"""
                
                launch_file += """  </node>
  
"""
            
            launch_file += """</launch>
"""
            
            # Write launch file
            launch_file_path = os.path.join(launch_dir, f"{launch_name}.launch")
            with open(launch_file_path, "w") as f:
                f.write(launch_file)
            
            logger.info(f"Created ROS launch file: {launch_file_path}")
            return {
                "success": True,
                "package_name": package_name,
                "launch_name": launch_name,
                "launch_file_path": launch_file_path
            }
            
        except Exception as e:
            logger.error(f"Failed to create ROS launch file: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_roslaunch(self, package_name: str, launch_name: str) -> Dict[str, Any]:
        """
        Run a ROS launch file.
        
        Args:
            package_name: Name of the package
            launch_name: Name of the launch file
            
        Returns:
            Dictionary with launch result
        """
        try:
            # Check if the launch file exists
            launch_file_path = os.path.join(self.work_dir, package_name, "launch", f"{launch_name}.launch")
            
            if not os.path.exists(launch_file_path):
                return {"success": False, "error": f"Launch file not found: {launch_file_path}"}
            
            # Run roslaunch
            cmd = ["roslaunch", launch_file_path]
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for process to start
            time.sleep(1.0)
            
            # Check if process is running
            if process.poll() is not None:
                # Process has exited
                stdout, stderr = process.communicate()
                return {
                    "success": False,
                    "error": f"roslaunch failed: {stderr}",
                    "output": stdout
                }
            
            logger.info(f"Started roslaunch: {package_name} {launch_name}")
            return {
                "success": True,
                "package_name": package_name,
                "launch_name": launch_name,
                "process_pid": process.pid
            }
            
        except Exception as e:
            logger.error(f"Failed to run roslaunch: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    # Create ROS integration
    ros_integration = ROSIntegration(node_name="oni_ros_example")
    
    # Initialize ROS node
    if ros_integration.initialize():
        # Start ROS node
        ros_integration.start()
        
        # Create publisher
        ros_integration.create_publisher("/oni/output", "std_msgs/String")
        
        # Create subscriber
        ros_integration.create_subscriber("/oni/input", "std_msgs/String")
        
        # Publish a message
        ros_integration.publish_message("/oni/output", {"data": "Hello from ONI!"})
        
        # Wait for a message
        message = ros_integration.wait_for_message("/oni/input", timeout=5.0)
        print(f"Received message: {message}")
        
        # Stop ROS node
        ros_integration.stop()
    else:
        print("Failed to initialize ROS node")