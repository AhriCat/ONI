import importlib
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Union, Callable
import traceback
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONIAppRouter:
    """
    Central router for ONI applications and tools.
    Dynamically loads and manages apps, providing a unified interface.
    """
    
    def __init__(self, app_directories: List[str] = None):
        """
        Initialize the ONI App Router.
        
        Args:
            app_directories: List of directories to scan for apps
        """
        self.app_directories = app_directories or ["oniapps", "tools"]
        self.apps = {}  # name -> module
        self.app_metadata = {}  # name -> metadata
        self.app_instances = {}  # name -> instance
        self.default_app = None
        
        # Load all available apps
        self._discover_apps()
        
        logger.info(f"ONI App Router initialized with {len(self.apps)} apps")
    
    def _discover_apps(self) -> None:
        """Discover all available ONI apps in the specified directories."""
        for directory in self.app_directories:
            if not os.path.exists(directory):
                logger.warning(f"App directory not found: {directory}")
                continue
                
            # Add directory to path if not already there
            if directory not in sys.path:
                sys.path.append(directory)
            
            # Scan for Python files
            for filename in os.listdir(directory):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = filename[:-3]  # Remove .py extension
                    full_module_name = f"{os.path.basename(directory)}.{module_name}"
                    
                    try:
                        # Import the module
                        module = importlib.import_module(full_module_name)
                        
                        # Check if it's a valid ONI app
                        if hasattr(module, "APP_NAME") and hasattr(module, "APP_DESCRIPTION"):
                            app_name = module.APP_NAME
                            
                            # Extract metadata
                            metadata = {
                                "name": app_name,
                                "description": getattr(module, "APP_DESCRIPTION", ""),
                                "version": getattr(module, "APP_VERSION", "1.0.0"),
                                "author": getattr(module, "APP_AUTHOR", "Unknown"),
                                "category": getattr(module, "APP_CATEGORY", "Miscellaneous"),
                                "dependencies": getattr(module, "APP_DEPENDENCIES", []),
                                "is_default": getattr(module, "APP_DEFAULT", False),
                                "module_path": full_module_name,
                                "file_path": os.path.join(directory, filename)
                            }
                            
                            # Register the app
                            self.apps[app_name] = module
                            self.app_metadata[app_name] = metadata
                            
                            # Set as default if specified
                            if metadata["is_default"] and self.default_app is None:
                                self.default_app = app_name
                            
                            logger.info(f"Discovered app: {app_name} ({metadata['category']})")
                            
                    except Exception as e:
                        logger.error(f"Error loading module {full_module_name}: {e}")
                        logger.debug(traceback.format_exc())
    
    def get_app_list(self) -> List[Dict[str, Any]]:
        """Get a list of all available apps with their metadata."""
        return [self.app_metadata[name] for name in sorted(self.app_metadata.keys())]
    
    def get_app_categories(self) -> Dict[str, List[str]]:
        """Get apps organized by category."""
        categories = {}
        
        for app_name, metadata in self.app_metadata.items():
            category = metadata["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(app_name)
        
        return categories
    
    def get_app_metadata(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific app."""
        return self.app_metadata.get(app_name)
    
    def get_app_instance(self, app_name: str) -> Any:
        """
        Get an instance of the specified app.
        Creates a new instance if one doesn't exist.
        """
        if app_name not in self.apps:
            raise ValueError(f"App not found: {app_name}")
        
        # Create instance if it doesn't exist
        if app_name not in self.app_instances:
            module = self.apps[app_name]
            
            # Check if the module has a main class with the same name
            class_name = app_name.replace("_", "").capitalize()
            if hasattr(module, class_name):
                app_class = getattr(module, class_name)
                self.app_instances[app_name] = app_class()
            else:
                # If no matching class, use the module itself
                self.app_instances[app_name] = module
        
        return self.app_instances[app_name]
    
    def run_app(self, app_name: str, method_name: str = "run", **kwargs) -> Any:
        """
        Run a method on the specified app.
        
        Args:
            app_name: Name of the app to run
            method_name: Name of the method to call (default: "run")
            **kwargs: Arguments to pass to the method
            
        Returns:
            Result of the method call
        """
        try:
            # Get app instance
            app = self.get_app_instance(app_name)
            
            # Get the method
            if hasattr(app, method_name):
                method = getattr(app, method_name)
                if callable(method):
                    # Call the method with the provided arguments
                    return method(**kwargs)
                else:
                    raise ValueError(f"Method {method_name} in app {app_name} is not callable")
            else:
                raise ValueError(f"Method {method_name} not found in app {app_name}")
                
        except Exception as e:
            logger.error(f"Error running app {app_name}.{method_name}: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def run_default_app(self, method_name: str = "run", **kwargs) -> Any:
        """Run the default app."""
        if self.default_app is None:
            raise ValueError("No default app specified")
        
        return self.run_app(self.default_app, method_name, **kwargs)
    
    def install_app(self, app_path: str) -> bool:
        """
        Install a new app from a Python file.
        
        Args:
            app_path: Path to the Python file containing the app
            
        Returns:
            bool: True if installation was successful
        """
        try:
            # Check if file exists
            if not os.path.exists(app_path):
                logger.error(f"App file not found: {app_path}")
                return False
            
            # Get filename and directory
            filename = os.path.basename(app_path)
            directory = os.path.dirname(app_path)
            
            # Check if it's a Python file
            if not filename.endswith(".py"):
                logger.error(f"Not a Python file: {app_path}")
                return False
            
            # Import the module
            module_name = filename[:-3]  # Remove .py extension
            
            # Add directory to path if not already there
            if directory not in sys.path:
                sys.path.append(directory)
            
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, app_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if it's a valid ONI app
            if not hasattr(module, "APP_NAME") or not hasattr(module, "APP_DESCRIPTION"):
                logger.error(f"Not a valid ONI app: {app_path}")
                return False
            
            app_name = module.APP_NAME
            
            # Extract metadata
            metadata = {
                "name": app_name,
                "description": getattr(module, "APP_DESCRIPTION", ""),
                "version": getattr(module, "APP_VERSION", "1.0.0"),
                "author": getattr(module, "APP_AUTHOR", "Unknown"),
                "category": getattr(module, "APP_CATEGORY", "Miscellaneous"),
                "dependencies": getattr(module, "APP_DEPENDENCIES", []),
                "is_default": getattr(module, "APP_DEFAULT", False),
                "module_path": module_name,
                "file_path": app_path
            }
            
            # Register the app
            self.apps[app_name] = module
            self.app_metadata[app_name] = metadata
            
            # Set as default if specified
            if metadata["is_default"] and self.default_app is None:
                self.default_app = app_name
            
            logger.info(f"Installed app: {app_name} ({metadata['category']})")
            return True
            
        except Exception as e:
            logger.error(f"Error installing app: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def uninstall_app(self, app_name: str) -> bool:
        """
        Uninstall an app.
        
        Args:
            app_name: Name of the app to uninstall
            
        Returns:
            bool: True if uninstallation was successful
        """
        if app_name not in self.apps:
            logger.warning(f"App not found: {app_name}")
            return False
        
        try:
            # Remove app instance if it exists
            if app_name in self.app_instances:
                # Call cleanup method if it exists
                app = self.app_instances[app_name]
                if hasattr(app, "cleanup"):
                    app.cleanup()
                
                del self.app_instances[app_name]
            
            # Remove app and metadata
            del self.apps[app_name]
            del self.app_metadata[app_name]
            
            # Update default app if needed
            if self.default_app == app_name:
                self.default_app = None
                
                # Find a new default app
                for name, metadata in self.app_metadata.items():
                    if metadata["is_default"]:
                        self.default_app = name
                        break
            
            logger.info(f"Uninstalled app: {app_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uninstalling app {app_name}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def reload_app(self, app_name: str) -> bool:
        """
        Reload an app.
        
        Args:
            app_name: Name of the app to reload
            
        Returns:
            bool: True if reload was successful
        """
        if app_name not in self.apps:
            logger.warning(f"App not found: {app_name}")
            return False
        
        try:
            # Get app metadata
            metadata = self.app_metadata[app_name]
            file_path = metadata["file_path"]
            
            # Uninstall the app
            self.uninstall_app(app_name)
            
            # Reinstall the app
            return self.install_app(file_path)
            
        except Exception as e:
            logger.error(f"Error reloading app {app_name}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def get_app_help(self, app_name: str) -> str:
        """Get help information for a specific app."""
        if app_name not in self.apps:
            return f"App not found: {app_name}"
        
        app = self.get_app_instance(app_name)
        
        # Check if the app has a help method
        if hasattr(app, "help"):
            help_method = getattr(app, "help")
            if callable(help_method):
                return help_method()
        
        # Fall back to docstring
        module = self.apps[app_name]
        return module.__doc__ or f"No help available for {app_name}"
    
    def get_app_methods(self, app_name: str) -> List[str]:
        """Get a list of available methods for a specific app."""
        if app_name not in self.apps:
            return []
        
        app = self.get_app_instance(app_name)
        
        # Get all public methods (not starting with underscore)
        methods = []
        for attr_name in dir(app):
            if not attr_name.startswith("_"):
                attr = getattr(app, attr_name)
                if callable(attr):
                    methods.append(attr_name)
        
        return methods
    
    def execute_command(self, command: str) -> Any:
        """
        Execute a command string in the format "app_name.method_name(arg1, arg2, kwarg1=value1)".
        
        Args:
            command: Command string to execute
            
        Returns:
            Result of the command execution
        """
        try:
            # Parse command
            if "." not in command:
                # Assume it's a method on the default app
                if self.default_app is None:
                    raise ValueError("No default app specified")
                
                app_name = self.default_app
                method_part = command
            else:
                # Split into app and method
                app_name, method_part = command.split(".", 1)
            
            # Parse method and arguments
            if "(" not in method_part or not method_part.endswith(")"):
                raise ValueError(f"Invalid method format: {method_part}")
            
            method_name = method_part[:method_part.index("(")]
            args_str = method_part[method_part.index("(") + 1:method_part.rindex(")")]
            
            # Parse arguments
            args = []
            kwargs = {}
            
            if args_str.strip():
                # Split by commas, but respect nested structures
                import re
                
                # This regex handles nested structures like lists and dicts
                parts = re.findall(r'(?:[^,()]|\([^()]*\))+', args_str)
                
                for part in parts:
                    part = part.strip()
                    if "=" in part:
                        # Keyword argument
                        key, value = part.split("=", 1)
                        kwargs[key.strip()] = eval(value)
                    else:
                        # Positional argument
                        args.append(eval(part))
            
            # Run the app method
            return self.run_app(app_name, method_name, *args, **kwargs)
            
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def get_app_status(self) -> Dict[str, Any]:
        """Get status information about all apps."""
        status = {
            "total_apps": len(self.apps),
            "loaded_instances": len(self.app_instances),
            "default_app": self.default_app,
            "categories": self.get_app_categories(),
            "timestamp": time.time()
        }
        
        return status
    
    def cleanup(self) -> None:
        """Clean up all app instances."""
        for app_name, app in self.app_instances.items():
            try:
                if hasattr(app, "cleanup"):
                    app.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up app {app_name}: {e}")
        
        self.app_instances.clear()
        logger.info("All app instances cleaned up")

# Example usage
if __name__ == "__main__":
    router = ONIAppRouter()
    
    # Get list of available apps
    apps = router.get_app_list()
    print(f"Available apps: {json.dumps(apps, indent=2)}")
    
    # Get app categories
    categories = router.get_app_categories()
    print(f"App categories: {json.dumps(categories, indent=2)}")
    
    # Run an app (if available)
    if apps:
        app_name = apps[0]["name"]
        try:
            result = router.run_app(app_name)
            print(f"Result of running {app_name}: {result}")
        except Exception as e:
            print(f"Error running {app_name}: {e}")
    
    # Clean up
    router.cleanup()