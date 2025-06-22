import logging
import time
import json
import os
import sys
import importlib
import inspect
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import traceback
import asyncio
import threading
import queue

# Import enhanced tool integration modules
from .api_discovery import APIDiscovery
from .tool_chaining import ToolChain, ToolChainRegistry
from .error_recovery import ErrorRecovery, ErrorClassifier
from .security_sandboxing import SecuritySandbox

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedToolIntegration:
    """
    Enhanced tool integration module for ONI.
    
    This module provides:
    1. API Discovery: Automatic API learning and integration
    2. Tool Chaining: Complex multi-tool workflows
    3. Error Recovery: Robust handling of tool failures
    4. Security Sandboxing: Safe execution environments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced tool integration.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Initialize components
        self.api_discovery = APIDiscovery(
            cache_dir=self.config.get('api_cache_dir')
        )
        
        self.tool_registry = ToolChainRegistry()
        
        self.error_recovery = ErrorRecovery(
            max_retries=self.config.get('max_retries', 3),
            retry_delay=self.config.get('retry_delay', 1.0),
            error_log_path=self.config.get('error_log_path')
        )
        
        self.security_sandbox = SecuritySandbox(
            sandbox_dir=self.config.get('sandbox_dir'),
            enable_network=self.config.get('enable_network', False),
            enable_file_io=self.config.get('enable_file_io', False),
            resource_limits=self.config.get('resource_limits'),
            allowed_modules=self.config.get('allowed_modules')
        )
        
        # Tool registry
        self.tools = {}
        
        # Register built-in tools
        self._register_builtin_tools()
        
        logger.info("Enhanced tool integration initialized")
    
    def _register_builtin_tools(self):
        """Register built-in tools."""
        # API Discovery tools
        self.register_tool(
            "discover_api_from_url",
            self.api_discovery.discover_from_url,
            "Discover API from a URL"
        )
        
        self.register_tool(
            "discover_api_from_openapi",
            self.api_discovery.discover_from_openapi,
            "Discover API from OpenAPI specification"
        )
        
        self.register_tool(
            "generate_api_client",
            self.api_discovery.generate_client_code,
            "Generate client code for an API"
        )
        
        # Tool Chaining tools
        self.register_tool(
            "create_tool_chain",
            self.create_tool_chain,
            "Create a new tool chain"
        )
        
        self.register_tool(
            "execute_tool_chain",
            self.execute_tool_chain,
            "Execute a tool chain"
        )
        
        # Error Recovery tools
        self.register_tool(
            "execute_with_recovery",
            self.execute_with_recovery,
            "Execute a function with error recovery"
        )
        
        self.register_tool(
            "analyze_errors",
            self.error_recovery.analyze_errors,
            "Analyze errors to identify patterns"
        )
        
        # Security Sandboxing tools
        self.register_tool(
            "execute_in_sandbox",
            self.execute_in_sandbox,
            "Execute code in a secure sandbox"
        )
        
        self.register_tool(
            "create_secure_environment",
            self.security_sandbox.create_secure_environment,
            "Create a secure execution environment"
        )
    
    def register_tool(self, tool_id: str, tool: Callable, description: str = "") -> None:
        """
        Register a tool.
        
        Args:
            tool_id: Unique identifier for the tool
            tool: Function or callable to execute
            description: Description of the tool
        """
        self.tools[tool_id] = {
            'tool': tool,
            'description': description,
            'signature': str(inspect.signature(tool))
        }
        
        # Also register in tool registry
        self.tool_registry.register_tool(tool_id, tool, description)
        
        logger.info(f"Registered tool: {tool_id}")
    
    def discover_api(self, url_or_path: str, api_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover an API from a URL or OpenAPI specification.
        
        Args:
            url_or_path: URL or file path
            api_name: Name to assign to the API (optional)
            
        Returns:
            Dictionary with API information
        """
        # Determine if this is a URL or file path
        if url_or_path.startswith(('http://', 'https://')):
            return self.api_discovery.discover_from_url(url_or_path, api_name)
        elif url_or_path.endswith(('.json', '.yaml', '.yml')):
            return self.api_discovery.discover_from_openapi(url_or_path, api_name)
        else:
            # Try to discover from Python module
            try:
                return self.api_discovery.discover_from_python_module(url_or_path, api_name)
            except:
                # Fall back to URL discovery
                return self.api_discovery.discover_from_url(url_or_path, api_name)
    
    def create_tool_chain(self, name: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a tool chain.
        
        Args:
            name: Name for the tool chain
            steps: List of step configurations
            
        Returns:
            Dictionary with tool chain information
        """
        try:
            # Create tool chain
            chain = ToolChain(name)
            
            # Add steps
            for step_config in steps:
                step_id = step_config['id']
                tool_id = step_config['tool_id']
                
                if tool_id not in self.tools:
                    return {
                        'success': False,
                        'error': f"Tool '{tool_id}' not found"
                    }
                
                tool = self.tools[tool_id]['tool']
                description = step_config.get('description', self.tools[tool_id]['description'])
                required_inputs = step_config.get('required_inputs')
                output_mapping = step_config.get('output_mapping', {})
                
                # Add step to chain
                if step_config.get('parallel', False):
                    # Add to parallel group
                    chain.add_parallel_steps([{
                        'id': step_id,
                        'tool': tool,
                        'description': description,
                        'required_inputs': required_inputs,
                        'output_mapping': output_mapping,
                        'optional': step_config.get('optional', False)
                    }])
                else:
                    # Add sequential step
                    chain.add_step(
                        step_id=step_id,
                        tool=tool,
                        description=description,
                        required_inputs=required_inputs,
                        output_mapping=output_mapping
                    )
            
            # Register the chain
            self.tool_registry.register_chain(chain)
            
            return {
                'success': True,
                'chain_name': name,
                'steps': len(steps)
            }
            
        except Exception as e:
            logger.error(f"Error creating tool chain: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def execute_tool_chain(self, chain_name: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a tool chain.
        
        Args:
            chain_name: Name of the tool chain
            inputs: Input data (optional)
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Execute chain
            result = self.tool_registry.execute_chain(chain_name, inputs)
            
            # Check for errors
            if 'errors' in result and result['errors']:
                logger.warning(f"Tool chain '{chain_name}' completed with errors: {result['errors']}")
                return {
                    'success': True,
                    'chain_name': chain_name,
                    'result': result,
                    'has_errors': True
                }
            
            return {
                'success': True,
                'chain_name': chain_name,
                'result': result,
                'has_errors': False
            }
            
        except Exception as e:
            logger.error(f"Error executing tool chain: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def execute_with_recovery(self, tool_id: str, args: Optional[List[Any]] = None, 
                            kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a tool with error recovery.
        
        Args:
            tool_id: ID of the tool to execute
            args: Positional arguments (optional)
            kwargs: Keyword arguments (optional)
            
        Returns:
            Dictionary with execution results
        """
        if tool_id not in self.tools:
            return {
                'success': False,
                'error': f"Tool '{tool_id}' not found"
            }
        
        tool = self.tools[tool_id]['tool']
        
        try:
            # Execute with recovery
            result = self.error_recovery.execute_with_recovery(
                tool,
                args or (),
                kwargs or {},
                tool_id
            )
            
            return {
                'success': True,
                'tool_id': tool_id,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error executing tool with recovery: {e}")
            
            # Get error recovery suggestions
            error_classifier = ErrorClassifier()
            suggestions = error_classifier.get_error_recovery_suggestions(
                e,
                {
                    'function': tool_id,
                    'args': args or (),
                    'kwargs': kwargs or {}
                }
            )
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'recovery_suggestions': suggestions
            }
    
    def execute_in_sandbox(self, code: str, language: str = 'python',
                         inputs: Optional[Dict[str, Any]] = None,
                         timeout: int = 30) -> Dict[str, Any]:
        """
        Execute code in a secure sandbox.
        
        Args:
            code: Code to execute
            language: Programming language ('python' or 'shell')
            inputs: Input variables (optional)
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        try:
            if language.lower() == 'python':
                return self.security_sandbox.execute_python_code(code, inputs, timeout)
            elif language.lower() == 'shell':
                return self.security_sandbox.execute_shell_command(code, timeout)
            else:
                return {
                    'success': False,
                    'error': f"Unsupported language: {language}"
                }
                
        except Exception as e:
            logger.error(f"Error executing in sandbox: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def discover_tools_from_module(self, module_name: str) -> Dict[str, Any]:
        """
        Discover tools from a Python module.
        
        Args:
            module_name: Name of the Python module
            
        Returns:
            Dictionary with discovered tools
        """
        try:
            # Import module
            module = importlib.import_module(module_name)
            
            # Find all functions
            discovered_tools = {}
            
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if not name.startswith('_'):
                    # Extract function signature
                    sig = inspect.signature(obj)
                    
                    # Extract docstring
                    docstring = obj.__doc__ or ""
                    
                    # Generate tool ID
                    tool_id = f"{module_name}.{name}"
                    
                    # Register tool
                    self.register_tool(tool_id, obj, docstring.split('\n')[0] if docstring else "")
                    
                    # Add to discovered tools
                    discovered_tools[tool_id] = {
                        'name': name,
                        'module': module_name,
                        'signature': str(sig),
                        'docstring': docstring
                    }
            
            logger.info(f"Discovered {len(discovered_tools)} tools from module {module_name}")
            return {
                'success': True,
                'module': module_name,
                'tools': discovered_tools
            }
            
        except Exception as e:
            logger.error(f"Error discovering tools from module: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def discover_local_tools(self) -> Dict[str, Any]:
        """
        Discover tools from installed Python packages.
        
        Returns:
            Dictionary with discovered tools
        """
        discovered_tools = {}
        
        # Get all installed packages
        import pkg_resources
        for package in pkg_resources.working_set:
            package_name = package.project_name
            
            # Skip common system packages
            if package_name.startswith(('_', 'pip', 'setuptools', 'wheel')):
                continue
                
            try:
                # Try to discover tools from the package
                result = self.discover_tools_from_module(package_name)
                if result['success'] and result['tools']:
                    discovered_tools[package_name] = result['tools']
            except:
                # Skip packages that can't be imported
                continue
        
        logger.info(f"Discovered tools from {len(discovered_tools)} packages")
        return {
            'success': True,
            'packages': list(discovered_tools.keys()),
            'tools': discovered_tools
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools.
        
        Returns:
            List of tool information dictionaries
        """
        return [
            {
                'id': tool_id,
                'description': info['description'],
                'signature': info['signature']
            }
            for tool_id, info in self.tools.items()
        ]
    
    def list_tool_chains(self) -> List[Dict[str, Any]]:
        """
        List all registered tool chains.
        
        Returns:
            List of tool chain information dictionaries
        """
        return self.tool_registry.list_chains()
    
    def list_apis(self) -> List[Dict[str, Any]]:
        """
        List all discovered APIs.
        
        Returns:
            List of API information dictionaries
        """
        return self.api_discovery.list_apis()
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        return self.error_recovery.get_error_stats()
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Get the security audit log.
        
        Returns:
            List of audit events
        """
        return self.security_sandbox.get_audit_log()
    
    def create_api_client(self, api_name: str, language: str = 'python') -> Dict[str, Any]:
        """
        Create a client for an API.
        
        Args:
            api_name: Name of the API
            language: Programming language for the client
            
        Returns:
            Dictionary with client code
        """
        try:
            # Generate client code
            client_code = self.api_discovery.generate_client_code(api_name, language)
            
            if client_code.startswith("API '"):
                # Error message
                return {
                    'success': False,
                    'error': client_code
                }
            
            # Create a secure module
            if language.lower() == 'python':
                module_result = self.security_sandbox.create_secure_module(
                    f"{api_name.lower().replace(' ', '_')}_client",
                    client_code
                )
                
                if module_result['success']:
                    return {
                        'success': True,
                        'api_name': api_name,
                        'language': language,
                        'client_code': client_code,
                        'module': module_result['module'],
                        'module_name': module_result['module_name']
                    }
                else:
                    return {
                        'success': False,
                        'error': module_result['error'],
                        'client_code': client_code
                    }
            else:
                return {
                    'success': True,
                    'api_name': api_name,
                    'language': language,
                    'client_code': client_code
                }
                
        except Exception as e:
            logger.error(f"Error creating API client: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def create_tool_from_api(self, api_name: str, endpoint_id: str, 
                           tool_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a tool from an API endpoint.
        
        Args:
            api_name: Name of the API
            endpoint_id: ID of the endpoint
            tool_id: ID for the new tool (optional)
            
        Returns:
            Dictionary with tool information
        """
        try:
            # Get API information
            api_info = self.api_discovery.get_api_info(api_name)
            if not api_info:
                return {
                    'success': False,
                    'error': f"API '{api_name}' not found"
                }
            
            # Get endpoint information
            if endpoint_id not in api_info['endpoints']:
                return {
                    'success': False,
                    'error': f"Endpoint '{endpoint_id}' not found in API '{api_name}'"
                }
            
            endpoint = api_info['endpoints'][endpoint_id]
            
            # Generate tool ID if not provided
            if not tool_id:
                tool_id = f"{api_name.lower().replace(' ', '_')}_{endpoint_id.lower()}"
            
            # Create tool function
            def api_tool(**kwargs):
                # Create API client
                client_result = self.create_api_client(api_name)
                if not client_result['success']:
                    raise ValueError(f"Failed to create API client: {client_result['error']}")
                
                client_module = client_result['module']
                client_class = getattr(client_module, f"{api_name.replace(' ', '')}Client")
                client = client_class()
                
                # Get method name
                method_name = self.api_discovery._endpoint_to_method_name(
                    endpoint['path'], endpoint['method']
                )
                
                # Call method
                method = getattr(client, method_name)
                return method(**kwargs)
            
            # Set function name and docstring
            api_tool.__name__ = tool_id
            api_tool.__doc__ = f"{endpoint['summary']}\n\n{endpoint['description']}"
            
            # Register tool
            self.register_tool(tool_id, api_tool, endpoint['summary'])
            
            return {
                'success': True,
                'tool_id': tool_id,
                'api_name': api_name,
                'endpoint_id': endpoint_id
            }
            
        except Exception as e:
            logger.error(f"Error creating tool from API: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def create_tool_from_code(self, code: str, tool_id: str, 
                            description: str = "",
                            inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a tool from Python code.
        
        Args:
            code: Python code for the tool
            tool_id: ID for the new tool
            description: Description of the tool (optional)
            inputs: Input variables for testing (optional)
            
        Returns:
            Dictionary with tool information
        """
        try:
            # Create a secure module
            module_result = self.security_sandbox.create_secure_module(
                f"tool_{tool_id.lower().replace(' ', '_')}",
                code
            )
            
            if not module_result['success']:
                return module_result
            
            module = module_result['module']
            
            # Find the main function in the module
            main_func = None
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name == 'main' or name == tool_id.lower().replace(' ', '_'):
                    main_func = obj
                    break
            
            if not main_func:
                # Use the first function found
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if not name.startswith('_'):
                        main_func = obj
                        break
            
            if not main_func:
                return {
                    'success': False,
                    'error': "No function found in the code"
                }
            
            # Test the function
            if inputs:
                try:
                    test_result = main_func(**inputs)
                    logger.info(f"Tool test successful: {test_result}")
                except Exception as e:
                    logger.warning(f"Tool test failed: {e}")
            
            # Register tool
            self.register_tool(tool_id, main_func, description or main_func.__doc__ or "")
            
            return {
                'success': True,
                'tool_id': tool_id,
                'function_name': main_func.__name__,
                'signature': str(inspect.signature(main_func))
            }
            
        except Exception as e:
            logger.error(f"Error creating tool from code: {e}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def execute_tool(self, tool_id: str, args: Optional[List[Any]] = None, 
                   kwargs: Optional[Dict[str, Any]] = None,
                   use_recovery: bool = True,
                   use_sandbox: bool = True) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_id: ID of the tool to execute
            args: Positional arguments (optional)
            kwargs: Keyword arguments (optional)
            use_recovery: Whether to use error recovery
            use_sandbox: Whether to use security sandboxing
            
        Returns:
            Dictionary with execution results
        """
        if tool_id not in self.tools:
            return {
                'success': False,
                'error': f"Tool '{tool_id}' not found"
            }
        
        tool = self.tools[tool_id]['tool']
        
        try:
            start_time = time.time()
            
            if use_recovery and use_sandbox:
                # Execute with recovery in sandbox
                result = self.error_recovery.execute_with_recovery(
                    lambda *a, **kw: self.security_sandbox.execute_function_in_sandbox(tool, a, kw),
                    args or (),
                    kwargs or {},
                    tool_id
                )
            elif use_recovery:
                # Execute with recovery
                result = self.error_recovery.execute_with_recovery(
                    tool,
                    args or (),
                    kwargs or {},
                    tool_id
                )
            elif use_sandbox:
                # Execute in sandbox
                result = self.security_sandbox.execute_function_in_sandbox(
                    tool,
                    args or (),
                    kwargs or {}
                )
            else:
                # Execute directly
                result = tool(*(args or ()), **(kwargs or {}))
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'tool_id': tool_id,
                'result': result,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            
            # Get error recovery suggestions if recovery is enabled
            if use_recovery:
                error_classifier = ErrorClassifier()
                suggestions = error_classifier.get_error_recovery_suggestions(
                    e,
                    {
                        'function': tool_id,
                        'args': args or (),
                        'kwargs': kwargs or {}
                    }
                )
                
                return {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'recovery_suggestions': suggestions
                }
            else:
                return {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
