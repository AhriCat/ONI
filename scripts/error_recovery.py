import logging
import time
import traceback
import sys
import os
import json
import tempfile
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import inspect
import importlib
import asyncio
import functools
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ErrorRecovery:
    """
    Error recovery module for robust handling of tool failures.
    
    This module provides:
    1. Automatic retry mechanisms with exponential backoff
    2. Fallback strategies when tools fail
    3. Error classification and appropriate recovery strategies
    4. Logging and monitoring of errors for analysis
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0,
                error_log_path: Optional[str] = None):
        """
        Initialize error recovery module.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (in seconds)
            error_log_path: Path to error log file (optional)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_log_path = error_log_path
        
        # Error handlers by error type
        self.error_handlers = {}
        
        # Fallback functions by tool ID
        self.fallbacks = {}
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'unrecovered_errors': 0,
            'by_error_type': {},
            'by_tool': {}
        }
        
        # Initialize error log
        if error_log_path:
            os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
            self.error_logger = logging.getLogger('oni.error_recovery')
            handler = logging.FileHandler(error_log_path)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.error_logger.addHandler(handler)
        else:
            self.error_logger = logger
    
    def register_error_handler(self, error_type: type, handler: Callable[[Exception, Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Register an error handler for a specific error type.
        
        Args:
            error_type: Type of error to handle
            handler: Function to handle the error
        """
        self.error_handlers[error_type] = handler
        logger.info(f"Registered error handler for {error_type.__name__}")
    
    def register_fallback(self, tool_id: str, fallback: Callable) -> None:
        """
        Register a fallback function for a tool.
        
        Args:
            tool_id: ID of the tool
            fallback: Fallback function to use when the tool fails
        """
        self.fallbacks[tool_id] = fallback
        logger.info(f"Registered fallback for tool '{tool_id}'")
    
    def with_recovery(self, func: Callable, tool_id: Optional[str] = None) -> Callable:
        """
        Decorator to add error recovery to a function.
        
        Args:
            func: Function to decorate
            tool_id: Optional tool ID for tracking
            
        Returns:
            Decorated function with error recovery
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_recovery(func, args, kwargs, tool_id)
        return wrapper
    
    def with_async_recovery(self, func: Callable) -> Callable:
        """
        Decorator to add error recovery to an async function.
        
        Args:
            func: Async function to decorate
            
        Returns:
            Decorated async function with error recovery
        """
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute_with_async_recovery(func, args, kwargs)
        return wrapper
    
    def execute_with_recovery(self, func: Callable, args: Tuple, kwargs: Dict[str, Any],
                             tool_id: Optional[str] = None) -> Any:
        """
        Execute a function with error recovery.
        
        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            tool_id: Optional tool ID for tracking
            
        Returns:
            Function result or fallback result
        """
        func_name = tool_id or func.__name__
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Execute function
                return func(*args, **kwargs)
                
            except Exception as e:
                # Log error
                self._log_error(e, func_name, retry_count)
                
                # Update error statistics
                self._update_error_stats(e, func_name)
                
                # Store last error
                last_error = e
                
                # Try to handle the error
                handled = self._handle_error(e, func_name, args, kwargs)
                if handled:
                    # Error was handled successfully
                    self.error_stats['recovered_errors'] += 1
                    return handled
                
                # Increment retry count
                retry_count += 1
                
                if retry_count <= self.max_retries:
                    # Calculate delay with exponential backoff and jitter
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    jitter = random.uniform(0, 0.1 * delay)
                    total_delay = delay + jitter
                    
                    logger.info(f"Retrying '{func_name}' in {total_delay:.2f}s (attempt {retry_count}/{self.max_retries})")
                    time.sleep(total_delay)
        
        # All retries failed, try fallback
        if tool_id in self.fallbacks:
            logger.info(f"Using fallback for '{func_name}'")
            try:
                return self.fallbacks[tool_id](*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback for '{func_name}' failed: {fallback_error}")
                # Update error statistics
                self._update_error_stats(fallback_error, f"{func_name}_fallback")
        
        # No fallback or fallback failed
        self.error_stats['unrecovered_errors'] += 1
        logger.error(f"All recovery attempts for '{func_name}' failed")
        
        # Re-raise the last error
        if last_error:
            raise last_error
    
    async def execute_with_async_recovery(self, func: Callable, args: Tuple, 
                                        kwargs: Dict[str, Any],
                                        tool_id: Optional[str] = None) -> Any:
        """
        Execute an async function with error recovery.
        
        Args:
            func: Async function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            tool_id: Optional tool ID for tracking
            
        Returns:
            Function result or fallback result
        """
        func_name = tool_id or func.__name__
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Execute function
                return await func(*args, **kwargs)
                
            except Exception as e:
                # Log error
                self._log_error(e, func_name, retry_count)
                
                # Update error statistics
                self._update_error_stats(e, func_name)
                
                # Store last error
                last_error = e
                
                # Try to handle the error
                handled = self._handle_error(e, func_name, args, kwargs)
                if handled:
                    # Error was handled successfully
                    self.error_stats['recovered_errors'] += 1
                    return handled
                
                # Increment retry count
                retry_count += 1
                
                if retry_count <= self.max_retries:
                    # Calculate delay with exponential backoff and jitter
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    jitter = random.uniform(0, 0.1 * delay)
                    total_delay = delay + jitter
                    
                    logger.info(f"Retrying '{func_name}' in {total_delay:.2f}s (attempt {retry_count}/{self.max_retries})")
                    await asyncio.sleep(total_delay)
        
        # All retries failed, try fallback
        if tool_id in self.fallbacks:
            logger.info(f"Using fallback for '{func_name}'")
            try:
                fallback = self.fallbacks[tool_id]
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                else:
                    return fallback(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback for '{func_name}' failed: {fallback_error}")
                # Update error statistics
                self._update_error_stats(fallback_error, f"{func_name}_fallback")
        
        # No fallback or fallback failed
        self.error_stats['unrecovered_errors'] += 1
        logger.error(f"All recovery attempts for '{func_name}' failed")
        
        # Re-raise the last error
        if last_error:
            raise last_error
    
    def _log_error(self, error: Exception, func_name: str, retry_count: int) -> None:
        """Log an error with details."""
        error_type = type(error).__name__
        error_msg = str(error)
        error_traceback = traceback.format_exc()
        
        log_message = (
            f"Error in '{func_name}' (retry {retry_count}/{self.max_retries}): "
            f"{error_type}: {error_msg}"
        )
        
        self.error_logger.error(log_message)
        self.error_logger.debug(error_traceback)
        
        # Log to file with more details
        if self.error_log_path:
            error_details = {
                'timestamp': time.time(),
                'function': func_name,
                'error_type': error_type,
                'error_message': error_msg,
                'traceback': error_traceback,
                'retry_count': retry_count
            }
            
            try:
                with open(self.error_log_path, 'a') as f:
                    f.write(json.dumps(error_details) + '\n')
            except Exception as log_error:
                logger.error(f"Failed to write to error log: {log_error}")
    
    def _update_error_stats(self, error: Exception, func_name: str) -> None:
        """Update error statistics."""
        error_type = type(error).__name__
        
        # Increment total errors
        self.error_stats['total_errors'] += 1
        
        # Update by error type
        if error_type not in self.error_stats['by_error_type']:
            self.error_stats['by_error_type'][error_type] = 0
        self.error_stats['by_error_type'][error_type] += 1
        
        # Update by tool
        if func_name not in self.error_stats['by_tool']:
            self.error_stats['by_tool'][func_name] = 0
        self.error_stats['by_tool'][func_name] += 1
    
    def _handle_error(self, error: Exception, func_name: str, args: Tuple, 
                     kwargs: Dict[str, Any]) -> Optional[Any]:
        """
        Handle an error using registered error handlers.
        
        Args:
            error: The exception that occurred
            func_name: Name of the function that failed
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Result from error handler or None if not handled
        """
        # Check for specific error handler
        error_type = type(error)
        if error_type in self.error_handlers:
            try:
                handler = self.error_handlers[error_type]
                context = {
                    'function': func_name,
                    'args': args,
                    'kwargs': kwargs,
                    'error': error,
                    'error_type': error_type.__name__,
                    'traceback': traceback.format_exc()
                }
                return handler(error, context)
            except Exception as handler_error:
                logger.error(f"Error handler for {error_type.__name__} failed: {handler_error}")
        
        # Check for parent error types
        for registered_type, handler in self.error_handlers.items():
            if issubclass(error_type, registered_type) and registered_type != error_type:
                try:
                    context = {
                        'function': func_name,
                        'args': args,
                        'kwargs': kwargs,
                        'error': error,
                        'error_type': error_type.__name__,
                        'traceback': traceback.format_exc()
                    }
                    return handler(error, context)
                except Exception as handler_error:
                    logger.error(f"Error handler for {registered_type.__name__} failed: {handler_error}")
        
        return None
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        return self.error_stats
    
    def reset_error_stats(self) -> None:
        """Reset error statistics."""
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'unrecovered_errors': 0,
            'by_error_type': {},
            'by_tool': {}
        }
        logger.info("Error statistics reset")
    
    def analyze_errors(self) -> Dict[str, Any]:
        """
        Analyze errors to identify patterns and common issues.
        
        Returns:
            Dictionary with error analysis
        """
        analysis = {
            'total_errors': self.error_stats['total_errors'],
            'recovery_rate': 0,
            'most_common_error_types': [],
            'most_problematic_tools': [],
            'recommendations': []
        }
        
        # Calculate recovery rate
        if self.error_stats['total_errors'] > 0:
            analysis['recovery_rate'] = self.error_stats['recovered_errors'] / self.error_stats['total_errors']
        
        # Get most common error types
        error_types = sorted(
            self.error_stats['by_error_type'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        analysis['most_common_error_types'] = error_types[:5]
        
        # Get most problematic tools
        problematic_tools = sorted(
            self.error_stats['by_tool'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        analysis['most_problematic_tools'] = problematic_tools[:5]
        
        # Generate recommendations
        if analysis['recovery_rate'] < 0.5:
            analysis['recommendations'].append(
                "Low error recovery rate. Consider adding more error handlers or fallbacks."
            )
        
        for error_type, count in analysis['most_common_error_types']:
            if error_type not in self.error_handlers:
                analysis['recommendations'].append(
                    f"Add an error handler for {error_type} errors."
                )
        
        for tool, count in analysis['most_problematic_tools']:
            if tool not in self.fallbacks:
                analysis['recommendations'].append(
                    f"Add a fallback for the '{tool}' tool."
                )
        
        return analysis
    
    def create_error_report(self, output_path: Optional[str] = None) -> str:
        """
        Create a detailed error report.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            Path to the report file or report content
        """
        # Analyze errors
        analysis = self.analyze_errors()
        
        # Create report
        report = f"""
# ONI Error Recovery Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Error Statistics
- Total Errors: {analysis['total_errors']}
- Recovered Errors: {self.error_stats['recovered_errors']}
- Unrecovered Errors: {self.error_stats['unrecovered_errors']}
- Recovery Rate: {analysis['recovery_rate']:.2%}

## Most Common Error Types
"""
        
        for error_type, count in analysis['most_common_error_types']:
            report += f"- {error_type}: {count} occurrences\n"
        
        report += """
## Most Problematic Tools
"""
        
        for tool, count in analysis['most_problematic_tools']:
            report += f"- {tool}: {count} errors\n"
        
        report += """
## Recommendations
"""
        
        for recommendation in analysis['recommendations']:
            report += f"- {recommendation}\n"
        
        # Save report to file if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            return output_path
        
        return report

class ErrorClassifier:
    """
    Classifies errors and suggests appropriate recovery strategies.
    """
    
    def __init__(self):
        """Initialize error classifier."""
        # Error categories and their common error types
        self.error_categories = {
            'network': [
                'ConnectionError',
                'TimeoutError',
                'ConnectionRefusedError',
                'ConnectionResetError',
                'requests.exceptions.RequestException',
                'requests.exceptions.ConnectionError',
                'requests.exceptions.Timeout',
                'urllib.error.URLError',
                'socket.timeout',
                'aiohttp.ClientError'
            ],
            'permission': [
                'PermissionError',
                'AccessDenied',
                'AuthenticationError',
                'AuthorizationError',
                'requests.exceptions.Forbidden',
                'requests.exceptions.Unauthorized'
            ],
            'resource': [
                'FileNotFoundError',
                'NotFoundError',
                'ResourceNotFound',
                'requests.exceptions.NotFound',
                'KeyError',
                'IndexError',
                'AttributeError'
            ],
            'validation': [
                'ValueError',
                'TypeError',
                'AssertionError',
                'ValidationError',
                'json.JSONDecodeError',
                'requests.exceptions.InvalidSchema',
                'requests.exceptions.InvalidURL'
            ],
            'system': [
                'MemoryError',
                'OverflowError',
                'OSError',
                'IOError',
                'SystemError',
                'RuntimeError'
            ],
            'timeout': [
                'TimeoutError',
                'requests.exceptions.Timeout',
                'asyncio.TimeoutError',
                'concurrent.futures.TimeoutError'
            ],
            'rate_limit': [
                'RateLimitError',
                'TooManyRequests',
                'requests.exceptions.TooManyRedirects'
            ]
        }
        
        # Recovery strategies by category
        self.recovery_strategies = {
            'network': [
                'Retry with exponential backoff',
                'Check network connectivity',
                'Verify endpoint URL',
                'Try alternative endpoint'
            ],
            'permission': [
                'Verify credentials',
                'Request necessary permissions',
                'Use alternative authentication method',
                'Escalate to user for authentication'
            ],
            'resource': [
                'Check if resource exists',
                'Create resource if missing',
                'Use alternative resource',
                'Provide default value'
            ],
            'validation': [
                'Validate input data',
                'Convert data to expected format',
                'Provide default values',
                'Use schema validation'
            ],
            'system': [
                'Reduce resource usage',
                'Free up memory',
                'Restart process',
                'Use alternative implementation'
            ],
            'timeout': [
                'Increase timeout value',
                'Retry with longer timeout',
                'Break task into smaller chunks',
                'Use asynchronous processing'
            ],
            'rate_limit': [
                'Implement rate limiting',
                'Use exponential backoff',
                'Reduce request frequency',
                'Use bulk operations'
            ]
        }
    
    def classify_error(self, error: Exception) -> Dict[str, Any]:
        """
        Classify an error and suggest recovery strategies.
        
        Args:
            error: The exception to classify
            
        Returns:
            Dictionary with error classification and recovery strategies
        """
        error_type = type(error).__name__
        error_module = type(error).__module__
        error_full_name = f"{error_module}.{error_type}" if error_module != "builtins" else error_type
        
        # Find matching categories
        categories = []
        for category, error_types in self.error_categories.items():
            for et in error_types:
                if et in error_type or et in error_full_name:
                    categories.append(category)
                    break
        
        # If no categories match, use the error's base classes
        if not categories:
            for base in type(error).__mro__[1:]:  # Skip the error type itself
                if base == Exception or base == BaseException:
                    break
                    
                base_name = base.__name__
                for category, error_types in self.error_categories.items():
                    if base_name in error_types:
                        categories.append(category)
                        break
        
        # If still no categories match, default to 'unknown'
        if not categories:
            categories = ['unknown']
        
        # Get recovery strategies
        strategies = []
        for category in categories:
            strategies.extend(self.recovery_strategies.get(category, []))
        
        # Remove duplicates while preserving order
        unique_strategies = []
        for strategy in strategies:
            if strategy not in unique_strategies:
                unique_strategies.append(strategy)
        
        return {
            'error_type': error_type,
            'error_message': str(error),
            'categories': categories,
            'recovery_strategies': unique_strategies
        }
    
    def suggest_recovery_code(self, error: Exception, context: Dict[str, Any]) -> str:
        """
        Suggest recovery code for an error.
        
        Args:
            error: The exception to recover from
            context: Error context
            
        Returns:
            Suggested recovery code as a string
        """
        classification = self.classify_error(error)
        error_type = classification['error_type']
        categories = classification['categories']
        
        # Get function name and arguments
        func_name = context.get('function', 'function')
        args = context.get('args', ())
        kwargs = context.get('kwargs', {})
        
        # Generate recovery code based on error category
        if 'network' in categories:
            return self._generate_network_recovery(func_name, args, kwargs)
        elif 'permission' in categories:
            return self._generate_permission_recovery(func_name, args, kwargs)
        elif 'resource' in categories:
            return self._generate_resource_recovery(func_name, error, args, kwargs)
        elif 'validation' in categories:
            return self._generate_validation_recovery(func_name, error, args, kwargs)
        elif 'timeout' in categories:
            return self._generate_timeout_recovery(func_name, args, kwargs)
        elif 'rate_limit' in categories:
            return self._generate_rate_limit_recovery(func_name, args, kwargs)
        else:
            return self._generate_generic_recovery(func_name, error_type, args, kwargs)
    
    def _generate_network_recovery(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate recovery code for network errors."""
        return f"""
# Recovery for network error in {func_name}
import time
import random

def retry_with_backoff(func, max_retries=5, initial_delay=1, max_delay=60):
    retries = 0
    while retries < max_retries:
        try:
            return func()
        except (ConnectionError, TimeoutError) as e:
            retries += 1
            if retries >= max_retries:
                raise
            delay = min(initial_delay * (2 ** (retries - 1)) + random.uniform(0, 1), max_delay)
            print(f"Network error, retrying in {{delay:.2f}}s ({{retries}}/{{max_retries}})")
            time.sleep(delay)

# Wrap the function call with retry logic
result = retry_with_backoff(lambda: {func_name}(*{args}, **{kwargs}))
"""
    
    def _generate_permission_recovery(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate recovery code for permission errors."""
        return f"""
# Recovery for permission error in {func_name}
def handle_permission_error():
    # Try to refresh credentials
    try:
        # Check for environment variables or config files
        import os
        new_token = os.environ.get('API_TOKEN') or os.environ.get('ACCESS_TOKEN')
        
        if new_token:
            # Update credentials in kwargs
            kwargs = {kwargs}
            if 'headers' in kwargs:
                kwargs['headers']['Authorization'] = f"Bearer {{new_token}}"
            else:
                kwargs['headers'] = {{'Authorization': f"Bearer {{new_token}}"}}
                
            # Retry with new credentials
            return {func_name}(*{args}, **kwargs)
        else:
            # Prompt for credentials
            print("Permission error: Please provide valid credentials")
            new_token = input("Enter API token: ")
            
            # Update credentials in kwargs
            kwargs = {kwargs}
            if 'headers' in kwargs:
                kwargs['headers']['Authorization'] = f"Bearer {{new_token}}"
            else:
                kwargs['headers'] = {{'Authorization': f"Bearer {{new_token}}"}}
                
            # Retry with new credentials
            return {func_name}(*{args}, **kwargs)
    except Exception as e:
        print(f"Failed to refresh credentials: {{e}}")
        raise

# Try to handle the permission error
result = handle_permission_error()
"""
    
    def _generate_resource_recovery(self, func_name: str, error: Exception, 
                                  args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate recovery code for resource errors."""
        error_msg = str(error)
        resource_name = ""
        
        # Try to extract resource name from error message
        if "file" in error_msg.lower() or "path" in error_msg.lower():
            # Look for quoted strings or paths
            import re
            matches = re.findall(r"['\"](.*?)['\"]", error_msg)
            if matches:
                resource_name = matches[0]
            else:
                # Look for words that might be filenames
                words = error_msg.split()
                for word in words:
                    if '.' in word or '/' in word or '\\' in word:
                        resource_name = word
                        break
        
        return f"""
# Recovery for resource error in {func_name}
def handle_resource_error():
    try:
        # Check if resource exists
        import os
        resource_name = "{resource_name}"
        
        if resource_name and not os.path.exists(resource_name):
            # Create directory if it's a path
            directory = os.path.dirname(resource_name)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {{directory}}")
            
            # Create empty file if it's a file
            if '.' in os.path.basename(resource_name):
                with open(resource_name, 'w') as f:
                    pass
                print(f"Created empty file: {{resource_name}}")
                
        # Try alternative resource
        alternatives = [
            resource_name,
            os.path.join(os.getcwd(), os.path.basename(resource_name)) if resource_name else None,
            "default_resource.txt"
        ]
        
        for alt in alternatives:
            if alt and os.path.exists(alt):
                # Update args or kwargs with alternative resource
                args_list = list({args})
                kwargs_dict = {kwargs}
                
                # Try to identify which argument contains the resource
                for i, arg in enumerate(args_list):
                    if isinstance(arg, str) and (resource_name in arg or arg in resource_name):
                        args_list[i] = alt
                        break
                else:
                    # Check kwargs
                    for key, value in kwargs_dict.items():
                        if isinstance(value, str) and (resource_name in value or value in resource_name):
                            kwargs_dict[key] = alt
                            break
                
                # Retry with alternative resource
                return {func_name}(*args_list, **kwargs_dict)
        
        # If no alternatives work, raise the original error
        raise ValueError(f"Resource not found: {{resource_name}}")
    except Exception as e:
        print(f"Failed to handle resource error: {{e}}")
        raise

# Try to handle the resource error
result = handle_resource_error()
"""
    
    def _generate_validation_recovery(self, func_name: str, error: Exception, 
                                    args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate recovery code for validation errors."""
        error_msg = str(error)
        
        return f"""
# Recovery for validation error in {func_name}
def handle_validation_error():
    try:
        # Parse error message for clues
        error_msg = "{error_msg}"
        
        # Check for common validation issues
        if "JSON" in error_msg or "json" in error_msg:
            # Fix JSON data
            import json
            kwargs_dict = {kwargs}
            
            # Try to find and fix JSON data
            for key, value in kwargs_dict.items():
                if isinstance(value, str) and (value.startswith('{{') or value.startswith('[')):
                    try:
                        # Try to parse and re-serialize to fix formatting
                        fixed_json = json.dumps(json.loads(value))
                        kwargs_dict[key] = fixed_json
                    except:
                        pass
            
            # Retry with fixed data
            return {func_name}(*{args}, **kwargs_dict)
            
        elif "type" in error_msg.lower():
            # Type conversion issue
            kwargs_dict = {kwargs}
            
            # Try to convert values to appropriate types
            for key, value in kwargs_dict.items():
                if isinstance(value, str):
                    # Try to convert string to int or float
                    try:
                        if value.isdigit():
                            kwargs_dict[key] = int(value)
                        elif value.replace('.', '', 1).isdigit():
                            kwargs_dict[key] = float(value)
                    except:
                        pass
            
            # Retry with converted data
            return {func_name}(*{args}, **kwargs_dict)
            
        else:
            # Generic validation fix - provide default values
            kwargs_dict = {kwargs}
            
            # Set some common default values
            defaults = {{
                'limit': 10,
                'offset': 0,
                'page': 1,
                'per_page': 10,
                'sort': 'id',
                'order': 'asc',
                'format': 'json'
            }}
            
            # Apply defaults for missing values
            for key, value in defaults.items():
                if key in kwargs_dict and kwargs_dict[key] is None:
                    kwargs_dict[key] = value
            
            # Retry with default values
            return {func_name}(*{args}, **kwargs_dict)
    except Exception as e:
        print(f"Failed to handle validation error: {{e}}")
        raise

# Try to handle the validation error
result = handle_validation_error()
"""
    
    def _generate_timeout_recovery(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate recovery code for timeout errors."""
        return f"""
# Recovery for timeout error in {func_name}
def handle_timeout_error():
    try:
        # Increase timeout value
        kwargs_dict = {kwargs}
        
        # Look for timeout parameter
        timeout_keys = ['timeout', 'request_timeout', 'connection_timeout', 'read_timeout']
        for key in timeout_keys:
            if key in kwargs_dict:
                # Double the timeout
                kwargs_dict[key] = kwargs_dict[key] * 2 if isinstance(kwargs_dict[key], (int, float)) else 30
                break
        else:
            # Add timeout if not present
            kwargs_dict['timeout'] = 30
        
        # Retry with increased timeout
        return {func_name}(*{args}, **kwargs_dict)
    except Exception as e:
        print(f"Failed to handle timeout error: {{e}}")
        raise

# Try to handle the timeout error
result = handle_timeout_error()
"""
    
    def _generate_rate_limit_recovery(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate recovery code for rate limit errors."""
        return f"""
# Recovery for rate limit error in {func_name}
import time
import random

def handle_rate_limit_error():
    try:
        # Wait for rate limit to reset
        wait_time = random.uniform(5, 15)  # Random wait between 5-15 seconds
        print(f"Rate limit exceeded, waiting {{wait_time:.2f}}s before retrying")
        time.sleep(wait_time)
        
        # Retry with the same parameters
        return {func_name}(*{args}, **{kwargs})
    except Exception as e:
        print(f"Failed to handle rate limit error: {{e}}")
        raise

# Try to handle the rate limit error
result = handle_rate_limit_error()
"""
    
    def _generate_generic_recovery(self, func_name: str, error_type: str, 
                                 args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate generic recovery code."""
        return f"""
# Generic recovery for {error_type} in {func_name}
def handle_generic_error():
    try:
        # Simplify the request
        kwargs_dict = {kwargs}
        
        # Remove optional parameters
        optional_params = []
        for key in list(kwargs_dict.keys()):
            if key not in ['id', 'name', 'key', 'path', 'url']:  # Keep essential parameters
                optional_params.append(key)
        
        # Try with only essential parameters
        essential_kwargs = {{k: v for k, v in kwargs_dict.items() if k not in optional_params}}
        return {func_name}(*{args}, **essential_kwargs)
    except Exception as e:
        print(f"Failed to handle generic error: {{e}}")
        raise

# Try to handle the error
result = handle_generic_error()
"""
    
    def get_error_recovery_suggestions(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get error recovery suggestions.
        
        Args:
            error: The exception to recover from
            context: Error context
            
        Returns:
            Dictionary with recovery suggestions
        """
        # Classify error
        classification = self.classify_error(error)
        
        # Generate recovery code
        recovery_code = self.suggest_recovery_code(error, context)
        
        return {
            'error_type': classification['error_type'],
            'error_message': classification['error_message'],
            'categories': classification['categories'],
            'recovery_strategies': classification['recovery_strategies'],
            'recovery_code': recovery_code
        }
