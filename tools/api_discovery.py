import requests
import json
import logging
import re
import time
from typing import Dict, List, Optional, Any, Union, Callable
import inspect
import importlib
import pkgutil
import sys
import os
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APIDiscovery:
    """
    Automatic API learning and integration module.
    
    This module can:
    1. Discover and learn APIs from documentation or examples
    2. Generate client code for API integration
    3. Test and validate API endpoints
    4. Maintain a registry of known APIs
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize API discovery module.
        
        Args:
            cache_dir: Directory to cache API specifications and client code
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".oni", "api_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Registry of known APIs
        self.api_registry = {}
        
        # Load cached API registry if available
        self._load_registry()
    
    def _load_registry(self):
        """Load API registry from cache."""
        registry_path = os.path.join(self.cache_dir, "api_registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    self.api_registry = json.load(f)
                logger.info(f"Loaded {len(self.api_registry)} APIs from registry")
            except Exception as e:
                logger.error(f"Failed to load API registry: {e}")
    
    def _save_registry(self):
        """Save API registry to cache."""
        registry_path = os.path.join(self.cache_dir, "api_registry.json")
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.api_registry, f, indent=2)
            logger.info(f"Saved {len(self.api_registry)} APIs to registry")
        except Exception as e:
            logger.error(f"Failed to save API registry: {e}")
    
    def discover_from_openapi(self, url_or_path: str, api_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover API from OpenAPI specification.
        
        Args:
            url_or_path: URL or file path to OpenAPI specification
            api_name: Name to assign to the API (optional)
            
        Returns:
            Dictionary with API information
        """
        try:
            # Load OpenAPI spec
            if url_or_path.startswith(('http://', 'https://')):
                response = requests.get(url_or_path)
                response.raise_for_status()
                spec = response.json()
            else:
                with open(url_or_path, 'r') as f:
                    spec = json.load(f)
            
            # Extract API information
            if 'info' in spec:
                title = spec['info'].get('title', 'Unknown API')
                version = spec['info'].get('version', '1.0.0')
                description = spec['info'].get('description', '')
            else:
                title = api_name or 'Unknown API'
                version = '1.0.0'
                description = ''
            
            # Extract endpoints
            endpoints = {}
            if 'paths' in spec:
                for path, path_item in spec['paths'].items():
                    for method, operation in path_item.items():
                        if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                            endpoint_id = f"{method.upper()}_{path.replace('/', '_')}"
                            endpoints[endpoint_id] = {
                                'path': path,
                                'method': method.upper(),
                                'summary': operation.get('summary', ''),
                                'description': operation.get('description', ''),
                                'parameters': operation.get('parameters', []),
                                'requestBody': operation.get('requestBody', {}),
                                'responses': operation.get('responses', {})
                            }
            
            # Create API entry
            api_info = {
                'name': api_name or title,
                'title': title,
                'version': version,
                'description': description,
                'spec_type': 'openapi',
                'spec_url': url_or_path,
                'endpoints': endpoints,
                'base_url': self._extract_base_url(spec),
                'auth_type': self._detect_auth_type(spec),
                'timestamp': time.time()
            }
            
            # Add to registry
            self.api_registry[api_info['name']] = api_info
            self._save_registry()
            
            logger.info(f"Discovered API: {api_info['name']} with {len(endpoints)} endpoints")
            return api_info
            
        except Exception as e:
            logger.error(f"Failed to discover API from OpenAPI spec: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def discover_from_url(self, url: str, api_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover API by analyzing a URL and making test requests.
        
        Args:
            url: Base URL of the API
            api_name: Name to assign to the API (optional)
            
        Returns:
            Dictionary with API information
        """
        try:
            # Parse URL
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Try to find API documentation
            docs_endpoints = [
                '/api-docs',
                '/swagger',
                '/openapi.json',
                '/docs',
                '/redoc',
                '/api/v1',
                '/api'
            ]
            
            # Check for OpenAPI/Swagger documentation
            for endpoint in docs_endpoints:
                try:
                    docs_url = f"{base_url}{endpoint}"
                    response = requests.get(docs_url, timeout=5)
                    if response.status_code == 200:
                        content_type = response.headers.get('Content-Type', '')
                        if 'application/json' in content_type:
                            try:
                                spec = response.json()
                                if 'swagger' in spec or 'openapi' in spec:
                                    # Found OpenAPI spec
                                    return self.discover_from_openapi(docs_url, api_name)
                            except:
                                pass
                except:
                    continue
            
            # If no documentation found, make a test request to the base URL
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
            except:
                # Try with a trailing slash
                if not url.endswith('/'):
                    url = url + '/'
                    response = requests.get(url, timeout=5)
            
            # Create basic API entry
            api_info = {
                'name': api_name or self._generate_api_name(url),
                'title': api_name or self._generate_api_name(url),
                'version': '1.0.0',
                'description': f"API discovered from {url}",
                'spec_type': 'discovered',
                'spec_url': url,
                'endpoints': {},
                'base_url': base_url,
                'auth_type': 'unknown',
                'timestamp': time.time()
            }
            
            # Try to discover endpoints
            common_endpoints = [
                '/api/v1/users',
                '/api/users',
                '/users',
                '/api/v1/items',
                '/api/items',
                '/items',
                '/api/v1/data',
                '/api/data',
                '/data'
            ]
            
            for endpoint in common_endpoints:
                try:
                    endpoint_url = f"{base_url}{endpoint}"
                    response = requests.get(endpoint_url, timeout=5)
                    if response.status_code < 500:  # Accept 2xx, 3xx, 4xx but not 5xx
                        endpoint_id = f"GET_{endpoint.replace('/', '_')}"
                        api_info['endpoints'][endpoint_id] = {
                            'path': endpoint,
                            'method': 'GET',
                            'summary': f"Discovered endpoint: {endpoint}",
                            'description': '',
                            'parameters': [],
                            'responses': {
                                str(response.status_code): {
                                    'description': 'Discovered response'
                                }
                            }
                        }
                except:
                    continue
            
            # Add to registry
            self.api_registry[api_info['name']] = api_info
            self._save_registry()
            
            logger.info(f"Discovered API: {api_info['name']} with {len(api_info['endpoints'])} endpoints")
            return api_info
            
        except Exception as e:
            logger.error(f"Failed to discover API from URL: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def discover_from_python_module(self, module_name: str, api_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover API by analyzing a Python module.
        
        Args:
            module_name: Name of the Python module
            api_name: Name to assign to the API (optional)
            
        Returns:
            Dictionary with API information
        """
        try:
            # Import module
            module = importlib.import_module(module_name)
            
            # Extract API information
            title = api_name or module_name
            version = getattr(module, '__version__', '1.0.0')
            description = getattr(module, '__doc__', '').strip()
            
            # Discover functions and classes
            endpoints = {}
            
            # Find all functions
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if not name.startswith('_'):
                    # Extract function signature
                    sig = inspect.signature(obj)
                    
                    # Extract parameters
                    parameters = []
                    for param_name, param in sig.parameters.items():
                        if param_name != 'self':
                            parameters.append({
                                'name': param_name,
                                'required': param.default == inspect.Parameter.empty,
                                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any'
                            })
                    
                    endpoints[name] = {
                        'path': f"/{name}",
                        'method': 'FUNCTION',
                        'summary': obj.__doc__.split('\n')[0] if obj.__doc__ else '',
                        'description': obj.__doc__ if obj.__doc__ else '',
                        'parameters': parameters,
                        'function': obj
                    }
            
            # Find all classes
            for class_name, obj in inspect.getmembers(module, inspect.isclass):
                if not class_name.startswith('_') and obj.__module__ == module_name:
                    # Extract class methods
                    for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                        if not method_name.startswith('_'):
                            # Extract method signature
                            sig = inspect.signature(method)
                            
                            # Extract parameters
                            parameters = []
                            for param_name, param in sig.parameters.items():
                                if param_name != 'self':
                                    parameters.append({
                                        'name': param_name,
                                        'required': param.default == inspect.Parameter.empty,
                                        'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any'
                                    })
                            
                            endpoint_id = f"{class_name}_{method_name}"
                            endpoints[endpoint_id] = {
                                'path': f"/{class_name}/{method_name}",
                                'method': 'METHOD',
                                'summary': method.__doc__.split('\n')[0] if method.__doc__ else '',
                                'description': method.__doc__ if method.__doc__ else '',
                                'parameters': parameters,
                                'class': obj,
                                'method': method
                            }
            
            # Create API entry
            api_info = {
                'name': api_name or title,
                'title': title,
                'version': version,
                'description': description,
                'spec_type': 'python_module',
                'spec_url': module_name,
                'endpoints': endpoints,
                'base_url': None,
                'auth_type': None,
                'timestamp': time.time()
            }
            
            # Add to registry
            self.api_registry[api_info['name']] = api_info
            self._save_registry()
            
            logger.info(f"Discovered API from Python module: {api_info['name']} with {len(endpoints)} endpoints")
            return api_info
            
        except Exception as e:
            logger.error(f"Failed to discover API from Python module: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def discover_local_apis(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover APIs from installed Python packages.
        
        Returns:
            Dictionary mapping API names to API information
        """
        discovered_apis = {}
        
        # Get all installed packages
        for module_info in pkgutil.iter_modules():
            module_name = module_info.name
            
            # Skip common system modules
            if module_name.startswith(('_', 'test_', 'setup')):
                continue
                
            try:
                # Try to import the module
                module = importlib.import_module(module_name)
                
                # Check if it has an API-like interface
                has_api = False
                
                # Check for common API indicators
                if (hasattr(module, 'Client') or 
                    hasattr(module, 'API') or 
                    hasattr(module, 'client') or 
                    hasattr(module, 'api') or
                    'api' in module_name.lower()):
                    has_api = True
                
                if has_api:
                    # Discover API
                    api_info = self.discover_from_python_module(module_name)
                    if 'error' not in api_info:
                        discovered_apis[api_info['name']] = api_info
            except:
                # Skip modules that can't be imported
                continue
        
        logger.info(f"Discovered {len(discovered_apis)} local APIs")
        return discovered_apis
    
    def generate_client_code(self, api_name: str, language: str = 'python') -> str:
        """
        Generate client code for an API.
        
        Args:
            api_name: Name of the API
            language: Programming language for the client code
            
        Returns:
            Generated client code as a string
        """
        if api_name not in self.api_registry:
            return f"API '{api_name}' not found in registry"
        
        api_info = self.api_registry[api_name]
        
        if language.lower() == 'python':
            return self._generate_python_client(api_info)
        elif language.lower() == 'javascript':
            return self._generate_javascript_client(api_info)
        else:
            return f"Unsupported language: {language}"
    
    def _generate_python_client(self, api_info: Dict[str, Any]) -> str:
        """Generate Python client code for an API."""
        code = f"""
# Generated Python client for {api_info['name']}
# API Version: {api_info['version']}
# {api_info['description']}

import requests
import json
from typing import Dict, List, Optional, Any, Union

class {api_info['name'].replace(' ', '')}Client:
    \"\"\"
    Python client for {api_info['name']} API.
    \"\"\"
    
    def __init__(self, base_url: str = "{api_info['base_url'] or ''}", api_key: Optional[str] = None):
        \"\"\"
        Initialize the API client.
        
        Args:
            base_url: Base URL for API requests
            api_key: API key for authentication (if required)
        \"\"\"
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set up authentication if API key is provided
        if api_key:
            self.session.headers.update({{"Authorization": f"Bearer {{api_key}}"}})
    
"""
        
        # Generate methods for each endpoint
        for endpoint_id, endpoint in api_info['endpoints'].items():
            if endpoint['method'] == 'FUNCTION' or endpoint['method'] == 'METHOD':
                continue  # Skip Python module functions/methods
                
            # Generate method name
            method_name = self._endpoint_to_method_name(endpoint['path'], endpoint['method'])
            
            # Generate method parameters
            params = []
            for param in endpoint.get('parameters', []):
                param_name = param['name']
                param_type = 'Any'
                required = param.get('required', False)
                
                if required:
                    params.append(f"{param_name}: {param_type}")
                else:
                    params.append(f"{param_name}: Optional[{param_type}] = None")
            
            # Add request body parameter if needed
            if 'requestBody' in endpoint and endpoint['requestBody']:
                params.append("data: Optional[Dict[str, Any]] = None")
            
            # Generate method docstring
            docstring = f"""
        \"\"\"
        {endpoint['summary']}
        
        {endpoint['description']}
        
        Args:
"""
            
            for param in endpoint.get('parameters', []):
                docstring += f"            {param['name']}: {param.get('description', '')}\n"
            
            if 'requestBody' in endpoint and endpoint['requestBody']:
                docstring += f"            data: Request body data\n"
                
            docstring += f"""
        Returns:
            API response
        \"\"\"
"""
            
            # Generate method body
            method_body = f"""
        url = f"{{self.base_url}}{endpoint['path']}"
        
"""
            
            # Handle different HTTP methods
            if endpoint['method'] == 'GET':
                method_body += """
        # Prepare query parameters
        params = {}
"""
                for param in endpoint.get('parameters', []):
                    if param.get('in', '') == 'query':
                        method_body += f"""
        if {param['name']} is not None:
            params['{param['name']}'] = {param['name']}
"""
                
                method_body += """
        # Make request
        response = self.session.get(url, params=params)
"""
            elif endpoint['method'] == 'POST':
                method_body += """
        # Make request
        response = self.session.post(url, json=data)
"""
            elif endpoint['method'] == 'PUT':
                method_body += """
        # Make request
        response = self.session.put(url, json=data)
"""
            elif endpoint['method'] == 'DELETE':
                method_body += """
        # Make request
        response = self.session.delete(url)
"""
            elif endpoint['method'] == 'PATCH':
                method_body += """
        # Make request
        response = self.session.patch(url, json=data)
"""
            
            method_body += """
        # Check for errors
        response.raise_for_status()
        
        # Parse response
        try:
            return response.json()
        except:
            return response.text
"""
            
            # Combine method components
            method_code = f"""
    def {method_name}({', '.join(['self'] + params)}):
{docstring}{method_body}
"""
            
            code += method_code
        
        return code
    
    def _generate_javascript_client(self, api_info: Dict[str, Any]) -> str:
        """Generate JavaScript client code for an API."""
        code = f"""
// Generated JavaScript client for {api_info['name']}
// API Version: {api_info['version']}
// {api_info['description']}

class {api_info['name'].replace(' ', '')}Client {{
  /**
   * Initialize the API client.
   * @param {{string}} baseUrl - Base URL for API requests
   * @param {{string}} apiKey - API key for authentication (if required)
   */
  constructor(baseUrl = "{api_info['base_url'] or ''}", apiKey = null) {{
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
    this.headers = {{
      'Content-Type': 'application/json'
    }};
    
    // Set up authentication if API key is provided
    if (apiKey) {{
      this.headers['Authorization'] = `Bearer ${{apiKey}}`;
    }}
  }}
  
"""
        
        # Generate methods for each endpoint
        for endpoint_id, endpoint in api_info['endpoints'].items():
            if endpoint['method'] == 'FUNCTION' or endpoint['method'] == 'METHOD':
                continue  # Skip Python module functions/methods
                
            # Generate method name
            method_name = self._endpoint_to_method_name(endpoint['path'], endpoint['method'])
            
            # Generate method parameters
            params = []
            query_params = []
            path_params = []
            
            for param in endpoint.get('parameters', []):
                param_name = param['name']
                required = param.get('required', False)
                
                if required:
                    params.append(param_name)
                else:
                    params.append(f"{param_name} = null")
                
                if param.get('in', '') == 'query':
                    query_params.append(param_name)
                elif param.get('in', '') == 'path':
                    path_params.append(param_name)
            
            # Add request body parameter if needed
            if 'requestBody' in endpoint and endpoint['requestBody']:
                params.append("data = null")
            
            # Generate method JSDoc
            jsdoc = f"""
  /**
   * {endpoint['summary']}
   *
   * {endpoint['description']}
   *
"""
            
            for param in endpoint.get('parameters', []):
                jsdoc += f"   * @param {{{param.get('type', 'any')}}} {param['name']} - {param.get('description', '')}\n"
            
            if 'requestBody' in endpoint and endpoint['requestBody']:
                jsdoc += f"   * @param {{object}} data - Request body data\n"
                
            jsdoc += f"""
   * @returns {{Promise<any>}} API response
   */
"""
            
            # Generate method body
            method_body = f"""
    let url = `${{this.baseUrl}}{endpoint['path']}`;
    
"""
            
            # Handle path parameters
            for param in path_params:
                method_body += f"""
    // Replace path parameter
    url = url.replace('{{{param}}}', {param});
"""
            
            # Handle different HTTP methods
            if endpoint['method'] == 'GET':
                method_body += """
    // Prepare query parameters
    const queryParams = new URLSearchParams();
"""
                for param in query_params:
                    method_body += f"""
    if ({param} !== null) {{
      queryParams.append('{param}', {param});
    }}
"""
                
                method_body += """
    // Add query parameters to URL
    const queryString = queryParams.toString();
    if (queryString) {
      url += `?${queryString}`;
    }
    
    // Make request
    const response = await fetch(url, {
      method: 'GET',
      headers: this.headers
    });
"""
            elif endpoint['method'] == 'POST':
                method_body += """
    // Make request
    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: data ? JSON.stringify(data) : undefined
    });
"""
            elif endpoint['method'] == 'PUT':
                method_body += """
    // Make request
    const response = await fetch(url, {
      method: 'PUT',
      headers: this.headers,
      body: data ? JSON.stringify(data) : undefined
    });
"""
            elif endpoint['method'] == 'DELETE':
                method_body += """
    // Make request
    const response = await fetch(url, {
      method: 'DELETE',
      headers: this.headers
    });
"""
            elif endpoint['method'] == 'PATCH':
                method_body += """
    // Make request
    const response = await fetch(url, {
      method: 'PATCH',
      headers: this.headers,
      body: data ? JSON.stringify(data) : undefined
    });
"""
            
            method_body += """
    // Check for errors
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }
    
    // Parse response
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return await response.json();
    } else {
      return await response.text();
    }
"""
            
            # Combine method components
            method_code = f"""
{jsdoc}
  async {method_name}({', '.join(params)}) {{{method_body}
  }}
"""
            
            code += method_code
        
        code += """
}

// Export the client
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Client };
}
"""
        
        return code
    
    def test_api_endpoint(self, api_name: str, endpoint_id: str, 
                         params: Optional[Dict[str, Any]] = None,
                         data: Optional[Dict[str, Any]] = None,
                         headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Test an API endpoint.
        
        Args:
            api_name: Name of the API
            endpoint_id: ID of the endpoint to test
            params: Query parameters (optional)
            data: Request body data (optional)
            headers: Request headers (optional)
            
        Returns:
            Dictionary with test results
        """
        if api_name not in self.api_registry:
            return {
                'success': False,
                'error': f"API '{api_name}' not found in registry"
            }
        
        api_info = self.api_registry[api_name]
        
        if endpoint_id not in api_info['endpoints']:
            return {
                'success': False,
                'error': f"Endpoint '{endpoint_id}' not found in API '{api_name}'"
            }
        
        endpoint = api_info['endpoints'][endpoint_id]
        
        # Handle Python module functions/methods
        if endpoint['method'] == 'FUNCTION':
            try:
                function = endpoint['function']
                result = function(**(params or {}))
                return {
                    'success': True,
                    'result': result
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        if endpoint['method'] == 'METHOD':
            try:
                cls = endpoint['class']
                method = endpoint['method']
                obj = cls()
                result = getattr(obj, method.__name__)(**(params or {}))
                return {
                    'success': True,
                    'result': result
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        # Handle HTTP endpoints
        try:
            # Prepare URL
            base_url = api_info['base_url']
            path = endpoint['path']
            url = f"{base_url}{path}"
            
            # Prepare headers
            request_headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            if headers:
                request_headers.update(headers)
            
            # Make request
            method = endpoint['method'].lower()
            if method == 'get':
                response = requests.get(url, params=params, headers=request_headers)
            elif method == 'post':
                response = requests.post(url, params=params, json=data, headers=request_headers)
            elif method == 'put':
                response = requests.put(url, params=params, json=data, headers=request_headers)
            elif method == 'delete':
                response = requests.delete(url, params=params, headers=request_headers)
            elif method == 'patch':
                response = requests.patch(url, params=params, json=data, headers=request_headers)
            else:
                return {
                    'success': False,
                    'error': f"Unsupported HTTP method: {method}"
                }
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            return {
                'success': response.status_code < 400,
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'data': response_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_base_url(self, spec: Dict[str, Any]) -> Optional[str]:
        """Extract base URL from OpenAPI specification."""
        if 'servers' in spec and spec['servers']:
            return spec['servers'][0].get('url')
        return None
    
    def _detect_auth_type(self, spec: Dict[str, Any]) -> str:
        """Detect authentication type from OpenAPI specification."""
        if 'components' in spec and 'securitySchemes' in spec['components']:
            schemes = spec['components']['securitySchemes']
            
            if 'oauth2' in schemes:
                return 'oauth2'
            elif 'bearerAuth' in schemes:
                return 'bearer'
            elif 'apiKey' in schemes:
                return 'api_key'
            elif 'basicAuth' in schemes:
                return 'basic'
        
        return 'unknown'
    
    def _generate_api_name(self, url: str) -> str:
        """Generate API name from URL."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Remove .com, .org, etc.
        domain = domain.split('.')[0]
        
        # Capitalize
        name = domain.capitalize()
        
        # Add API suffix
        if not name.endswith('API'):
            name += ' API'
        
        return name
    
    def _endpoint_to_method_name(self, path: str, method: str) -> str:
        """Convert endpoint path and method to a method name."""
        # Remove leading and trailing slashes
        path = path.strip('/')
        
        # Replace path parameters with placeholder
        path = re.sub(r'{[^}]+}', 'X', path)
        
        # Split path into components
        components = path.split('/')
        
        # Convert to camelCase
        method_name = method.lower()
        for component in components:
            if component:
                method_name += component[0].upper() + component[1:]
        
        return method_name
    
    def get_api_info(self, api_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered API.
        
        Args:
            api_name: Name of the API
            
        Returns:
            API information or None if not found
        """
        return self.api_registry.get(api_name)
    
    def list_apis(self) -> List[Dict[str, Any]]:
        """
        List all registered APIs.
        
        Returns:
            List of API information dictionaries
        """
        return [
            {
                'name': name,
                'title': info['title'],
                'version': info['version'],
                'description': info['description'],
                'endpoints': len(info['endpoints']),
                'base_url': info['base_url']
            }
            for name, info in self.api_registry.items()
        ]
    
    def remove_api(self, api_name: str) -> bool:
        """
        Remove an API from the registry.
        
        Args:
            api_name: Name of the API
            
        Returns:
            True if the API was removed, False otherwise
        """
        if api_name in self.api_registry:
            del self.api_registry[api_name]
            self._save_registry()
            return True
        return False
