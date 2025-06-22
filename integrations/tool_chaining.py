import logging
import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
import inspect
import traceback
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToolChain:
    """
    Tool chaining module for creating and executing complex multi-tool workflows.
    
    This module enables:
    1. Creating sequential and parallel tool chains
    2. Defining data transformations between tools
    3. Conditional branching based on tool outputs
    4. Monitoring and visualization of tool execution
    """
    
    def __init__(self, name: str = "default_chain"):
        """
        Initialize a tool chain.
        
        Args:
            name: Name of the tool chain
        """
        self.name = name
        self.steps = []
        self.transformers = {}
        self.conditions = {}
        self.error_handlers = {}
        self.global_error_handler = None
        self.execution_history = []
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0
        }
    
    def add_step(self, step_id: str, tool: Callable, 
                description: str = "", 
                required_inputs: Optional[List[str]] = None,
                output_mapping: Optional[Dict[str, str]] = None) -> 'ToolChain':
        """
        Add a step to the tool chain.
        
        Args:
            step_id: Unique identifier for the step
            tool: Function or callable to execute
            description: Description of the step
            required_inputs: List of required input parameter names
            output_mapping: Mapping of output keys to new names
            
        Returns:
            Self for method chaining
        """
        # Validate step_id uniqueness
        if any(step['id'] == step_id for step in self.steps):
            raise ValueError(f"Step ID '{step_id}' already exists in the chain")
        
        # Get function signature
        sig = inspect.signature(tool)
        param_names = list(sig.parameters.keys())
        
        # Create step
        step = {
            'id': step_id,
            'tool': tool,
            'description': description,
            'required_inputs': required_inputs or param_names,
            'output_mapping': output_mapping or {},
            'parallel': False,
            'optional': False
        }
        
        self.steps.append(step)
        return self
    
    def add_parallel_steps(self, steps: List[Dict[str, Any]]) -> 'ToolChain':
        """
        Add multiple steps to be executed in parallel.
        
        Args:
            steps: List of step configurations
            
        Returns:
            Self for method chaining
        """
        # Create a group ID for these parallel steps
        group_id = f"parallel_group_{len(self.steps)}"
        
        for step_config in steps:
            step_id = step_config['id']
            tool = step_config['tool']
            description = step_config.get('description', "")
            required_inputs = step_config.get('required_inputs')
            output_mapping = step_config.get('output_mapping', {})
            
            # Validate step_id uniqueness
            if any(step['id'] == step_id for step in self.steps):
                raise ValueError(f"Step ID '{step_id}' already exists in the chain")
            
            # Get function signature if required_inputs not provided
            if required_inputs is None:
                sig = inspect.signature(tool)
                required_inputs = list(sig.parameters.keys())
            
            # Create step
            step = {
                'id': step_id,
                'tool': tool,
                'description': description,
                'required_inputs': required_inputs,
                'output_mapping': output_mapping,
                'parallel': True,
                'parallel_group': group_id,
                'optional': step_config.get('optional', False)
            }
            
            self.steps.append(step)
        
        return self
    
    def add_transformer(self, from_step: str, to_step: str, 
                       transformer: Callable[[Dict[str, Any]], Dict[str, Any]]) -> 'ToolChain':
        """
        Add a data transformer between steps.
        
        Args:
            from_step: ID of the source step
            to_step: ID of the destination step
            transformer: Function to transform data
            
        Returns:
            Self for method chaining
        """
        self.transformers[(from_step, to_step)] = transformer
        return self
    
    def add_condition(self, step_id: str, condition: Callable[[Dict[str, Any]], bool]) -> 'ToolChain':
        """
        Add a condition for executing a step.
        
        Args:
            step_id: ID of the step
            condition: Function that returns True if the step should be executed
            
        Returns:
            Self for method chaining
        """
        self.conditions[step_id] = condition
        return self
    
    def add_error_handler(self, step_id: str, handler: Callable[[Exception, Dict[str, Any]], Dict[str, Any]]) -> 'ToolChain':
        """
        Add an error handler for a specific step.
        
        Args:
            step_id: ID of the step
            handler: Function to handle errors
            
        Returns:
            Self for method chaining
        """
        self.error_handlers[step_id] = handler
        return self
    
    def set_global_error_handler(self, handler: Callable[[Exception, str, Dict[str, Any]], Dict[str, Any]]) -> 'ToolChain':
        """
        Set a global error handler for all steps.
        
        Args:
            handler: Function to handle errors
            
        Returns:
            Self for method chaining
        """
        self.global_error_handler = handler
        return self
    
    def execute(self, initial_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the tool chain.
        
        Args:
            initial_inputs: Initial input data
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        self.execution_stats['total_executions'] += 1
        
        # Initialize context with initial inputs
        context = {
            'inputs': initial_inputs or {},
            'outputs': {},
            'errors': {},
            'execution_path': [],
            'start_time': start_time
        }
        
        try:
            # Group steps by parallel groups
            step_groups = []
            current_group = []
            current_group_id = None
            
            for step in self.steps:
                if step['parallel']:
                    if current_group_id is None or step['parallel_group'] != current_group_id:
                        if current_group:
                            step_groups.append(current_group)
                        current_group = [step]
                        current_group_id = step['parallel_group']
                    else:
                        current_group.append(step)
                else:
                    if current_group:
                        step_groups.append(current_group)
                        current_group = []
                        current_group_id = None
                    step_groups.append([step])
            
            if current_group:
                step_groups.append(current_group)
            
            # Execute step groups
            for group in step_groups:
                if len(group) == 1 and not group[0]['parallel']:
                    # Sequential step
                    step = group[0]
                    self._execute_step(step, context)
                else:
                    # Parallel steps
                    self._execute_parallel_steps(group, context)
            
            # Record successful execution
            self.execution_stats['successful_executions'] += 1
            
        except Exception as e:
            # Record failed execution
            self.execution_stats['failed_executions'] += 1
            
            # Add error to context
            context['errors']['global'] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            # Try global error handler
            if self.global_error_handler:
                try:
                    context = self.global_error_handler(e, "global", context)
                except Exception as handler_error:
                    logger.error(f"Global error handler failed: {handler_error}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        context['execution_time'] = execution_time
        
        # Update average execution time
        total_successful = self.execution_stats['successful_executions']
        if total_successful > 0:
            current_avg = self.execution_stats['average_execution_time']
            new_avg = ((current_avg * (total_successful - 1)) + execution_time) / total_successful
            self.execution_stats['average_execution_time'] = new_avg
        
        # Add to execution history
        self.execution_history.append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'success': 'global' not in context['errors'],
            'execution_path': context['execution_path']
        })
        
        # Limit history size
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        return context
    
    async def execute_async(self, initial_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the tool chain asynchronously.
        
        Args:
            initial_inputs: Initial input data
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        self.execution_stats['total_executions'] += 1
        
        # Initialize context with initial inputs
        context = {
            'inputs': initial_inputs or {},
            'outputs': {},
            'errors': {},
            'execution_path': [],
            'start_time': start_time
        }
        
        try:
            # Group steps by parallel groups
            step_groups = []
            current_group = []
            current_group_id = None
            
            for step in self.steps:
                if step['parallel']:
                    if current_group_id is None or step['parallel_group'] != current_group_id:
                        if current_group:
                            step_groups.append(current_group)
                        current_group = [step]
                        current_group_id = step['parallel_group']
                    else:
                        current_group.append(step)
                else:
                    if current_group:
                        step_groups.append(current_group)
                        current_group = []
                        current_group_id = None
                    step_groups.append([step])
            
            if current_group:
                step_groups.append(current_group)
            
            # Execute step groups
            for group in step_groups:
                if len(group) == 1 and not group[0]['parallel']:
                    # Sequential step
                    step = group[0]
                    await self._execute_step_async(step, context)
                else:
                    # Parallel steps
                    await self._execute_parallel_steps_async(group, context)
            
            # Record successful execution
            self.execution_stats['successful_executions'] += 1
            
        except Exception as e:
            # Record failed execution
            self.execution_stats['failed_executions'] += 1
            
            # Add error to context
            context['errors']['global'] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            # Try global error handler
            if self.global_error_handler:
                try:
                    context = self.global_error_handler(e, "global", context)
                except Exception as handler_error:
                    logger.error(f"Global error handler failed: {handler_error}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        context['execution_time'] = execution_time
        
        # Update average execution time
        total_successful = self.execution_stats['successful_executions']
        if total_successful > 0:
            current_avg = self.execution_stats['average_execution_time']
            new_avg = ((current_avg * (total_successful - 1)) + execution_time) / total_successful
            self.execution_stats['average_execution_time'] = new_avg
        
        # Add to execution history
        self.execution_history.append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'success': 'global' not in context['errors'],
            'execution_path': context['execution_path']
        })
        
        # Limit history size
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        return context
    
    def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Execute a single step in the chain."""
        step_id = step['id']
        
        # Check if step should be executed based on condition
        if step_id in self.conditions:
            condition = self.conditions[step_id]
            if not condition(context):
                logger.info(f"Skipping step '{step_id}' due to condition")
                return
        
        # Prepare inputs for the step
        step_inputs = {}
        missing_inputs = []
        
        for input_name in step['required_inputs']:
            if input_name in context['inputs']:
                step_inputs[input_name] = context['inputs'][input_name]
            else:
                # Check if input is available from previous step outputs
                found = False
                for prev_step_id, outputs in context['outputs'].items():
                    if input_name in outputs:
                        step_inputs[input_name] = outputs[input_name]
                        found = True
                        break
                
                if not found:
                    missing_inputs.append(input_name)
        
        # Check for missing inputs
        if missing_inputs and not step['optional']:
            error_msg = f"Missing required inputs for step '{step_id}': {', '.join(missing_inputs)}"
            logger.error(error_msg)
            context['errors'][step_id] = {
                'error': error_msg,
                'missing_inputs': missing_inputs
            }
            
            # Try error handler
            if step_id in self.error_handlers:
                try:
                    handler = self.error_handlers[step_id]
                    context = handler(ValueError(error_msg), context)
                except Exception as handler_error:
                    logger.error(f"Error handler for step '{step_id}' failed: {handler_error}")
            
            # Try global error handler
            elif self.global_error_handler:
                try:
                    context = self.global_error_handler(ValueError(error_msg), step_id, context)
                except Exception as handler_error:
                    logger.error(f"Global error handler failed: {handler_error}")
            
            return
        
        # Apply transformers
        for (from_step, to_step), transformer in self.transformers.items():
            if to_step == step_id and from_step in context['outputs']:
                try:
                    transformed_data = transformer(context['outputs'][from_step])
                    for key, value in transformed_data.items():
                        step_inputs[key] = value
                except Exception as e:
                    logger.error(f"Transformer from '{from_step}' to '{step_id}' failed: {e}")
                    context['errors'][f"transformer_{from_step}_{to_step}"] = {
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
        
        # Record step in execution path
        context['execution_path'].append(step_id)
        
        # Execute the step
        try:
            logger.info(f"Executing step '{step_id}'")
            start_time = time.time()
            
            # Call the tool
            result = step['tool'](**step_inputs)
            
            execution_time = time.time() - start_time
            logger.info(f"Step '{step_id}' completed in {execution_time:.2f}s")
            
            # Process result
            if isinstance(result, dict):
                # Apply output mapping
                mapped_result = {}
                for key, value in result.items():
                    output_key = step['output_mapping'].get(key, key)
                    mapped_result[output_key] = value
                
                # Store outputs
                context['outputs'][step_id] = mapped_result
                
                # Add to inputs for subsequent steps
                for key, value in mapped_result.items():
                    context['inputs'][key] = value
            else:
                # Store non-dict result
                context['outputs'][step_id] = {'result': result}
                context['inputs']['result'] = result
            
        except Exception as e:
            logger.error(f"Step '{step_id}' failed: {e}")
            context['errors'][step_id] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            # Try step-specific error handler
            if step_id in self.error_handlers:
                try:
                    handler = self.error_handlers[step_id]
                    context = handler(e, context)
                except Exception as handler_error:
                    logger.error(f"Error handler for step '{step_id}' failed: {handler_error}")
            
            # Try global error handler
            elif self.global_error_handler:
                try:
                    context = self.global_error_handler(e, step_id, context)
                except Exception as handler_error:
                    logger.error(f"Global error handler failed: {handler_error}")
    
    def _execute_parallel_steps(self, steps: List[Dict[str, Any]], context: Dict[str, Any]) -> None:
        """Execute multiple steps in parallel."""
        # Create a copy of the context for each step
        contexts = [copy.deepcopy(context) for _ in steps]
        
        # Execute each step with its own context
        for i, step in enumerate(steps):
            self._execute_step(step, contexts[i])
        
        # Merge contexts
        for i, step_context in enumerate(contexts):
            step_id = steps[i]['id']
            
            # Add to execution path if not already added
            if step_id not in context['execution_path']:
                context['execution_path'].append(step_id)
            
            # Merge outputs
            if step_id in step_context['outputs']:
                context['outputs'][step_id] = step_context['outputs'][step_id]
                
                # Add to inputs for subsequent steps
                for key, value in step_context['outputs'][step_id].items():
                    context['inputs'][key] = value
            
            # Merge errors
            if step_id in step_context['errors']:
                context['errors'][step_id] = step_context['errors'][step_id]
    
    async def _execute_step_async(self, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Execute a single step in the chain asynchronously."""
        step_id = step['id']
        
        # Check if step should be executed based on condition
        if step_id in self.conditions:
            condition = self.conditions[step_id]
            if not condition(context):
                logger.info(f"Skipping step '{step_id}' due to condition")
                return
        
        # Prepare inputs for the step
        step_inputs = {}
        missing_inputs = []
        
        for input_name in step['required_inputs']:
            if input_name in context['inputs']:
                step_inputs[input_name] = context['inputs'][input_name]
            else:
                # Check if input is available from previous step outputs
                found = False
                for prev_step_id, outputs in context['outputs'].items():
                    if input_name in outputs:
                        step_inputs[input_name] = outputs[input_name]
                        found = True
                        break
                
                if not found:
                    missing_inputs.append(input_name)
        
        # Check for missing inputs
        if missing_inputs and not step['optional']:
            error_msg = f"Missing required inputs for step '{step_id}': {', '.join(missing_inputs)}"
            logger.error(error_msg)
            context['errors'][step_id] = {
                'error': error_msg,
                'missing_inputs': missing_inputs
            }
            
            # Try error handler
            if step_id in self.error_handlers:
                try:
                    handler = self.error_handlers[step_id]
                    context = handler(ValueError(error_msg), context)
                except Exception as handler_error:
                    logger.error(f"Error handler for step '{step_id}' failed: {handler_error}")
            
            # Try global error handler
            elif self.global_error_handler:
                try:
                    context = self.global_error_handler(ValueError(error_msg), step_id, context)
                except Exception as handler_error:
                    logger.error(f"Global error handler failed: {handler_error}")
            
            return
        
        # Apply transformers
        for (from_step, to_step), transformer in self.transformers.items():
            if to_step == step_id and from_step in context['outputs']:
                try:
                    transformed_data = transformer(context['outputs'][from_step])
                    for key, value in transformed_data.items():
                        step_inputs[key] = value
                except Exception as e:
                    logger.error(f"Transformer from '{from_step}' to '{step_id}' failed: {e}")
                    context['errors'][f"transformer_{from_step}_{to_step}"] = {
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
        
        # Record step in execution path
        context['execution_path'].append(step_id)
        
        # Execute the step
        try:
            logger.info(f"Executing step '{step_id}'")
            start_time = time.time()
            
            # Call the tool
            tool = step['tool']
            if asyncio.iscoroutinefunction(tool):
                # Async function
                result = await tool(**step_inputs)
            else:
                # Sync function
                result = await asyncio.to_thread(tool, **step_inputs)
            
            execution_time = time.time() - start_time
            logger.info(f"Step '{step_id}' completed in {execution_time:.2f}s")
            
            # Process result
            if isinstance(result, dict):
                # Apply output mapping
                mapped_result = {}
                for key, value in result.items():
                    output_key = step['output_mapping'].get(key, key)
                    mapped_result[output_key] = value
                
                # Store outputs
                context['outputs'][step_id] = mapped_result
                
                # Add to inputs for subsequent steps
                for key, value in mapped_result.items():
                    context['inputs'][key] = value
            else:
                # Store non-dict result
                context['outputs'][step_id] = {'result': result}
                context['inputs']['result'] = result
            
        except Exception as e:
            logger.error(f"Step '{step_id}' failed: {e}")
            context['errors'][step_id] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            # Try step-specific error handler
            if step_id in self.error_handlers:
                try:
                    handler = self.error_handlers[step_id]
                    context = handler(e, context)
                except Exception as handler_error:
                    logger.error(f"Error handler for step '{step_id}' failed: {handler_error}")
            
            # Try global error handler
            elif self.global_error_handler:
                try:
                    context = self.global_error_handler(e, step_id, context)
                except Exception as handler_error:
                    logger.error(f"Global error handler failed: {handler_error}")
    
    async def _execute_parallel_steps_async(self, steps: List[Dict[str, Any]], context: Dict[str, Any]) -> None:
        """Execute multiple steps in parallel asynchronously."""
        # Create a copy of the context for each step
        contexts = [copy.deepcopy(context) for _ in steps]
        
        # Create tasks for each step
        tasks = [self._execute_step_async(step, contexts[i]) for i, step in enumerate(steps)]
        
        # Execute all tasks in parallel
        await asyncio.gather(*tasks)
        
        # Merge contexts
        for i, step_context in enumerate(contexts):
            step_id = steps[i]['id']
            
            # Add to execution path if not already added
            if step_id not in context['execution_path']:
                context['execution_path'].append(step_id)
            
            # Merge outputs
            if step_id in step_context['outputs']:
                context['outputs'][step_id] = step_context['outputs'][step_id]
                
                # Add to inputs for subsequent steps
                for key, value in step_context['outputs'][step_id].items():
                    context['inputs'][key] = value
            
            # Merge errors
            if step_id in step_context['errors']:
                context['errors'][step_id] = step_context['errors'][step_id]
    
    def visualize(self) -> Dict[str, Any]:
        """
        Generate a visualization of the tool chain.
        
        Returns:
            Dictionary with visualization data
        """
        nodes = []
        edges = []
        
        # Add nodes for each step
        for i, step in enumerate(self.steps):
            nodes.append({
                'id': step['id'],
                'label': step['id'],
                'description': step['description'],
                'type': 'parallel' if step['parallel'] else 'sequential',
                'optional': step['optional'],
                'inputs': step['required_inputs'],
                'outputs': list(step['output_mapping'].values()) if step['output_mapping'] else []
            })
            
            # Add edges between steps
            if i > 0:
                prev_step = self.steps[i-1]
                
                # Skip if this is the start of a parallel group
                if step['parallel'] and (i == 0 or not prev_step['parallel'] or 
                                       prev_step.get('parallel_group') != step.get('parallel_group')):
                    continue
                
                # Add edge
                edges.append({
                    'from': prev_step['id'],
                    'to': step['id'],
                    'type': 'flow'
                })
        
        # Add edges for transformers
        for (from_step, to_step) in self.transformers:
            edges.append({
                'from': from_step,
                'to': to_step,
                'type': 'transformer'
            })
        
        # Add edges for conditions
        for step_id in self.conditions:
            edges.append({
                'to': step_id,
                'type': 'condition'
            })
        
        return {
            'name': self.name,
            'nodes': nodes,
            'edges': edges,
            'stats': self.execution_stats
        }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the execution history of the tool chain.
        
        Returns:
            List of execution history entries
        """
        return self.execution_history
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for the tool chain.
        
        Returns:
            Dictionary with execution statistics
        """
        return self.execution_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool chain to a dictionary.
        
        Returns:
            Dictionary representation of the tool chain
        """
        return {
            'name': self.name,
            'steps': [
                {
                    'id': step['id'],
                    'description': step['description'],
                    'required_inputs': step['required_inputs'],
                    'output_mapping': step['output_mapping'],
                    'parallel': step['parallel'],
                    'parallel_group': step.get('parallel_group'),
                    'optional': step['optional']
                }
                for step in self.steps
            ],
            'transformers': list(self.transformers.keys()),
            'conditions': list(self.conditions.keys()),
            'error_handlers': list(self.error_handlers.keys()),
            'has_global_error_handler': self.global_error_handler is not None,
            'stats': self.execution_stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], tools: Dict[str, Callable],
                transformers: Dict[str, Callable], conditions: Dict[str, Callable],
                error_handlers: Dict[str, Callable]) -> 'ToolChain':
        """
        Create a tool chain from a dictionary.
        
        Args:
            data: Dictionary representation of the tool chain
            tools: Dictionary mapping step IDs to tool functions
            transformers: Dictionary mapping (from_step, to_step) tuples to transformer functions
            conditions: Dictionary mapping step IDs to condition functions
            error_handlers: Dictionary mapping step IDs to error handler functions
            
        Returns:
            ToolChain instance
        """
        chain = cls(data['name'])
        
        # Add steps
        for step_data in data['steps']:
            step_id = step_data['id']
            
            if step_id not in tools:
                raise ValueError(f"Tool for step '{step_id}' not found")
                
            tool = tools[step_id]
            
            if step_data['parallel']:
                # Add to parallel group
                chain.add_parallel_steps([{
                    'id': step_id,
                    'tool': tool,
                    'description': step_data['description'],
                    'required_inputs': step_data['required_inputs'],
                    'output_mapping': step_data['output_mapping'],
                    'optional': step_data['optional']
                }])
            else:
                # Add sequential step
                chain.add_step(
                    step_id=step_id,
                    tool=tool,
                    description=step_data['description'],
                    required_inputs=step_data['required_inputs'],
                    output_mapping=step_data['output_mapping']
                )
        
        # Add transformers
        for from_step, to_step in data['transformers']:
            transformer_key = (from_step, to_step)
            if transformer_key in transformers:
                chain.add_transformer(from_step, to_step, transformers[transformer_key])
        
        # Add conditions
        for step_id in data['conditions']:
            if step_id in conditions:
                chain.add_condition(step_id, conditions[step_id])
        
        # Add error handlers
        for step_id in data['error_handlers']:
            if step_id in error_handlers:
                chain.add_error_handler(step_id, error_handlers[step_id])
        
        # Add global error handler
        if data.get('has_global_error_handler') and 'global' in error_handlers:
            chain.set_global_error_handler(error_handlers['global'])
        
        return chain

class ToolChainRegistry:
    """Registry for managing and executing tool chains."""
    
    def __init__(self):
        """Initialize the tool chain registry."""
        self.chains = {}
        self.tools = {}
    
    def register_tool(self, tool_id: str, tool: Callable, description: str = "") -> None:
        """
        Register a tool in the registry.
        
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
        logger.info(f"Registered tool: {tool_id}")
    
    def register_chain(self, chain: ToolChain) -> None:
        """
        Register a tool chain in the registry.
        
        Args:
            chain: ToolChain instance
        """
        self.chains[chain.name] = chain
        logger.info(f"Registered tool chain: {chain.name}")
    
    def get_chain(self, name: str) -> Optional[ToolChain]:
        """
        Get a tool chain by name.
        
        Args:
            name: Name of the tool chain
            
        Returns:
            ToolChain instance or None if not found
        """
        return self.chains.get(name)
    
    def get_tool(self, tool_id: str) -> Optional[Callable]:
        """
        Get a tool by ID.
        
        Args:
            tool_id: ID of the tool
            
        Returns:
            Tool function or None if not found
        """
        tool_info = self.tools.get(tool_id)
        return tool_info['tool'] if tool_info else None
    
    def list_chains(self) -> List[Dict[str, Any]]:
        """
        List all registered tool chains.
        
        Returns:
            List of tool chain information dictionaries
        """
        return [
            {
                'name': name,
                'steps': len(chain.steps),
                'stats': chain.execution_stats
            }
            for name, chain in self.chains.items()
        ]
    
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
    
    def execute_chain(self, chain_name: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a tool chain.
        
        Args:
            chain_name: Name of the tool chain
            inputs: Input data
            
        Returns:
            Dictionary with execution results
        """
        chain = self.get_chain(chain_name)
        if not chain:
            return {
                'success': False,
                'error': f"Tool chain '{chain_name}' not found"
            }
        
        return chain.execute(inputs)
    
    async def execute_chain_async(self, chain_name: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a tool chain asynchronously.
        
        Args:
            chain_name: Name of the tool chain
            inputs: Input data
            
        Returns:
            Dictionary with execution results
        """
        chain = self.get_chain(chain_name)
        if not chain:
            return {
                'success': False,
                'error': f"Tool chain '{chain_name}' not found"
            }
        
        return await chain.execute_async(inputs)
    
    def create_chain_from_tools(self, chain_name: str, tool_ids: List[str], 
                              parallel: bool = False) -> ToolChain:
        """
        Create a tool chain from a list of tools.
        
        Args:
            chain_name: Name for the new chain
            tool_ids: List of tool IDs to include in the chain
            parallel: Whether to execute tools in parallel
            
        Returns:
            New ToolChain instance
        """
        chain = ToolChain(chain_name)
        
        if parallel:
            # Add all tools as parallel steps
            steps = []
            for tool_id in tool_ids:
                if tool_id not in self.tools:
                    raise ValueError(f"Tool '{tool_id}' not found")
                
                tool_info = self.tools[tool_id]
                steps.append({
                    'id': tool_id,
                    'tool': tool_info['tool'],
                    'description': tool_info['description']
                })
            
            chain.add_parallel_steps(steps)
        else:
            # Add tools as sequential steps
            for tool_id in tool_ids:
                if tool_id not in self.tools:
                    raise ValueError(f"Tool '{tool_id}' not found")
                
                tool_info = self.tools[tool_id]
                chain.add_step(
                    step_id=tool_id,
                    tool=tool_info['tool'],
                    description=tool_info['description']
                )
        
        # Register the new chain
        self.register_chain(chain)
        
        return chain
