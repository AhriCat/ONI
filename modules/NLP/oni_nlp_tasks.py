"""
Task identification and processing for ONI NLP
"""
import re
from typing import List, Dict, Any
from .oni_nlp_core import OniModule

class TaskIdentifier(OniModule):
    """Identifies tasks from input text"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Task patterns for identification
        self.task_patterns = {
            'search': ['/search', 'search for', 'find', 'look up', 'google'],
            'draw': ['/draw', 'draw', 'create image', 'generate picture', 'make art'],
            'animate': ['/animate', 'animate', 'create animation', 'make video'],
            'calculate': ['calculate', 'compute', 'math', 'solve', 'equation'],
            'code': ['code', 'program', 'write code', 'debug', 'programming'],
            'monitor': ['monitor', 'watch', 'observe', 'track', 'surveillance'],
            'browse': ['browse', 'open website', 'go to', 'navigate to'],
            'read': ['read', 'analyze document', 'process file', 'extract text'],
            'write': ['write', 'compose', 'create text', 'draft'],
            'translate': ['translate', 'convert language', 'interpret'],
            'summarize': ['summarize', 'summary', 'brief', 'overview'],
            'explain': ['explain', 'describe', 'clarify', 'elaborate'],
            'help': ['help', 'assist', 'support', 'guide'],
            'play': ['play', 'music', 'audio', 'sound'],
            'trade': ['trade', 'buy', 'sell', 'invest', 'market'],
            'schedule': ['schedule', 'plan', 'organize', 'calendar'],
            'email': ['email', 'send message', 'compose email'],
            'weather': ['weather', 'forecast', 'temperature', 'climate'],
            'news': ['news', 'current events', 'headlines', 'updates']
        }
        
        self.initialized = True
    
    def identify_tasks(self, text: str) -> List[str]:
        """Identify tasks from input text"""
        if not isinstance(text, str):
            return []
        
        text_lower = text.lower().strip()
        identified_tasks = []
        
        for task, patterns in self.task_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    identified_tasks.append(task)
                    break  # Only add each task once
        
        return identified_tasks
    
    def extract_task_parameters(self, text: str, task: str) -> Dict[str, Any]:
        """Extract parameters for specific tasks"""
        text_lower = text.lower().strip()
        parameters = {}
        
        if task == 'search':
            # Extract search query
            for pattern in ['/search', 'search for', 'find', 'look up']:
                if pattern in text_lower:
                    query = text_lower.split(pattern, 1)[-1].strip()
                    parameters['query'] = query
                    break
        
        elif task == 'draw':
            # Extract drawing prompt
            for pattern in ['/draw', 'draw', 'create image', 'generate picture']:
                if pattern in text_lower:
                    prompt = text_lower.split(pattern, 1)[-1].strip()
                    parameters['prompt'] = prompt
                    break
        
        elif task == 'animate':
            # Extract animation prompt
            for pattern in ['/animate', 'animate', 'create animation']:
                if pattern in text_lower:
                    prompt = text_lower.split(pattern, 1)[-1].strip()
                    parameters['prompt'] = prompt
                    break
        
        elif task == 'calculate':
            # Extract mathematical expression
            # Look for mathematical expressions
            math_patterns = [
                r'\d+[\+\-\*/]\d+',  # Simple arithmetic
                r'solve\s+(.+)',      # Solve equations
                r'calculate\s+(.+)',  # Calculate expressions
            ]
            for pattern in math_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    parameters['expression'] = match.group(1) if match.groups() else match.group(0)
                    break
        
        elif task == 'browse':
            # Extract URL or website name
            url_pattern = r'https?://[^\s]+'
            url_match = re.search(url_pattern, text)
            if url_match:
                parameters['url'] = url_match.group(0)
            else:
                # Look for website names
                for pattern in ['go to', 'open', 'browse', 'navigate to']:
                    if pattern in text_lower:
                        site = text_lower.split(pattern, 1)[-1].strip()
                        parameters['site'] = site
                        break
        
        elif task == 'translate':
            # Extract source and target languages
            lang_pattern = r'translate\s+(.+?)\s+(?:to|into)\s+(\w+)'
            match = re.search(lang_pattern, text_lower)
            if match:
                parameters['text'] = match.group(1)
                parameters['target_language'] = match.group(2)
        
        return parameters
    
    def get_task_priority(self, task: str) -> int:
        """Get priority level for tasks (lower number = higher priority)"""
        priority_map = {
            'help': 1,
            'calculate': 2,
            'search': 3,
            'read': 3,
            'translate': 4,
            'summarize': 4,
            'explain': 4,
            'write': 5,
            'code': 5,
            'draw': 6,
            'animate': 7,
            'browse': 8,
            'monitor': 9,
            'play': 10,
            'trade': 10
        }
        return priority_map.get(task, 5)  # Default priority
    
    def _get_fallback_output(self, text: str, **kwargs) -> List[str]:
        """Fallback output when task identification fails"""
        return []  # Return empty list as fallback
