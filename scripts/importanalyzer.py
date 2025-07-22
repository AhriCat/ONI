#!/usr/bin/env python3
"""
ONI Import Analyzer and Fixer
Scans Python modules for import issues and suggests fixes
"""

import os
import ast
import sys
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

class ImportAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.python_files = []
        self.imports = defaultdict(list)  # file -> list of imports
        self.issues = []
        
    def scan_directory(self):
        """Scan for all Python files in the project"""
        for root, dirs, files in os.walk(self.project_root):
            # Skip common directories that don't need scanning
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'oni_env'}]
            
            for file in files:
                if file.endswith('.py'):
                    self.python_files.append(Path(root) / file)
    
    def parse_imports(self, file_path: Path) -> List[Dict]:
        """Parse imports from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'type': 'import',
                            'module': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append({
                            'type': 'from_import',
                            'module': module,
                            'name': alias.name,
                            'alias': alias.asname,
                            'level': node.level,
                            'line': node.lineno
                        })
            
            return imports
            
        except Exception as e:
            self.issues.append(f"Error parsing {file_path}: {e}")
            return []
    
    def analyze_file(self, file_path: Path):
        """Analyze imports in a single file"""
        imports = self.parse_imports(file_path)
        self.imports[str(file_path)] = imports
        
        # Check for common import issues
        for imp in imports:
            self.check_import_issues(file_path, imp)
    
    def check_import_issues(self, file_path: Path, import_info: Dict):
        """Check for specific import issues"""
        module = import_info.get('module', '')
        name = import_info.get('name', '')
        line = import_info.get('line', 0)
        
        # Check for common problematic imports in ML/AI projects
        problematic_patterns = [
            ('torch', 'Check if PyTorch is properly installed'),
            ('transformers', 'Check if Transformers library is installed'),
            ('diffusers', 'Check if Diffusers library is installed'),
            ('opencv', 'OpenCV might need cv2 import instead'),
            ('PIL', 'Should be "from PIL import Image"'),
            ('numpy', 'Common import: import numpy as np'),
            ('pandas', 'Common import: import pandas as pd'),
            ('matplotlib', 'Common import: import matplotlib.pyplot as plt'),
            ('tensorflow', 'Check TensorFlow installation'),
            ('sklearn', 'Should be "from sklearn import ..."'),
        ]
        
        for pattern, suggestion in problematic_patterns:
            if pattern in module.lower() or pattern in name.lower():
                self.issues.append(f"{file_path}:{line} - {suggestion}")
        
        # Check for relative imports that might be broken
        if import_info.get('level', 0) > 0:  # Relative import
            relative_path = self.resolve_relative_import(file_path, import_info)
            if not relative_path.exists():
                self.issues.append(f"{file_path}:{line} - Relative import may be broken: {module}")
    
    def resolve_relative_import(self, file_path: Path, import_info: Dict) -> Path:
        """Resolve relative import path"""
        level = import_info.get('level', 0)
        module = import_info.get('module', '')
        
        # Go up 'level' directories from current file
        current_dir = file_path.parent
        for _ in range(level):
            current_dir = current_dir.parent
        
        # Build the module path
        if module:
            module_path = current_dir / module.replace('.', '/')
            # Check for __init__.py or .py file
            if (module_path / '__init__.py').exists():
                return module_path / '__init__.py'
            elif (module_path.parent / f'{module_path.name}.py').exists():
                return module_path.parent / f'{module_path.name}.py'
        
        return current_dir
    
    def generate_requirements_suggestions(self) -> List[str]:
        """Generate requirements.txt suggestions based on imports"""
        common_packages = {
            'torch': 'torch>=2.0.0',
            'torchvision': 'torchvision>=0.15.0',
            'transformers': 'transformers>=4.20.0',
            'diffusers': 'diffusers>=0.20.0',
            'numpy': 'numpy>=1.21.0',
            'pandas': 'pandas>=1.3.0',
            'matplotlib': 'matplotlib>=3.5.0',
            'opencv': 'opencv-python>=4.5.0',
            'PIL': 'Pillow>=8.0.0',
            'sklearn': 'scikit-learn>=1.0.0',
            'tensorflow': 'tensorflow>=2.8.0',
            'fastapi': 'fastapi>=0.70.0',
            'pydantic': 'pydantic>=1.8.0',
            'requests': 'requests>=2.25.0',
            'aiohttp': 'aiohttp>=3.8.0',
            'websockets': 'websockets>=10.0',
            'sqlalchemy': 'SQLAlchemy>=1.4.0',
        }
        
        found_packages = set()
        for file_imports in self.imports.values():
            for imp in file_imports:
                module = imp.get('module', '').split('.')[0]  # Get root module
                name = imp.get('name', '').split('.')[0] if imp.get('name') else ''
                
                for pkg in [module, name]:
                    if pkg in common_packages:
                        found_packages.add(common_packages[pkg])
        
        return sorted(list(found_packages))
    
    def suggest_import_fixes(self) -> Dict[str, List[str]]:
        """Suggest fixes for common import patterns"""
        fixes = defaultdict(list)
        
        for file_path, file_imports in self.imports.items():
            for imp in file_imports:
                module = imp.get('module', '')
                name = imp.get('name', '')
                line = imp.get('line', 0)
                
                # Common fix suggestions
                if module == 'cv2':
                    fixes[file_path].append(f"Line {line}: Consider 'import cv2' for OpenCV")
                
                if 'torch' in module and 'cuda' in name:
                    fixes[file_path].append(f"Line {line}: Check CUDA availability with torch.cuda.is_available()")
                
                if module == 'transformers' and name == '*':
                    fixes[file_path].append(f"Line {line}: Avoid wildcard imports from transformers")
                
                if 'huggingface_hub' in module:
                    fixes[file_path].append(f"Line {line}: May need: pip install huggingface-hub")
        
        return dict(fixes)
    
    def run_analysis(self):
        """Run complete import analysis"""
        print("🔍 Scanning ONI project for Python files...")
        self.scan_directory()
        print(f"Found {len(self.python_files)} Python files")
        
        print("\n📋 Analyzing imports...")
        for file_path in self.python_files:
            self.analyze_file(file_path)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate analysis report"""
        print("\n" + "="*60)
        print("ONI IMPORT ANALYSIS REPORT")
        print("="*60)
        
        if self.issues:
            print(f"\n⚠️  Found {len(self.issues)} potential issues:")
            for issue in self.issues:
                print(f"  • {issue}")
        else:
            print("\n✅ No obvious import issues found!")
        
        print(f"\n📦 Suggested requirements.txt entries:")
        requirements = self.generate_requirements_suggestions()
        if requirements:
            for req in requirements:
                print(f"  {req}")
        else:
            print("  No common packages detected")
        
        print(f"\n🔧 Import fix suggestions:")
        fixes = self.suggest_import_fixes()
        if fixes:
            for file_path, file_fixes in fixes.items():
                print(f"\n  {file_path}:")
                for fix in file_fixes:
                    print(f"    • {fix}")
        else:
            print("  No specific fixes suggested")
        
        print(f"\n📊 Import Statistics:")
        total_imports = sum(len(imports) for imports in self.imports.values())
        print(f"  • Total files analyzed: {len(self.python_files)}")
        print(f"  • Total imports found: {total_imports}")
        print(f"  • Average imports per file: {total_imports/len(self.python_files) if self.python_files else 0:.1f}")

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python import_analyzer.py <project_directory>")
        print("Example: python import_analyzer.py /path/to/ONI-Public")
        sys.exit(1)
    
    project_root = sys.argv[1]
    if not os.path.exists(project_root):
        print(f"Error: Directory '{project_root}' does not exist")
        sys.exit(1)
    
    analyzer = ImportAnalyzer(project_root)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
