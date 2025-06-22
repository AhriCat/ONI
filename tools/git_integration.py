import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import subprocess
import threading
import queue
import traceback
import tempfile
import shutil

# Import git libraries
try:
    import git
    from git import Repo
    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False

try:
    import isomorphic_git as iso_git
    HAS_ISOMORPHIC_GIT = True
except ImportError:
    HAS_ISOMORPHIC_GIT = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitIntegration:
    """Git integration for ONI to manage code repositories and version control."""
    
    def __init__(self, work_dir: Optional[str] = None):
        """
        Initialize Git integration.
        
        Args:
            work_dir: Working directory for Git operations (default: temporary directory)
        """
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="oni_git_")
        self.repos = {}  # name -> Repo object
        self.active_repo = None
        self.active_repo_name = None
        
        # Check available Git libraries
        self.git_python_available = HAS_GITPYTHON
        self.isomorphic_git_available = HAS_ISOMORPHIC_GIT
        
        if not (self.git_python_available or self.isomorphic_git_available):
            logger.warning("No Git library available. Install GitPython or isomorphic-git.")
        
        # Create work directory if it doesn't exist
        os.makedirs(self.work_dir, exist_ok=True)
        
        logger.info(f"Git integration initialized with work directory: {self.work_dir}")
    
    def __del__(self):
        """Clean up resources."""
        if self.work_dir.startswith(tempfile.gettempdir()):
            try:
                shutil.rmtree(self.work_dir)
                logger.info(f"Removed temporary directory: {self.work_dir}")
            except Exception as e:
                logger.error(f"Failed to remove temporary directory: {e}")
    
    def clone_repository(self, url: str, repo_name: Optional[str] = None, 
                        branch: Optional[str] = None) -> Dict[str, Any]:
        """
        Clone a Git repository.
        
        Args:
            url: Repository URL
            repo_name: Name to identify the repository (default: derived from URL)
            branch: Branch to checkout (default: default branch)
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Derive repo name from URL if not provided
            if repo_name is None:
                repo_name = url.split('/')[-1]
                if repo_name.endswith('.git'):
                    repo_name = repo_name[:-4]
            
            # Create repo directory
            repo_path = os.path.join(self.work_dir, repo_name)
            
            if os.path.exists(repo_path):
                return {
                    "success": False,
                    "error": f"Repository directory already exists: {repo_path}"
                }
            
            if self.git_python_available:
                # Clone using GitPython
                repo = git.Repo.clone_from(url, repo_path)
                
                # Checkout specific branch if requested
                if branch:
                    repo.git.checkout(branch)
                
                # Store repo
                self.repos[repo_name] = repo
                self.active_repo = repo
                self.active_repo_name = repo_name
                
                logger.info(f"Cloned repository: {url} to {repo_path}")
                return {
                    "success": True,
                    "repo_name": repo_name,
                    "path": repo_path,
                    "branch": branch or repo.active_branch.name
                }
            
            elif self.isomorphic_git_available:
                # Clone using isomorphic-git
                os.makedirs(repo_path, exist_ok=True)
                
                result = iso_git.clone({
                    'fs': iso_git.fs,
                    'dir': repo_path,
                    'url': url,
                    'ref': branch,
                    'singleBranch': branch is not None,
                    'depth': 1
                })
                
                # Store repo info
                self.repos[repo_name] = {
                    'path': repo_path,
                    'url': url
                }
                self.active_repo = self.repos[repo_name]
                self.active_repo_name = repo_name
                
                logger.info(f"Cloned repository: {url} to {repo_path}")
                return {
                    "success": True,
                    "repo_name": repo_name,
                    "path": repo_path,
                    "branch": branch or "main"  # Assuming main as default
                }
            
            else:
                # Fallback to command line git
                os.makedirs(repo_path, exist_ok=True)
                
                cmd = ["git", "clone"]
                if branch:
                    cmd.extend(["--branch", branch, "--single-branch"])
                cmd.extend([url, repo_path])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git clone failed: {result.stderr}"
                    }
                
                # Store repo info
                self.repos[repo_name] = {
                    'path': repo_path,
                    'url': url
                }
                self.active_repo = self.repos[repo_name]
                self.active_repo_name = repo_name
                
                logger.info(f"Cloned repository: {url} to {repo_path}")
                return {
                    "success": True,
                    "repo_name": repo_name,
                    "path": repo_path,
                    "branch": branch or "main"  # Assuming main as default
                }
                
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def init_repository(self, repo_name: str, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize a new Git repository.
        
        Args:
            repo_name: Name to identify the repository
            path: Path for the repository (default: subdirectory in work_dir)
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Set repo path
            repo_path = path or os.path.join(self.work_dir, repo_name)
            
            # Create directory if it doesn't exist
            os.makedirs(repo_path, exist_ok=True)
            
            if self.git_python_available:
                # Initialize using GitPython
                repo = git.Repo.init(repo_path)
                
                # Store repo
                self.repos[repo_name] = repo
                self.active_repo = repo
                self.active_repo_name = repo_name
                
                logger.info(f"Initialized repository: {repo_path}")
                return {
                    "success": True,
                    "repo_name": repo_name,
                    "path": repo_path
                }
            
            elif self.isomorphic_git_available:
                # Initialize using isomorphic-git
                result = iso_git.init({
                    'fs': iso_git.fs,
                    'dir': repo_path
                })
                
                # Store repo info
                self.repos[repo_name] = {
                    'path': repo_path
                }
                self.active_repo = self.repos[repo_name]
                self.active_repo_name = repo_name
                
                logger.info(f"Initialized repository: {repo_path}")
                return {
                    "success": True,
                    "repo_name": repo_name,
                    "path": repo_path
                }
            
            else:
                # Fallback to command line git
                cmd = ["git", "init", repo_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git init failed: {result.stderr}"
                    }
                
                # Store repo info
                self.repos[repo_name] = {
                    'path': repo_path
                }
                self.active_repo = self.repos[repo_name]
                self.active_repo_name = repo_name
                
                logger.info(f"Initialized repository: {repo_path}")
                return {
                    "success": True,
                    "repo_name": repo_name,
                    "path": repo_path
                }
                
        except Exception as e:
            logger.error(f"Failed to initialize repository: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def set_active_repository(self, repo_name: str) -> Dict[str, Any]:
        """
        Set the active repository for operations.
        
        Args:
            repo_name: Name of the repository to set as active
            
        Returns:
            Dictionary with operation result
        """
        if repo_name not in self.repos:
            return {
                "success": False,
                "error": f"Repository not found: {repo_name}"
            }
        
        self.active_repo = self.repos[repo_name]
        self.active_repo_name = repo_name
        
        logger.info(f"Active repository set to: {repo_name}")
        return {
            "success": True,
            "repo_name": repo_name
        }
    
    def get_status(self, repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of a repository.
        
        Args:
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with repository status
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Get status using GitPython
                repo_path = repo.working_dir
                
                # Get branch
                try:
                    branch = repo.active_branch.name
                except TypeError:
                    branch = "DETACHED_HEAD"
                
                # Get modified files
                modified_files = [item.a_path for item in repo.index.diff(None)]
                
                # Get staged files
                staged_files = [item.a_path for item in repo.index.diff("HEAD")]
                
                # Get untracked files
                untracked_files = repo.untracked_files
                
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "path": repo_path,
                    "branch": branch,
                    "modified_files": modified_files,
                    "staged_files": staged_files,
                    "untracked_files": untracked_files
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                # Get status
                cmd = ["git", "-C", repo_path, "status", "--porcelain", "--branch"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git status failed: {result.stderr}"
                    }
                
                # Parse output
                lines = result.stdout.strip().split('\n')
                
                # Get branch
                branch_line = lines[0] if lines else ""
                branch = branch_line.replace("## ", "").split("...")[0] if branch_line.startswith("## ") else "unknown"
                
                # Parse status
                modified_files = []
                staged_files = []
                untracked_files = []
                
                for line in lines[1:]:
                    if not line.strip():
                        continue
                        
                    status = line[:2]
                    file_path = line[3:]
                    
                    if status == "??":
                        untracked_files.append(file_path)
                    elif status[0] == " " and status[1] != " ":
                        modified_files.append(file_path)
                    elif status[0] != " ":
                        staged_files.append(file_path)
                
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "path": repo_path,
                    "branch": branch,
                    "modified_files": modified_files,
                    "staged_files": staged_files,
                    "untracked_files": untracked_files
                }
                
        except Exception as e:
            logger.error(f"Failed to get repository status: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_files(self, files: List[str], repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Add files to the Git index.
        
        Args:
            files: List of file paths to add
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Add files using GitPython
                repo.git.add(files)
                
                logger.info(f"Added files to index: {files}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "files": files
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                cmd = ["git", "-C", repo_path, "add"] + files
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git add failed: {result.stderr}"
                    }
                
                logger.info(f"Added files to index: {files}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "files": files
                }
                
        except Exception as e:
            logger.error(f"Failed to add files: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def commit(self, message: str, repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Commit changes to the repository.
        
        Args:
            message: Commit message
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Commit using GitPython
                commit = repo.git.commit(m=message)
                
                logger.info(f"Committed changes: {message}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "message": message,
                    "commit": commit
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                cmd = ["git", "-C", repo_path, "commit", "-m", message]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git commit failed: {result.stderr}"
                    }
                
                logger.info(f"Committed changes: {message}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "message": message,
                    "commit": result.stdout
                }
                
        except Exception as e:
            logger.error(f"Failed to commit changes: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def checkout(self, branch: str, create: bool = False, 
                repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Checkout a branch.
        
        Args:
            branch: Branch name
            create: Whether to create the branch if it doesn't exist
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Checkout using GitPython
                if create:
                    repo.git.checkout('-b', branch)
                else:
                    repo.git.checkout(branch)
                
                logger.info(f"Checked out branch: {branch}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "branch": branch,
                    "created": create
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                cmd = ["git", "-C", repo_path, "checkout"]
                if create:
                    cmd.append("-b")
                cmd.append(branch)
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git checkout failed: {result.stderr}"
                    }
                
                logger.info(f"Checked out branch: {branch}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "branch": branch,
                    "created": create
                }
                
        except Exception as e:
            logger.error(f"Failed to checkout branch: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def pull(self, remote: str = "origin", branch: Optional[str] = None, 
            repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Pull changes from a remote repository.
        
        Args:
            remote: Remote name
            branch: Branch name (default: current branch)
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Pull using GitPython
                if branch:
                    result = repo.git.pull(remote, branch)
                else:
                    result = repo.git.pull(remote)
                
                logger.info(f"Pulled changes from {remote}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "remote": remote,
                    "branch": branch,
                    "result": result
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                cmd = ["git", "-C", repo_path, "pull", remote]
                if branch:
                    cmd.append(branch)
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git pull failed: {result.stderr}"
                    }
                
                logger.info(f"Pulled changes from {remote}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "remote": remote,
                    "branch": branch,
                    "result": result.stdout
                }
                
        except Exception as e:
            logger.error(f"Failed to pull changes: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def push(self, remote: str = "origin", branch: Optional[str] = None, 
            repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Push changes to a remote repository.
        
        Args:
            remote: Remote name
            branch: Branch name (default: current branch)
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Push using GitPython
                if branch:
                    result = repo.git.push(remote, branch)
                else:
                    result = repo.git.push(remote)
                
                logger.info(f"Pushed changes to {remote}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "remote": remote,
                    "branch": branch,
                    "result": result
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                cmd = ["git", "-C", repo_path, "push", remote]
                if branch:
                    cmd.append(branch)
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git push failed: {result.stderr}"
                    }
                
                logger.info(f"Pushed changes to {remote}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "remote": remote,
                    "branch": branch,
                    "result": result.stdout
                }
                
        except Exception as e:
            logger.error(f"Failed to push changes: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_commit_history(self, max_count: int = 10, 
                          repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get commit history of a repository.
        
        Args:
            max_count: Maximum number of commits to retrieve
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with commit history
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Get history using GitPython
                commits = []
                for commit in repo.iter_commits(max_count=max_count):
                    commits.append({
                        "hash": commit.hexsha,
                        "author": commit.author.name,
                        "email": commit.author.email,
                        "date": commit.committed_datetime.isoformat(),
                        "message": commit.message.strip()
                    })
                
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "commits": commits
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                cmd = ["git", "-C", repo_path, "log", f"--max-count={max_count}", 
                      "--pretty=format:%H|%an|%ae|%aI|%s"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git log failed: {result.stderr}"
                    }
                
                # Parse output
                commits = []
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                        
                    parts = line.split('|')
                    if len(parts) >= 5:
                        commits.append({
                            "hash": parts[0],
                            "author": parts[1],
                            "email": parts[2],
                            "date": parts[3],
                            "message": parts[4]
                        })
                
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "commits": commits
                }
                
        except Exception as e:
            logger.error(f"Failed to get commit history: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_branch(self, branch_name: str, start_point: Optional[str] = None, 
                     repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new branch.
        
        Args:
            branch_name: Name of the new branch
            start_point: Starting point for the branch (default: HEAD)
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Create branch using GitPython
                if start_point:
                    repo.git.branch(branch_name, start_point)
                else:
                    repo.git.branch(branch_name)
                
                logger.info(f"Created branch: {branch_name}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "branch": branch_name,
                    "start_point": start_point or "HEAD"
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                cmd = ["git", "-C", repo_path, "branch", branch_name]
                if start_point:
                    cmd.append(start_point)
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git branch failed: {result.stderr}"
                    }
                
                logger.info(f"Created branch: {branch_name}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "branch": branch_name,
                    "start_point": start_point or "HEAD"
                }
                
        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def merge_branch(self, branch: str, message: Optional[str] = None, 
                    repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Merge a branch into the current branch.
        
        Args:
            branch: Branch to merge
            message: Merge commit message (default: auto-generated)
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Merge using GitPython
                if message:
                    result = repo.git.merge(branch, m=message)
                else:
                    result = repo.git.merge(branch)
                
                logger.info(f"Merged branch: {branch}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "branch": branch,
                    "result": result
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                cmd = ["git", "-C", repo_path, "merge", branch]
                if message:
                    cmd.extend(["-m", message])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git merge failed: {result.stderr}"
                    }
                
                logger.info(f"Merged branch: {branch}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "branch": branch,
                    "result": result.stdout
                }
                
        except Exception as e:
            logger.error(f"Failed to merge branch: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_tag(self, tag_name: str, message: Optional[str] = None, 
                  commit: Optional[str] = None, repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a tag.
        
        Args:
            tag_name: Name of the tag
            message: Tag message
            commit: Commit to tag (default: HEAD)
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Create tag using GitPython
                if message:
                    if commit:
                        repo.git.tag(tag_name, commit, m=message)
                    else:
                        repo.git.tag(tag_name, m=message)
                else:
                    if commit:
                        repo.git.tag(tag_name, commit)
                    else:
                        repo.git.tag(tag_name)
                
                logger.info(f"Created tag: {tag_name}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "tag": tag_name,
                    "commit": commit or "HEAD"
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                cmd = ["git", "-C", repo_path, "tag"]
                if message:
                    cmd.extend(["-a", tag_name, "-m", message])
                else:
                    cmd.append(tag_name)
                
                if commit:
                    cmd.append(commit)
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git tag failed: {result.stderr}"
                    }
                
                logger.info(f"Created tag: {tag_name}")
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "tag": tag_name,
                    "commit": commit or "HEAD"
                }
                
        except Exception as e:
            logger.error(f"Failed to create tag: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_diff(self, file_path: Optional[str] = None, 
                repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get diff for a file or the entire repository.
        
        Args:
            file_path: Path to the file (default: all files)
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with diff information
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Get diff using GitPython
                if file_path:
                    diff = repo.git.diff(file_path)
                else:
                    diff = repo.git.diff()
                
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "file_path": file_path,
                    "diff": diff
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                cmd = ["git", "-C", repo_path, "diff"]
                if file_path:
                    cmd.append(file_path)
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git diff failed: {result.stderr}"
                    }
                
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "file_path": file_path,
                    "diff": result.stdout
                }
                
        except Exception as e:
            logger.error(f"Failed to get diff: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_branches(self, repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get list of branches in the repository.
        
        Args:
            repo_name: Name of the repository (default: active repository)
            
        Returns:
            Dictionary with branch information
        """
        try:
            # Get repo
            repo = self._get_repo(repo_name)
            
            if not repo:
                return {
                    "success": False,
                    "error": "No active repository"
                }
            
            if self.git_python_available and isinstance(repo, git.Repo):
                # Get branches using GitPython
                branches = []
                for branch in repo.branches:
                    branches.append({
                        "name": branch.name,
                        "commit": branch.commit.hexsha,
                        "is_active": branch.name == repo.active_branch.name
                    })
                
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "branches": branches,
                    "active_branch": repo.active_branch.name
                }
            
            else:
                # Fallback to command line git
                repo_path = repo['path'] if isinstance(repo, dict) else repo.working_dir
                
                # Get all branches
                cmd = ["git", "-C", repo_path, "branch", "--format=%(refname:short)|%(objectname)|%(HEAD)"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Git branch failed: {result.stderr}"
                    }
                
                # Parse output
                branches = []
                active_branch = None
                
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                        
                    parts = line.split('|')
                    if len(parts) >= 3:
                        name = parts[0]
                        commit = parts[1]
                        is_active = parts[2] == '*'
                        
                        branches.append({
                            "name": name,
                            "commit": commit,
                            "is_active": is_active
                        })
                        
                        if is_active:
                            active_branch = name
                
                return {
                    "success": True,
                    "repo_name": repo_name or self.active_repo_name,
                    "branches": branches,
                    "active_branch": active_branch
                }
                
        except Exception as e:
            logger.error(f"Failed to get branches: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_repo(self, repo_name: Optional[str] = None):
        """Get repository object by name or active repository."""
        if repo_name:
            if repo_name in self.repos:
                return self.repos[repo_name]
            else:
                logger.error(f"Repository not found: {repo_name}")
                return None
        else:
            if self.active_repo:
                return self.active_repo
            else:
                logger.error("No active repository")
                return None

# Example usage
if __name__ == "__main__":
    git_integration = GitIntegration()
    
    # Clone a repository
    result = git_integration.clone_repository("https://github.com/example/repo.git", "example-repo")
    print(result)
    
    # Get status
    status = git_integration.get_status("example-repo")
    print(status)
    
    # Create a new branch
    branch_result = git_integration.create_branch("feature-branch", repo_name="example-repo")
    print(branch_result)
    
    # Checkout the branch
    checkout_result = git_integration.checkout("feature-branch", repo_name="example-repo")
    print(checkout_result)
    
    # Make changes and commit
    # (This would typically involve file operations)
    add_result = git_integration.add_files(["README.md"], repo_name="example-repo")
    print(add_result)
    
    commit_result = git_integration.commit("Update README", repo_name="example-repo")
    print(commit_result)