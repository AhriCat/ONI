import unittest
import os
import tempfile
import shutil
from pathlib import Path
from tools.git_integration import GitIntegration

class TestGitIntegration(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.git_integration = GitIntegration(work_dir=self.test_dir)
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test that GitIntegration initializes correctly."""
        self.assertEqual(self.git_integration.work_dir, self.test_dir)
        self.assertIsNone(self.git_integration.active_repo)
        self.assertIsNone(self.git_integration.active_repo_name)
        
    def test_init_repository(self):
        """Test initializing a new Git repository."""
        result = self.git_integration.init_repository("test-repo")
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["repo_name"], "test-repo")
        
        # Check that repository was created
        repo_path = os.path.join(self.test_dir, "test-repo")
        self.assertTrue(os.path.exists(repo_path))
        self.assertTrue(os.path.exists(os.path.join(repo_path, ".git")))
        
        # Check that repository is active
        self.assertEqual(self.git_integration.active_repo_name, "test-repo")
        
    def test_set_active_repository(self):
        """Test setting the active repository."""
        # Create two repositories
        self.git_integration.init_repository("repo1")
        self.git_integration.init_repository("repo2")
        
        # Set active repository
        result = self.git_integration.set_active_repository("repo1")
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["repo_name"], "repo1")
        
        # Check that repository is active
        self.assertEqual(self.git_integration.active_repo_name, "repo1")
        
    def test_get_status(self):
        """Test getting repository status."""
        # Initialize repository
        self.git_integration.init_repository("test-repo")
        
        # Get status
        result = self.git_integration.get_status()
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["repo_name"], "test-repo")
        
        # Status should include branch information
        self.assertIn("branch", result)
        
    def test_add_files(self):
        """Test adding files to the Git index."""
        # Initialize repository
        self.git_integration.init_repository("test-repo")
        
        # Create a test file
        repo_path = os.path.join(self.test_dir, "test-repo")
        test_file = os.path.join(repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test content")
        
        # Add file
        result = self.git_integration.add_files(["test.txt"])
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["files"], ["test.txt"])
        
    def test_commit(self):
        """Test committing changes."""
        # Initialize repository
        self.git_integration.init_repository("test-repo")
        
        # Create and add a test file
        repo_path = os.path.join(self.test_dir, "test-repo")
        test_file = os.path.join(repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test content")
        
        self.git_integration.add_files(["test.txt"])
        
        # Commit changes
        result = self.git_integration.commit("Initial commit")
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Initial commit")
        
    def test_checkout(self):
        """Test checking out a branch."""
        # Initialize repository
        self.git_integration.init_repository("test-repo")
        
        # Create and commit a test file
        repo_path = os.path.join(self.test_dir, "test-repo")
        test_file = os.path.join(repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test content")
        
        self.git_integration.add_files(["test.txt"])
        self.git_integration.commit("Initial commit")
        
        # Create a new branch
        result = self.git_integration.create_branch("feature-branch")
        self.assertTrue(result["success"])
        
        # Checkout the branch
        result = self.git_integration.checkout("feature-branch")
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["branch"], "feature-branch")
        
    def test_get_branches(self):
        """Test getting list of branches."""
        # Initialize repository
        self.git_integration.init_repository("test-repo")
        
        # Create and commit a test file
        repo_path = os.path.join(self.test_dir, "test-repo")
        test_file = os.path.join(repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test content")
        
        self.git_integration.add_files(["test.txt"])
        self.git_integration.commit("Initial commit")
        
        # Create a new branch
        self.git_integration.create_branch("feature-branch")
        
        # Get branches
        result = self.git_integration.get_branches()
        
        # Check result
        self.assertTrue(result["success"])
        self.assertIn("branches", result)
        
        # Should have at least two branches (main/master and feature-branch)
        self.assertGreaterEqual(len(result["branches"]), 2)
        
        # Check that feature-branch exists
        branch_names = [branch["name"] for branch in result["branches"]]
        self.assertIn("feature-branch", branch_names)
        
    def test_get_diff(self):
        """Test getting diff for a file."""
        # Initialize repository
        self.git_integration.init_repository("test-repo")
        
        # Create and commit a test file
        repo_path = os.path.join(self.test_dir, "test-repo")
        test_file = os.path.join(repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("Initial content")
        
        self.git_integration.add_files(["test.txt"])
        self.git_integration.commit("Initial commit")
        
        # Modify the file
        with open(test_file, "w") as f:
            f.write("Modified content")
        
        # Get diff
        result = self.git_integration.get_diff("test.txt")
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["file_path"], "test.txt")
        self.assertIn("diff", result)
        
        # Diff should contain the changes
        self.assertIn("Modified content", result["diff"])
        
    def test_create_tag(self):
        """Test creating a tag."""
        # Initialize repository
        self.git_integration.init_repository("test-repo")
        
        # Create and commit a test file
        repo_path = os.path.join(self.test_dir, "test-repo")
        test_file = os.path.join(repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test content")
        
        self.git_integration.add_files(["test.txt"])
        self.git_integration.commit("Initial commit")
        
        # Create tag
        result = self.git_integration.create_tag("v1.0.0", message="Version 1.0.0")
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["tag"], "v1.0.0")

if __name__ == '__main__':
    unittest.main()