"""
Centralized configuration for Oni system
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configurations
DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 896,
    "num_heads": 8,
    "num_layers": 6,
    "vocab_size": 300000,
    "max_length": 4096
}

# Memory configuration
MEMORY_CONFIG = {
    "working_memory_capacity": 5,
    "ltm_capacity": 10_000_000_000_000,
    "episodic_memory_path": DATA_DIR / "episodes",
    "semantic_memory_path": DATA_DIR / "semantic_memory.json",
    "ltm_summary_path": DATA_DIR / "ltm_data.json"
}

# API Keys (from environment variables)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "oni.log"
}