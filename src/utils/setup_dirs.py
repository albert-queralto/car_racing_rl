import os
from pathlib import Path

def create_directory(directory: Path) -> None:
    """Creates a directory if it doesn't exist."""
    if not directory.exists():
        os.makedirs(directory)

def setup_dirs(path: Path) -> None:
    """Sets up the directories required for the training and testing."""
    VIDEOS_DIR = path / 'videos'
    LOGS_DIR = path / 'logs'
    MODELS_DIR = path / 'models'

    create_directory(VIDEOS_DIR)
    create_directory(LOGS_DIR)
    create_directory(MODELS_DIR)
