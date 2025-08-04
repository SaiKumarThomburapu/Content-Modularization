import os
from pathlib import Path
import time
import uuid

def generate_unique_filename(input_path: str, output_folder: str, prefix: str = "", extension: str = ".json") -> str:
    """
    Generates a unique filename for processed content results.
    
    Parameters:
        input_path (str): The path of the original input.
        output_folder (str): The folder where results will be saved.
        prefix (str): Prefix for the filename.
        extension (str): File extension.
    
    Returns:
        str: A unique output file path.
    """
    input_stem = Path(input_path).stem if input_path else "content"
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    unique_name = f"{prefix}{input_stem}_{timestamp}_{unique_id}{extension}"
    return os.path.join(output_folder, unique_name)

def create_directories(directories: list):
    """Create multiple directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
