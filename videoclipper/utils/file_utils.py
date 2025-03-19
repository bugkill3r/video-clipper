"""Utility functions for file operations."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from videoclipper.exceptions import FileError


def ensure_directory(directory: str) -> str:
    """Create directory if it doesn't exist.

    Args:
        directory: Path to the directory

    Returns:
        The directory path

    Raises:
        FileError: If directory creation fails
    """
    if not directory:
        return directory

    try:
        os.makedirs(directory, exist_ok=True)
        return directory
    except Exception as e:
        raise FileError(f"Failed to create directory {directory}: {e}")


def temp_directory() -> str:
    """Create a temporary directory.

    Returns:
        Path to the temporary directory
    """
    return tempfile.mkdtemp()


def temp_audio_path() -> str:
    """Get a path for a temporary audio file.

    Returns:
        Path to a temporary audio file
    """
    temp_dir = temp_directory()
    return os.path.join(temp_dir, "audio.wav")


def list_files(directory: str, extension: Optional[str] = None) -> List[str]:
    """List files in a directory, optionally filtered by extension.

    Args:
        directory: Directory to list files from
        extension: Optional file extension to filter by (e.g., '.mp4')

    Returns:
        List of file paths

    Raises:
        FileError: If directory doesn't exist or isn't readable
    """
    try:
        if not os.path.isdir(directory):
            raise FileError(f"Directory does not exist: {directory}")

        files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                if extension is None or file.endswith(extension):
                    files.append(file_path)
        return files
    except Exception as e:
        raise FileError(f"Error listing files in {directory}: {e}")


def get_file_extension(file_path: str) -> str:
    """Get the extension of a file.

    Args:
        file_path: Path to the file

    Returns:
        File extension with leading dot (e.g., '.mp4')
    """
    return os.path.splitext(file_path)[1].lower()


def clean_temp_files(directory: str) -> None:
    """Remove temporary files and directories.

    Args:
        directory: Directory to clean up

    Raises:
        FileError: If cleanup fails
    """
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    except Exception as e:
        raise FileError(f"Failed to clean up temporary directory {directory}: {e}")