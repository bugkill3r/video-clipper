"""Utility functions for file operations."""

import os
import json
import shutil
import tempfile
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

from videoclipper.exceptions import FileError
from videoclipper.models.segment import Segment


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


def save_segments(segments: List[Segment], video_id: str) -> bool:
    """Save segments to a cache file specific to the video ID.
    
    Args:
        segments: List of segments to cache
        video_id: YouTube video ID to use as a key
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create the cache directory structure
        cache_dir = os.path.join("downloads", video_id, "cache")
        ensure_directory(cache_dir)
        
        # Create cache file path
        cache_file = os.path.join(cache_dir, "segments.pkl")
        
        # Save segments using pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(segments, f)
            
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to save segments for video {video_id}: {e}")
        return False


def load_segments(video_id: str) -> Optional[List[Segment]]:
    """Load cached segments for a specific video ID.
    
    Args:
        video_id: YouTube video ID to load segments for
        
    Returns:
        List of segments if cached, None otherwise
    """
    try:
        # Check if cache file exists
        cache_file = os.path.join("downloads", video_id, "cache", "segments.pkl")
        if not os.path.exists(cache_file):
            return None
            
        # Load segments from cache
        with open(cache_file, 'rb') as f:
            segments = pickle.load(f)
            
        return segments
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to load segments for video {video_id}: {e}")
        return None


def has_cached_segments(video_id: str) -> bool:
    """Check if segments are cached for a specific video ID.
    
    Args:
        video_id: YouTube video ID to check
        
    Returns:
        True if segments are cached, False otherwise
    """
    cache_file = os.path.join("downloads", video_id, "cache", "segments.pkl")
    return os.path.exists(cache_file)


def save_video_metadata(video_id: str, metadata: Dict[str, Any]) -> bool:
    """Save video metadata to a cache file.
    
    Args:
        video_id: YouTube video ID to use as a key
        metadata: Dictionary of metadata to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create the cache directory structure
        cache_dir = os.path.join("downloads", video_id, "cache")
        ensure_directory(cache_dir)
        
        # Create cache file path
        cache_file = os.path.join(cache_dir, "metadata.json")
        
        # Save metadata using JSON
        with open(cache_file, 'w') as f:
            json.dump(metadata, f)
            
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to save metadata for video {video_id}: {e}")
        return False


def load_video_metadata(video_id: str) -> Optional[Dict[str, Any]]:
    """Load cached metadata for a specific video ID.
    
    Args:
        video_id: YouTube video ID to load metadata for
        
    Returns:
        Dictionary of metadata if cached, None otherwise
    """
    try:
        # Check if cache file exists
        cache_file = os.path.join("downloads", video_id, "cache", "metadata.json")
        if not os.path.exists(cache_file):
            return None
            
        # Load metadata from cache
        with open(cache_file, 'r') as f:
            metadata = json.load(f)
            
        return metadata
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to load metadata for video {video_id}: {e}")
        return None