"""Configuration settings for the VideoClipper application."""

import os
from pathlib import Path
from typing import Dict, Optional

# Default configuration
DEFAULT_CONFIG: Dict[str, any] = {
    # General settings
    "output_dir": "output",
    "temp_dir": None,  # Use system temp by default

    # Video processing
    "default_output_format": "mp4",
    "default_codec": "libx264",
    "default_audio_codec": "aac",

    # Highlight generation
    "default_highlight_duration": 60,  # seconds
    "min_segment_duration": 3,  # seconds
    "max_segment_duration": 15,  # seconds

    # Analysis settings
    "scene_detection_threshold": 30.0,
    "audio_energy_percentile": 75,
    "transcribe_by_default": True,
    "default_whisper_model": "base",

    # Advanced settings
    "ffmpeg_threads": 2,
}


def get_config(key: str, default: Optional[any] = None) -> any:
    """Get configuration value by key.

    Args:
        key: Configuration key
        default: Default value if key not found

    Returns:
        Configuration value
    """
    # Environment variables override defaults
    env_key = f"VIDEOCLIPPER_{key.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]

    # Otherwise use defaults
    return DEFAULT_CONFIG.get(key, default)


def get_output_dir() -> str:
    """Get the configured output directory.

    Returns:
        Path to output directory
    """
    output_dir = get_config("output_dir")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_temp_dir() -> str:
    """Get the configured temporary directory.

    Returns:
        Path to temporary directory
    """
    temp_dir = get_config("temp_dir")
    if not temp_dir:
        import tempfile
        temp_dir = tempfile.gettempdir()

    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir
