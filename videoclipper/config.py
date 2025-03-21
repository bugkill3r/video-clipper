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
    "default_highlight_duration": 45,  # seconds (target for viral clips)
    "min_segment_duration": 3,  # seconds
    "max_segment_duration": 45,  # seconds (longer segments for viral clips)
    "min_spacing_between_segments": 10.0,  # seconds
    "num_video_zones": 12,  # number of zones to divide video into for segment selection
    "max_words_per_caption_line": 3,  # Maximum words per caption line
    "max_caption_lines": 2,  # Maximum number of caption lines

    # Caption styling
    "caption_position": 0.66,  # Position from top (0.66 = bottom third)
    "word_spacing": 0,  # Spacing between words in pixels
    "caption_highlight_color": "#00FF00",  # Bright green for highlighted words
    "caption_normal_color": "white",  # Color for non-highlighted words
    "crossfade_duration": 0.5,  # seconds for transitions between clips

    # Audio/speech detection
    "speech_preroll_ms": 100,  # ms before speech to show captions
    "speech_postroll_ms": 50,  # ms after speech to keep captions visible
    "min_speech_segment_duration": 250,  # ms minimum speech segment duration
    "min_gap_between_speech": 150,  # ms minimum gap between speech segments
    "max_gap_for_caption_merge": 200,  # ms maximum gap to merge caption segments

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
