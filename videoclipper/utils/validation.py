"""Validation utilities for VideoClipper."""

import os
from typing import List, Optional, Union

from videoclipper.exceptions import ValidationError


def validate_video_file(file_path: str) -> str:
    """Validate that a file exists and appears to be a video file.

    Args:
        file_path: Path to the file to validate

    Returns:
        The validated file path

    Raises:
        ValidationError: If the file doesn't exist or doesn't look like a video
    """
    if not os.path.exists(file_path):
        raise ValidationError(f"File does not exist: {file_path}")

    if not os.path.isfile(file_path):
        raise ValidationError(f"Not a file: {file_path}")

    # Check file extension
    valid_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in valid_extensions:
        raise ValidationError(
            f"File doesn't appear to be a video. Extension {ext} not in {valid_extensions}"
        )

    return file_path


def validate_output_path(file_path: str) -> str:
    """Validate that an output path is writable.

    Args:
        file_path: Path to validate

    Returns:
        The validated output path

    Raises:
        ValidationError: If the output directory isn't writable
    """
    directory = os.path.dirname(file_path) or "."

    if os.path.exists(directory) and not os.path.isdir(directory):
        raise ValidationError(f"Output directory path exists but is not a directory: {directory}")

    if os.path.exists(directory) and not os.access(directory, os.W_OK):
        raise ValidationError(f"Output directory is not writable: {directory}")

    # Try to create the directory if it doesn't exist
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            raise ValidationError(f"Failed to create output directory: {e}")

    return file_path


def validate_time_range(
    start_time: float, end_time: float, max_duration: Optional[float] = None
) -> tuple[float, float]:
    """Validate a time range is sensible.

    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        max_duration: Optional maximum duration

    Returns:
        Tuple of (start_time, end_time)

    Raises:
        ValidationError: If the time range is invalid
    """
    if start_time < 0:
        raise ValidationError(f"Start time cannot be negative: {start_time}")

    if end_time <= start_time:
        raise ValidationError(
            f"End time ({end_time}) must be greater than start time ({start_time})"
        )

    if max_duration is not None and end_time - start_time > max_duration:
        raise ValidationError(
            f"Duration ({end_time - start_time}) exceeds maximum allowed ({max_duration})"
        )

    return (start_time, end_time)


def validate_model_name(model_name: str, allowed_models: List[str]) -> str:
    """Validate that a model name is in the list of allowed models.

    Args:
        model_name: Name of the model to validate
        allowed_models: List of allowed model names

    Returns:
        The validated model name

    Raises:
        ValidationError: If the model name is not in the allowed list
    """
    if model_name not in allowed_models:
        raise ValidationError(f"Invalid model name: {model_name}. Must be one of {allowed_models}")

    return model_name
