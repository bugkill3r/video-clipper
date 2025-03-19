"""
Custom exceptions for the videoclipper package.
"""

class VideoClipperError(Exception):
    """Base exception for the videoclipper package."""
    pass


class FileError(VideoClipperError):
    """Exception raised for file-related errors."""
    pass


class ValidationError(VideoClipperError):
    """Exception raised for validation errors."""
    pass


class TranscriptionError(VideoClipperError):
    """Exception raised for transcription-related errors."""
    pass


class AnalysisError(VideoClipperError):
    """Exception raised for analysis-related errors."""
    pass


class ClippingError(VideoClipperError):
    """Exception raised for clipping-related errors."""
    pass