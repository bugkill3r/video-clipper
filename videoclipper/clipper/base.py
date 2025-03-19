"""Base video clipper class."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from videoclipper.models.segment import Segment


class VideoClipper(ABC):
    """Base class for video clipping components.

    A video clipper takes segments and creates a final highlight video.
    """

    def __init__(self, video_path: str) -> None:
        """Initialize the video clipper with the path to the video file.

        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path

    @abstractmethod
    def create_clip(
        self, segments: List[Segment], output_path: str, max_duration: Optional[float] = None
    ) -> Tuple[str, float]:
        """Create a video clip from the given segments.

        Args:
            segments: List of segments to include in the clip
            output_path: Path to save the output video
            max_duration: Maximum duration of the output clip in seconds

        Returns:
            Tuple of (output_path, duration)
        """
        pass
