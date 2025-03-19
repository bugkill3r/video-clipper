"""Base analyzer class for video analysis components."""

from abc import ABC, abstractmethod
from typing import List, Optional

from videoclipper.models.segment import Segment


class Analyzer(ABC):
    """Base class for analyzer components.

    An analyzer takes a video file and generates meaningful segments.
    """

    def __init__(self, file_path: str) -> None:
        """Initialize the analyzer with the path to the file to analyze.

        Args:
            file_path: Path to the file to analyze
        """
        self.file_path = file_path
        self.segments: List[Segment] = []

    @abstractmethod
    def analyze(self, **kwargs) -> List[Segment]:
        """Run analysis on the file to generate segments.

        Returns:
            List of identified segments
        """
        pass

    def get_segments(self) -> List[Segment]:
        """Get the segments identified by this analyzer.

        Returns:
            List of identified segments
        """
        return self.segments
