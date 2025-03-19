"""Base transcriber class for speech-to-text components."""

from abc import ABC, abstractmethod
from typing import List, Optional

from videoclipper.models.segment import Segment


class Transcriber(ABC):
    """Base class for transcriber components.

    A transcriber converts speech to text and generates meaningful segments.
    """

    def __init__(self, audio_path: str) -> None:
        """Initialize the transcriber with the path to the audio file.

        Args:
            audio_path: Path to the audio file
        """
        self.audio_path = audio_path
        self.segments: List[Segment] = []

    @abstractmethod
    def transcribe(self, **kwargs) -> List[Segment]:
        """Transcribe the audio to generate text segments.

        Returns:
            List of segments with transcribed text
        """
        pass

    def get_segments(self) -> List[Segment]:
        """Get the segments identified by this transcriber.

        Returns:
            List of identified segments
        """
        return self.segments
