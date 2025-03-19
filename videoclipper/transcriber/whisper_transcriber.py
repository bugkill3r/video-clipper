"""Speech transcription using OpenAI's Whisper model."""

import os
from typing import Dict, List, Optional, Union

import whisper

from videoclipper.exceptions import TranscriptionError
from videoclipper.models.segment import Segment, SegmentType
from videoclipper.transcriber.base import Transcriber
from videoclipper.utils.validation import validate_model_name


class WhisperTranscriber(Transcriber):
    """Transcribes audio using OpenAI's Whisper model."""

    # Available Whisper model sizes
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, audio_path: str) -> None:
        """Initialize the Whisper transcriber.

        Args:
            audio_path: Path to the audio file
        """
        super().__init__(audio_path)
        self._model: Optional[whisper.Whisper] = None
        self._model_name: Optional[str] = None

    def _load_model(self, model_size: str = "base") -> whisper.Whisper:
        """Load the Whisper model.

        Args:
            model_size: Size of the Whisper model to use

        Returns:
            Loaded Whisper model

        Raises:
            TranscriptionError: If model loading fails
        """
        try:
            model_size = validate_model_name(model_size, self.AVAILABLE_MODELS)

            # Only load if not already loaded or if model size has changed
            if self._model is None or self._model_name != model_size:
                self._model = whisper.load_model(model_size)
                self._model_name = model_size

            return self._model
        except Exception as e:
            raise TranscriptionError(f"Failed to load Whisper model: {e}")

    def transcribe(
        self, model_size: str = "base", min_segment_length: float = 1.0
    ) -> List[Segment]:
        """Transcribe audio using Whisper.

        Args:
            model_size: Size of the Whisper model to use
            min_segment_length: Minimum segment length in seconds

        Returns:
            List of segments containing transcribed text

        Raises:
            TranscriptionError: If transcription fails
        """
        try:
            model = self._load_model(model_size)

            # Transcribe audio
            result = model.transcribe(self.audio_path)

            # Convert Whisper segments to our Segment model
            segments = []
            for idx, segment in enumerate(result["segments"]):
                # Skip very short segments
                if segment["end"] - segment["start"] < min_segment_length:
                    continue

                # Create segment
                new_segment = Segment(
                    start=segment["start"],
                    end=segment["end"],
                    score=min(1.0, float(segment.get("confidence", 0.5))),
                    segment_type=SegmentType.SPEECH,
                    text=segment["text"].strip(),
                    metadata={
                        "segment_id": idx,
                        "language": result.get("language", "unknown"),
                    },
                )
                segments.append(new_segment)

            self.segments = segments
            return segments

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}")
