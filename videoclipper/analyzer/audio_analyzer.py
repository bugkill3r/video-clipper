"""Audio analysis for identifying interesting audio segments in videos."""

from typing import List, Optional

import numpy as np
from pydub import AudioSegment

from videoclipper.analyzer.base import Analyzer
from videoclipper.exceptions import AnalysisError
from videoclipper.models.segment import Segment, SegmentType


class AudioAnalyzer(Analyzer):
    """Analyzes audio to find potential highlights based on energy levels."""

    def __init__(self, audio_path: str) -> None:
        """Initialize the audio analyzer.

        Args:
            audio_path: Path to the audio file
        """
        super().__init__(audio_path)
        self._audio: Optional[AudioSegment] = None
        self._load_audio()

    def _load_audio(self) -> None:
        """Load the audio file.

        Raises:
            AnalysisError: If audio loading fails
        """
        try:
            self._audio = AudioSegment.from_file(self.file_path)
        except Exception as e:
            raise AnalysisError(f"Failed to load audio file {self.file_path}: {e}")

    def analyze_energy(
        self, chunk_ms: int = 1000, percentile_threshold: float = 75
    ) -> List[Segment]:
        """Analyze audio energy to find potential highlights.

        Args:
            chunk_ms: Chunk size in milliseconds for energy analysis
            percentile_threshold: Percentile threshold for identifying high energy moments

        Returns:
            List of segments representing high energy parts of the audio

        Raises:
            AnalysisError: If audio analysis fails
        """
        if not self._audio:
            raise AnalysisError("Audio not loaded")

        try:
            # Split audio into chunks and get their dBFS (energy)
            chunk_length_ms = chunk_ms
            audio_duration_sec = len(self._audio) / 1000.0  # in seconds
            energy_profile = []

            for i in range(0, len(self._audio), chunk_length_ms):
                chunk = self._audio[i:i+chunk_length_ms]
                if len(chunk) > 0:
                    energy_profile.append((i / 1000.0, chunk.dBFS))

            # Find peaks in energy (potential exciting moments)
            # Use percentile to determine threshold
            energies = [e[1] for e in energy_profile if e[1] > float('-inf')]
            if not energies:
                return []

            threshold = np.percentile(energies, percentile_threshold)
            high_energy_times = [time for time, energy in energy_profile if energy >= threshold]

            # Group nearby energy peaks (within 3 seconds)
            energy_highlights = []
            if high_energy_times:
                current_group = [high_energy_times[0]]

                for i in range(1, len(high_energy_times)):
                    # If this peak is within 3 seconds of the last one in current group
                    if high_energy_times[i] - current_group[-1] < 3:
                        current_group.append(high_energy_times[i])
                    else:
                        # Calculate the average timestamp for this group
                        avg_time = sum(current_group) / len(current_group)
                        energy_highlights.append(avg_time)
                        current_group = [high_energy_times[i]]

                # Add the last group
                if current_group:
                    avg_time = sum(current_group) / len(current_group)
                    energy_highlights.append(avg_time)

            # Convert energy highlights to segments
            segments = []
            for i, timestamp in enumerate(energy_highlights):
                # Create a segment of 5 seconds, centered on the energy peak
                # unless the peak is near the beginning or end of the audio
                start = max(0, timestamp - 2.5)
                end = min(audio_duration_sec, timestamp + 2.5)

                segment = Segment(
                    start=start,
                    end=end,
                    score=0.7,  # Default score for energy-based segments
                    segment_type=SegmentType.ENERGY_PEAK,
                    text=f"Energy peak at {timestamp:.2f}s",
                    metadata={
                        "energy_index": i,
                        "peak_time": timestamp,
                    },
                )
                segments.append(segment)

            self.segments = segments
            return segments

        except Exception as e:
            raise AnalysisError(f"Audio analysis failed: {e}")

    def analyze(self, **kwargs) -> List[Segment]:
        """Run audio energy analysis.

        Args:
            **kwargs: Keyword arguments passed to analyze_energy

        Returns:
            List of audio energy segments
        """
        return self.analyze_energy(**kwargs)
