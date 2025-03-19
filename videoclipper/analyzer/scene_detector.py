"""Scene detection for identifying transitions in videos."""

import cv2
import numpy as np
from typing import List, Optional

from videoclipper.analyzer.base import Analyzer
from videoclipper.exceptions import AnalysisError
from videoclipper.models.segment import Segment, SegmentType


class SceneDetector(Analyzer):
    """Detects scene changes in a video based on frame differences."""

    def __init__(self, file_path: str) -> None:
        """Initialize the scene detector.

        Args:
            file_path: Path to the video file
        """
        super().__init__(file_path)
        self.threshold = 30.0  # Default threshold for scene detection

    def detect_scenes(self, threshold: Optional[float] = None) -> List[Segment]:
        """Detect scenes in the video.

        Args:
            threshold: Threshold for frame difference (higher = fewer scenes)

        Returns:
            List of segments representing scene changes

        Raises:
            AnalysisError: If scene detection fails
        """
        if threshold is not None:
            self.threshold = threshold

        try:
            cap = cv2.VideoCapture(self.file_path)
            if not cap.isOpened():
                raise AnalysisError(f"Failed to open video file: {self.file_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            prev_frame = None
            frame_count = 0
            scene_changes = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    # Convert frames to grayscale
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

                    # Calculate difference between frames
                    frame_diff = cv2.absdiff(gray_frame, gray_prev)
                    mean_diff = np.mean(frame_diff)

                    # If difference exceeds threshold, mark as scene change
                    if mean_diff > self.threshold:
                        timestamp = frame_count / fps
                        scene_changes.append(timestamp)

                prev_frame = frame
                frame_count += 1

            cap.release()

            # Convert scene changes to segments
            segments = []

            for i, timestamp in enumerate(scene_changes):
                # Create a segment of 5 seconds starting from the scene change
                # unless we're near the end of the video
                start = timestamp
                end = min(duration, timestamp + 5.0)

                segment = Segment(
                    start=start,
                    end=end,
                    score=0.6,  # Default score for scene changes
                    segment_type=SegmentType.SCENE_CHANGE,
                    text=f"Scene change at {timestamp:.2f}s",
                    metadata={
                        "frame_index": int(timestamp * fps),
                        "scene_index": i,
                    },
                )
                segments.append(segment)

            self.segments = segments
            return segments

        except Exception as e:
            raise AnalysisError(f"Scene detection failed: {e}")

    def analyze(self, **kwargs) -> List[Segment]:
        """Run scene detection analysis.

        Args:
            **kwargs: Keyword arguments passed to detect_scenes

        Returns:
            List of scene change segments
        """
        return self.detect_scenes(**kwargs)
