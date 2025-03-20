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
            frame_differences = []  # Store frame differences for dynamic scoring
            
            # Variables for motion detection
            motion_scores = []
            window_size = int(fps * 1.5)  # 1.5 second window
            motion_history = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    # Convert frames to grayscale
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

                    # Calculate difference between frames for scene detection
                    frame_diff = cv2.absdiff(gray_prev, gray_frame)
                    mean_diff = np.mean(frame_diff)
                    frame_differences.append(mean_diff)
                    
                    # Track motion for the current window
                    motion_history.append(mean_diff)
                    if len(motion_history) > window_size:
                        motion_history.pop(0)
                    
                    # Calculate motion score for this frame
                    if len(motion_history) >= window_size // 2:
                        motion_score = sum(motion_history) / len(motion_history)
                        motion_scores.append((frame_count, motion_score))
                    
                    # If difference exceeds threshold, mark as scene change
                    if mean_diff > self.threshold:
                        timestamp = frame_count / fps
                        scene_changes.append((timestamp, mean_diff))

                prev_frame = frame
                frame_count += 1

            cap.release()

            # Find the 95th percentile for better scoring
            if frame_differences:
                high_threshold = np.percentile(frame_differences, 95)
            else:
                high_threshold = self.threshold * 1.5
                
            # Sort motion scores to find high-motion segments
            motion_scores.sort(key=lambda x: x[1], reverse=True)
            high_motion_frames = [frame for frame, score in motion_scores[:min(20, len(motion_scores))]]
            
            # Convert scene changes to segments
            segments = []

            # Add segments for scene changes
            for i, (timestamp, diff_score) in enumerate(scene_changes):
                # Calculate score based on the frame difference
                base_score = min(1.0, diff_score / high_threshold)
                score = 0.5 + (base_score * 0.5)  # Scale between 0.5 and 1.0
                
                # Create a segment of variable length based on score
                segment_duration = 5.0 + (score * 5.0)  # 5-10 seconds based on score
                start = max(0, timestamp - 0.5)  # Start slightly before the scene change
                end = min(duration, start + segment_duration)

                segment = Segment(
                    start=start,
                    end=end,
                    score=score,
                    segment_type=SegmentType.SCENE_CHANGE,
                    text=f"Scene change at {timestamp:.2f}s",
                    metadata={
                        "frame_index": int(timestamp * fps),
                        "scene_index": i,
                        "diff_score": diff_score,
                    },
                )
                segments.append(segment)
                
            # Add segments for high motion areas
            for frame_idx in high_motion_frames:
                timestamp = frame_idx / fps
                
                # Skip if too close to an existing scene change
                if any(abs(timestamp - seg.start) < 5.0 for seg in segments):
                    continue
                    
                start = max(0, timestamp - 1.0)
                end = min(duration, start + 7.0)
                
                segment = Segment(
                    start=start,
                    end=end,
                    score=0.75,  # High score for motion segments
                    segment_type=SegmentType.ENERGY_PEAK,
                    text=f"High motion at {timestamp:.2f}s",
                    metadata={
                        "frame_index": frame_idx,
                        "motion": True,
                    },
                )
                segments.append(segment)
                
            # If we have very few segments, add some evenly distributed segments
            if len(segments) < 5:
                step = duration / 10
                for i in range(10):
                    start = i * step
                    end = min(duration, start + 5.0)
                    
                    # Skip if too close to an existing segment
                    if any(abs(start - seg.start) < 5.0 for seg in segments):
                        continue
                        
                    segment = Segment(
                        start=start,
                        end=end,
                        score=0.4,  # Lower score for fallback segments
                        segment_type=SegmentType.CUSTOM,
                        text=f"Fallback segment at {start:.2f}s",
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
