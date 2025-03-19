"""Advanced content analysis for videos."""

from typing import List, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

from videoclipper.analyzer.base import Analyzer
from videoclipper.exceptions import AnalysisError
from videoclipper.models.segment import Segment, SegmentType


class ContentAnalyzer(Analyzer):
    """Analyzes video content to find visually interesting segments."""

    def __init__(self, file_path: str) -> None:
        """Initialize the content analyzer.

        Args:
            file_path: Path to the video file
        """
        super().__init__(file_path)

    def analyze_visual_interest(
        self, sample_rate: int = 24, n_clusters: int = 5
    ) -> List[Segment]:
        """Analyze visual interest to find diverse and interesting segments.

        This uses frame clustering to identify visually distinct parts of the video.

        Args:
            sample_rate: Number of frames to sample per second
            n_clusters: Number of visual clusters to identify

        Returns:
            List of segments representing visually interesting parts

        Raises:
            AnalysisError: If visual analysis fails
        """
        try:
            cap = cv2.VideoCapture(self.file_path)
            if not cap.isOpened():
                raise AnalysisError(f"Failed to open video file: {self.file_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Determine frame sampling interval
            sample_interval = max(1, int(fps / sample_rate))
            features = []
            frame_indices = []

            # Sample frames and extract features
            frame_idx = 0
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to reduce dimensionality
                small_frame = cv2.resize(frame, (32, 32))
                # Convert to grayscale
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                # Flatten to feature vector
                feature = gray.flatten().astype(np.float32)
                # Normalize
                feature /= np.linalg.norm(feature) + 1e-6

                features.append(feature)
                frame_indices.append(frame_idx)

                frame_idx += sample_interval
                if frame_idx >= total_frames:
                    break

            cap.release()

            if not features:
                return []

            # Cluster frames using K-means
            kmeans = KMeans(n_clusters=min(n_clusters, len(features)))
            clusters = kmeans.fit_predict(features)

            # Find the frame closest to each cluster center
            centermost_frames = []
            for cluster_idx in range(min(n_clusters, len(features))):
                cluster_frames = [i for i, c in enumerate(clusters) if c == cluster_idx]
                if not cluster_frames:
                    continue

                # Find frame closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_idx]
                distances = [
                    np.linalg.norm(features[i] - cluster_center) for i in cluster_frames
                ]
                centermost_idx = cluster_frames[np.argmin(distances)]
                centermost_frames.append(frame_indices[centermost_idx])

            # Sort by frame index
            centermost_frames.sort()

            # Convert to segments
            segments = []
            for frame_idx in centermost_frames:
                # Calculate timestamp
                timestamp = frame_idx / fps

                # Create a segment of 5 seconds, centered on the key frame
                start = max(0, timestamp - 2.5)
                end = min(duration, timestamp + 2.5)

                segment = Segment(
                    start=start,
                    end=end,
                    score=0.65,  # Default score for visual interest segments
                    segment_type=SegmentType.CUSTOM,
                    text=f"Visually interesting at {timestamp:.2f}s",
                    metadata={
                        "frame_index": frame_idx,
                        "visual_cluster": True,
                    },
                )
                segments.append(segment)

            self.segments = segments
            return segments

        except Exception as e:
            raise AnalysisError(f"Visual analysis failed: {e}")

    def analyze(self, **kwargs) -> List[Segment]:
        """Run visual interest analysis.

        Args:
            **kwargs: Keyword arguments passed to analyze_visual_interest

        Returns:
            List of visually interesting segments
        """
        return self.analyze_visual_interest(**kwargs)
