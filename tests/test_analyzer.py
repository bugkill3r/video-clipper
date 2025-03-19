"""Tests for the analyzer components."""

import os
import tempfile
from unittest import mock

import pytest

from videoclipper.analyzer.scene_detector import SceneDetector
from videoclipper.exceptions import AnalysisError
from videoclipper.models.segment import SegmentType


@pytest.fixture
def mock_video_path():
    """Create a fake video path for testing."""
    return "/path/to/fake_video.mp4"


class TestSceneDetector:
    """Tests for the SceneDetector class."""

    def test_init(self, mock_video_path):
        """Test initialization."""
        detector = SceneDetector(mock_video_path)
        assert detector.file_path == mock_video_path
        assert detector.segments == []

    @mock.patch("cv2.VideoCapture")
    def test_detect_scenes_error(self, mock_video_capture, mock_video_path):
        """Test error handling when video can't be opened."""
        # Setup mock to simulate failure to open video
        mock_instance = mock_video_capture.return_value
        mock_instance.isOpened.return_value = False

        detector = SceneDetector(mock_video_path)

        with pytest.raises(AnalysisError):
            detector.detect_scenes()

    @mock.patch("cv2.VideoCapture")
    @mock.patch("cv2.cvtColor")
    @mock.patch("cv2.absdiff")
    @mock.patch("numpy.mean")
    def test_detect_scenes(self, mock_mean, mock_absdiff, mock_cvtcolor, mock_video_capture, mock_video_path):
        """Test scene detection."""
        # Setup mocks
        mock_instance = mock_video_capture.return_value
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 30.0 if prop == cv2.CAP_PROP_FPS else 300

        # Mock read to return two frames and then False
        mock_instance.read.side_effect = [
            (True, mock.MagicMock()),
            (True, mock.MagicMock()),
            (False, None)
        ]

        # Mock frame difference calculation
        mock_mean.return_value = 35.0  # Above default threshold of 30.0

        # Run detector
        detector = SceneDetector(mock_video_path)
        segments = detector.detect_scenes()

__all__ = ["VideoClipper", "VideoProcessor"]
        # Verify results
        assert len(segments) == 1
        assert segments[0].segment_type == SegmentType.SCENE_CHANGE
        assert segments[0].start == 1.0  # Second frame at 30fps
