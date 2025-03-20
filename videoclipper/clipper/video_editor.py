"""Video editing functionality for creating highlight clips."""

from typing import List, Optional, Tuple

from moviepy import VideoFileClip, concatenate_videoclips

from videoclipper.clipper.base import VideoClipper
from videoclipper.clipper.segment_selector import SegmentSelector
from videoclipper.exceptions import VideoProcessingError
from videoclipper.models.segment import Segment
from videoclipper.utils.validation import validate_output_path


class VideoEditor(VideoClipper):
    """Creates and edits video clips from interesting segments."""

    def __init__(self, video_path: str) -> None:
        """Initialize the video editor.

        Args:
            video_path: Path to the source video file
        """
        super().__init__(video_path)
        self._video: Optional[VideoFileClip] = None
        self._duration: Optional[float] = None

    def _load_video(self) -> VideoFileClip:
        """Load the video file.

        Returns:
            Loaded video file

        Raises:
            VideoProcessingError: If video loading fails
        """
        if self._video is None:
            try:
                self._video = VideoFileClip(self.video_path)
                self._duration = self._video.duration
                return self._video
            except Exception as e:
                raise VideoProcessingError(f"Failed to load video {self.video_path}: {e}")
        return self._video

    def create_clip(
        self, segments: List[Segment], output_path: str, max_duration: Optional[float] = None,
        viral_style: bool = True  # Enable viral-style edits
    ) -> Tuple[str, float]:
        """Create a video clip from the given segments.

        Args:
            segments: List of segments to include in the clip
            output_path: Path to save the output video
            max_duration: Maximum duration of the output clip in seconds
            viral_style: Whether to apply viral-style effects like speed changes

        Returns:
            Tuple of (output_path, duration)

        Raises:
            VideoProcessingError: If clip creation fails
        """
        try:
            from moviepy.video.fx.all import speedx, resize, crop, margin
            from moviepy.audio.fx.all import volumex
            
            video = self._load_video()
            output_path = validate_output_path(output_path)

            if not segments:
                raise VideoProcessingError("No segments provided for clip creation")

            # Extract the relevant subclips with viral-style effects
            subclips = []
            for i, segment in enumerate(segments):
                # Ensure segment boundaries are within video duration
                start = max(0, segment.start)
                end = min(self._duration or float('inf'), segment.end)

                if start < end:
                    # Use subclipped method instead of subclip
                    subclip = video.subclipped(start, end)
                    
                    # Apply viral-style effects if enabled
                    if viral_style:
                        # Apply different effects to alternating segments for variety
                        if segment.segment_type == SegmentType.SCENE_CHANGE:
                            # For scene changes, keep normal speed but boost audio
                            if hasattr(subclip, 'audio') and subclip.audio is not None:
                                subclip = subclip.set_audio(subclip.audio.fx(volumex, 1.3))
                                
                        elif i % 3 == 0:  # Speed up some segments
                            # Speed up by 10-25% for more dynamic pace
                            speed_factor = 1.1 + (i % 4) * 0.05  # Varies from 1.1 to 1.25
                            subclip = subclip.fx(speedx, speed_factor)
                            
                        # Add small pause between segments if not the last segment
                        if i < len(segments) - 1 and segment.duration > 3.0:
                            # Slow down the last 0.5 seconds slightly
                            end_time = subclip.duration
                            if end_time > 0.75:
                                main_part = subclip.subclipped(0, end_time - 0.5)
                                end_part = subclip.subclipped(end_time - 0.5, end_time)
                                end_part = end_part.fx(speedx, 0.9)  # Slow down
                                subclip = concatenate_videoclips([main_part, end_part])
                    
                    subclips.append(subclip)

            if not subclips:
                raise VideoProcessingError("No valid subclips could be created")

            # Concatenate the subclips
            final_clip = concatenate_videoclips(subclips)

            # Apply final viral-style effects if enabled
            if viral_style:
                # Boost contrast slightly
                def boost_contrast(img):
                    import numpy as np
                    return np.clip((img.astype(float) - 128) * 1.1 + 128, 0, 255).astype('uint8')
                
                final_clip = final_clip.fl_image(boost_contrast)
                
                # Boost audio level
                if hasattr(final_clip, 'audio') and final_clip.audio is not None:
                    final_clip = final_clip.set_audio(final_clip.audio.fx(volumex, 1.2))
            
            # Write the final clip to file
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                fps=video.fps,
                bitrate="2000k"  # Higher bitrate for better quality
            )

            clip_duration = final_clip.duration

            # Clean up
            final_clip.close()
            for clip in subclips:
                clip.close()

            return output_path, clip_duration

        except Exception as e:
            raise VideoProcessingError(f"Failed to create clip: {e}")

    def create_highlight_clip(
        self,
        segments: List[Segment],
        output_path: str,
        max_duration: float = 60.0,
        top_segments: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Create a highlight clip from the most interesting segments.

        Args:
            segments: List of segments to consider for the highlight
            output_path: Path to save the output video
            max_duration: Maximum duration of the highlight clip in seconds
            top_segments: Optional number of top segments to include

        Returns:
            Tuple of (output_path, duration)

        Raises:
            VideoProcessingError: If highlight creation fails
        """
        try:
            video = self._load_video()

            # Create segment selector
            selector = SegmentSelector(
                video_duration=self._duration or 0,
                min_segment_duration=3.0,
                max_segment_duration=15.0,
            )

            # Process segments
            processed_segments = selector.process_segments(segments)

            # Select top segments
            selected_segments = selector.select_top_segments(
                processed_segments, max_duration, top_segments
            )

            # Create the clip
            return self.create_clip(selected_segments, output_path)

        except Exception as e:
            raise VideoProcessingError(f"Failed to create highlight clip: {e}")
