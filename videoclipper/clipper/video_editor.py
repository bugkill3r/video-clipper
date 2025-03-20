"""Video editing functionality for creating highlight clips."""

from typing import List, Optional, Tuple, Dict, Any
import random
import numpy as np

from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.tools.drawing import color_gradient
from moviepy.video.VideoClip import TextClip, ImageClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

from videoclipper.clipper.base import VideoClipper
from videoclipper.clipper.segment_selector import SegmentSelector
from videoclipper.exceptions import VideoProcessingError
from videoclipper.models.segment import Segment, SegmentType
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

    def _create_caption_clip(self, text: str, duration: float, video_size: Tuple[int, int], 
                        highlight_words: Optional[List[str]] = None) -> TextClip:
        """Create a styled caption clip with word highlighting.
        
        Args:
            text: The caption text
            duration: Duration of the caption
            video_size: Size of the video (width, height)
            highlight_words: List of words to highlight
            
        Returns:
            TextClip with styled captions
        """
        try:
            if not text:
                return None

            # Import inside method to avoid dependency issues
            from moviepy.video.VideoClip import TextClip, ColorClip
            from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
            import numpy as np
                
            video_width, video_height = video_size
            
            # Define caption style parameters
            font_size = max(24, int(video_height * 0.07))  # Scale with video height
            
            # Try to find a suitable font that exists
            available_fonts = ["Arial-Bold", "Arial", "Helvetica-Bold", "Helvetica", "DejaVuSans-Bold", "DejaVuSans"]
            font = None
            
            for test_font in available_fonts:
                try:
                    # Test font by creating a small TextClip
                    test_clip = TextClip("Test", font=test_font, fontsize=12)
                    test_clip.close()
                    font = test_font
                    break
                except Exception:
                    continue
                    
            # Default to None if no font works (will use default system font)
            if font is None:
                print("Warning: Could not find a suitable font, using system default")
            
            # Use enhanced styled text with colored words
            if highlight_words and len(highlight_words) > 0:
                # Create individual clips for each word with different colors for highlights
                word_clips = []
                words = text.split()
                x_pos = 0
                word_height = 0
                
                # Choose high-contrast colors for highlights
                highlight_colors = [
                    (255, 82, 82),    # Red
                    (255, 157, 80),   # Orange
                    (255, 213, 79),   # Yellow
                    (76, 175, 80),    # Green
                    (33, 150, 243),   # Blue
                    (156, 39, 176),   # Purple
                    (255, 64, 129)    # Pink
                ]
                
                for word in words:
                    word_clean = word.lower().strip(".,!?;:'\"()-")
                    is_highlight = any(word_clean == hw.lower() for hw in highlight_words)
                    
                    # Choose color based on highlight status
                    if is_highlight:
                        color = random.choice(highlight_colors)
                        fontweight = 'bold'
                    else:
                        color = 'white'
                        fontweight = 'normal'
                    
                    # Create text clip for this word
                    word_clip = TextClip(
                        word + " ",  # Add space after word
                        fontsize=font_size,
                        color=color,
                        font=font,
                        stroke_color='black',
                        stroke_width=1.5,
                        method='label'
                    ).set_duration(duration)
                    
                    # Add shadow effect for better visibility
                    if is_highlight:
                        shadow = TextClip(
                            word + " ",
                            fontsize=font_size,
                            color='black',
                            font=font,
                            stroke_width=0,
                            method='label'
                        ).set_duration(duration)
                        
                        # Shift shadow slightly
                        shadow = shadow.set_position((2, 2))
                        word_clip = CompositeVideoClip([shadow, word_clip])
                    
                    word_clips.append(word_clip)
                
                # Create background with certain opacity
                bg_color = (0, 0, 0)
                bg_opacity = 0.7
                
                # Calculate total width and height needed
                total_width = sum(clip.w for clip in word_clips)
                max_height = max(clip.h for clip in word_clips)
                
                # Create a background clip
                bg_width = min(video_width * 0.95, total_width + 40)  # Add padding
                bg_height = max_height + 20  # Add padding
                
                bg = ColorClip(
                    size=(int(bg_width), int(bg_height)),
                    color=bg_color
                ).set_opacity(bg_opacity).set_duration(duration)
                
                # Position word clips on the background
                positioned_clips = [bg]
                x_offset = (bg_width - total_width) / 2  # Center text horizontally
                
                for word_clip in word_clips:
                    word_clip = word_clip.set_position((x_offset, 10))  # 10px from top
                    x_offset += word_clip.w
                    positioned_clips.append(word_clip)
                
                # Composite all clips
                caption_clip = CompositeVideoClip(
                    positioned_clips,
                    size=(int(bg_width), int(bg_height))
                ).set_duration(duration)
                
                # Position at bottom of screen
                caption_clip = caption_clip.set_position(('center', 'bottom'))
                return caption_clip
            
            else:
                # Simpler version without individual word highlighting
                caption_clip = TextClip(
                    text,
                    fontsize=font_size,
                    color='white',
                    font=font,
                    stroke_color='black',
                    stroke_width=1.5,
                    method='label',
                    size=(int(video_width * 0.9), None),
                    align='center'
                ).set_duration(duration)
                
                # Create semi-transparent background
                bg_width = caption_clip.w + 40
                bg_height = caption_clip.h + 20
                
                bg = ColorClip(
                    size=(int(bg_width), int(bg_height)),
                    color=(0, 0, 0)
                ).set_opacity(0.7).set_duration(duration)
                
                # Composite with background
                caption_with_bg = CompositeVideoClip([
                    bg,
                    caption_clip.set_position(('center', 'center'))
                ], size=(int(bg_width), int(bg_height)))
                
                # Position at bottom of screen
                return caption_with_bg.set_position(('center', 'bottom'))
        
        except Exception as e:
            print(f"Caption creation failed: {str(e)}")
            # Return None if we can't create the caption
            return None
    
    def create_clip(
        self, segments: List[Segment], output_path: str, max_duration: Optional[float] = None,
        viral_style: bool = True,  # Enable viral-style edits
        add_captions: bool = True,  # Add captions to the clip
        highlight_keywords: Optional[List[str]] = None  # Words to highlight in captions
    ) -> Tuple[str, float]:
        """Create a video clip from the given segments.

        Args:
            segments: List of segments to include in the clip
            output_path: Path to save the output video
            max_duration: Maximum duration of the output clip in seconds
            viral_style: Whether to apply viral-style effects like speed changes
            add_captions: Whether to add captions to the clips
            highlight_keywords: Words to highlight in captions

        Returns:
            Tuple of (output_path, duration)

        Raises:
            VideoProcessingError: If clip creation fails
        """
        try:
            video = self._load_video()
            output_path = validate_output_path(output_path)

            if not segments:
                raise VideoProcessingError("No segments provided for clip creation")

            # Sort segments by start time to ensure chronological order
            segments.sort(key=lambda x: x.start)
            
            # Ensure good spacing between segments (minimum 10 seconds gap)
            selected_segments = []
            last_end_time = -20  # Initialize to negative value to ensure first segment is always included
            
            for segment in segments:
                start = max(0, segment.start)
                end = min(self._duration or float('inf'), segment.end)
                
                # Skip segments that are too close to the previous one
                if start < last_end_time + 10 and selected_segments:
                    continue
                
                if start < end:
                    selected_segments.append(segment)
                    last_end_time = end
                    
                    # Break if we have enough segments
                    if len(selected_segments) >= 5:
                        break

            # Extract the relevant subclips with captions if requested
            subclips = []
            for segment in selected_segments:
                # Ensure segment boundaries are within video duration
                start = max(0, segment.start)
                end = min(self._duration or float('inf'), segment.end)
                
                if start < end:
                    # Use subclipped method instead of subclip
                    subclip = video.subclipped(start, end)
                    
                    # Add captions if requested and segment has text
                    if add_captions and hasattr(segment, 'text') and segment.text:
                        try:
                            print(f"Adding captions to segment {segment.start:.1f}-{segment.end:.1f}")
                            print(f"Caption text: '{segment.text}'")
                            
                            # Get video dimensions
                            video_size = (subclip.w, subclip.h)
                            print(f"Video dimensions: {video_size}")
                            
                            # Auto-generate keywords if not provided
                            if highlight_keywords is None:
                                # Extract important words (nouns, verbs, adjectives)
                                words = segment.text.split()
                                # Words with 4+ characters that aren't common stopwords
                                stopwords = {'the', 'and', 'that', 'this', 'with', 'for', 'from', 'but'}
                                potential_keywords = [w for w in words if len(w) >= 4 and w.lower() not in stopwords]
                                # Select up to 3 words to highlight
                                segment_keywords = potential_keywords[:3] if potential_keywords else []
                                print(f"Auto-generated keywords: {segment_keywords}")
                            else:
                                segment_keywords = highlight_keywords
                                print(f"Using provided keywords: {segment_keywords}")
                            
                            # Create caption clip
                            caption_clip = self._create_caption_clip(
                                segment.text, 
                                subclip.duration,
                                video_size,
                                segment_keywords
                            )
                            
                            if caption_clip:
                                print("Caption clip created successfully")
                                # Composite the caption with the video
                                subclip = CompositeVideoClip([
                                    subclip, 
                                    caption_clip
                                ])
                            else:
                                print("Failed to create caption clip")
                        except Exception as e:
                            # If caption fails, just use the original clip
                            print(f"Caption addition failed: {str(e)}")
                            import traceback
                            print(traceback.format_exc())
                    
                    subclips.append(subclip)

            if not subclips:
                raise VideoProcessingError("No valid subclips could be created")

            # Concatenate the subclips
            final_clip = concatenate_videoclips(subclips)
            
            # Write the final clip to file with higher quality
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                fps=video.fps,
                bitrate="5000k",  # Higher bitrate for better quality
                threads=4,
                preset="fast"
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