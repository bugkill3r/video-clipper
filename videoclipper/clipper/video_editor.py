"""Video editing functionality for creating highlight clips."""

from typing import List, Optional, Tuple
import random
import logging
import numpy as np

from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.VideoClip import TextClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

from videoclipper.clipper.base import VideoClipper
from videoclipper.clipper.segment_selector import SegmentSelector
from videoclipper.exceptions import VideoProcessingError
from videoclipper.models.segment import Segment, SegmentType
from videoclipper.utils.validation import validate_output_path

# Set up logging
logger = logging.getLogger(__name__)

# Initialize stopwords for keyword extraction
NLTK_AVAILABLE = False
STOPWORDS = {'the', 'and', 'that', 'this', 'with', 'for', 'from', 'but', 'not', 'are', 'was',
            'were', 'have', 'has', 'had', 'you', 'your', 'they', 'their', 'our', 'is', 'it', 
            'a', 'an', 'i', 'me', 'my', 'we', 'us', 'what', 'who', 'how', 'when', 'where', 
            'which', 'there', 'here', 'to', 'of', 'in', 'on', 'at', 'by', 'as', 'be', 'been',
            'being', 'am', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can',
            'could', 'may', 'might', 'must', 'ought'}


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
    
    def _extract_keywords(self, text: str, max_keywords: int = 3) -> List[str]:
        """Extract important keywords from text.
        
        Args:
            text: The text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        if not text:
            return []
            
        # Clean and normalize the text
        text = text.lower().strip()
        
        # Basic extraction
        words = text.split()
        # Filter stopwords and short words
        keywords = [word.strip('.,!?;:"\'()[]{}') for word in words 
                   if word.strip('.,!?;:"\'()[]{}').lower() not in STOPWORDS 
                   and len(word.strip('.,!?;:"\'()[]{}')) > 3]
        
        # Return unique keywords
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen and kw.isalpha():
                seen.add(kw)
                unique_keywords.append(kw)
                if len(unique_keywords) >= max_keywords:
                    break
                    
        return unique_keywords

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
            
            # Debug logs
            logger.info(f"Creating caption for text: '{text}'")
            logger.info(f"Duration: {duration}, Video size: {video_size}")
                
            video_width, video_height = video_size
            
            # Define caption style parameters - larger font for better visibility
            font_size = max(36, int(video_height * 0.1))  # Increased font size further
            
            # Set a default font - use system default
            font = None
            
            # If no highlight words provided, extract them automatically
            if not highlight_words or len(highlight_words) == 0:
                highlight_words = self._extract_keywords(text)
                logger.info(f"Auto-extracted keywords: {highlight_words}")
            
            # Match the style from the example screenshot
            # Use a simpler, more visible approach like in the example
            text_with_highlights = text
            
            # Define a gradient-like set of colors for an eye-catching look
            text_colors = [
                (255, 255, 255),    # White
                (252, 238, 33),     # Yellow - similar to example
                (252, 131, 33),     # Orange - similar to example
                (247, 72, 48),      # Red-Orange - similar to example
            ]
            
            # Choose color based on position in video (top, middle, bottom)
            # This helps create a more visually interesting layout
            segment_position = 0.5  # Default to middle
            if hasattr(duration, 'start') and self._duration:
                segment_position = duration.start / self._duration
            
            # Choose a color based on position
            if segment_position < 0.33:
                text_color = text_colors[1]  # Yellow for early segments
            elif segment_position < 0.66:
                text_color = text_colors[2]  # Orange for middle segments
            else:
                text_color = text_colors[3]  # Red-Orange for later segments
            
            # Create a simple text clip with thick outlines - matches the example style
            text_clip_args = {
                'txt': text,
                'fontsize': font_size,
                'color': text_color,
                'stroke_color': 'black',
                'stroke_width': 3.0,  # Thicker stroke for better visibility
                'method': 'label',
                'align': 'center'
            }
            
            caption_clip = TextClip(**text_clip_args).set_duration(duration)
            
            # Create a dark semi-transparent background with rounded corners appearance
            margin = int(font_size * 0.7)  # Margin around text
            bg_width = min(video_width * 0.95, caption_clip.w + margin * 2)
            bg_height = caption_clip.h + margin
            
            # Create background with high contrast
            bg = ColorClip(
                size=(int(bg_width), int(bg_height)),
                color=(0, 0, 0)
            ).set_opacity(0.85).set_duration(duration)  # Higher opacity for better readability
            
            # Position text on background
            text_on_bg = CompositeVideoClip([
                bg,
                caption_clip.set_position(('center', 'center'))
            ], size=(int(bg_width), int(bg_height)))
            
            # Position captions similar to example - prominently in lower part
            # But not too close to the bottom (avoid device UI elements)
            position_y = video_height - bg_height - int(video_height * 0.1)
            
            # Apply final positioning
            final_caption = text_on_bg.set_position(('center', position_y))
            
            logger.info(f"Created caption clip with size {bg_width}x{bg_height}")
            return final_caption
            
        except Exception as e:
            logger.error(f"Caption creation failed: {str(e)}")
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
            viral_style: Whether to apply viral-style effects
            add_captions: Whether to add captions to the clips
            highlight_keywords: Words to highlight in captions

        Returns:
            Tuple of (output_path, duration)

        Raises:
            VideoProcessingError: If clip creation fails
        """
        try:
            logger.info(f"Creating clip with {len(segments)} segments")
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

            # Extract the relevant subclips
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
                            # Get video dimensions
                            video_size = (subclip.w, subclip.h)
                            logger.info(f"Adding captions to segment {start}-{end}")
                            
                            # Auto-generate keywords if not provided
                            if highlight_keywords is None:
                                # Extract important words from the text
                                segment_keywords = self._extract_keywords(segment.text)
                                logger.info(f"Auto-extracted keywords: {segment_keywords}")
                            else:
                                segment_keywords = highlight_keywords
                                logger.info(f"Using provided keywords: {segment_keywords}")
                            
                            # Create caption clip
                            caption_clip = self._create_caption_clip(
                                segment.text, 
                                subclip.duration,
                                video_size,
                                segment_keywords
                            )
                            
                            if caption_clip:
                                logger.info("Caption clip created successfully")
                                # Composite the caption with the video
                                try:
                                    subclip = CompositeVideoClip([
                                        subclip, 
                                        caption_clip
                                    ], size=(subclip.w, subclip.h))
                                    logger.info("Caption composited with video")
                                except Exception as e:
                                    logger.error(f"Failed to composite caption: {e}")
                            else:
                                logger.warning("Failed to create caption clip - returned None")
                        except Exception as e:
                            # If caption fails, just use the original clip
                            logger.error(f"Caption addition failed: {str(e)}")
                    else:
                        if not hasattr(segment, 'text') or not segment.text:
                            logger.warning(f"Segment at {start}-{end} has no text for captions")
                        elif not add_captions:
                            logger.info("Captions disabled")
                    
                    subclips.append(subclip)

            if not subclips:
                raise VideoProcessingError("No valid subclips could be created")

            # Concatenate the subclips
            logger.info(f"Concatenating {len(subclips)} subclips")
            final_clip = concatenate_videoclips(subclips)
            
            # Write the final clip to file with higher quality
            logger.info(f"Writing final clip to {output_path}")
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
            logger.info(f"Clip creation complete, duration: {clip_duration}s")

            # Clean up
            final_clip.close()
            for clip in subclips:
                clip.close()

            return output_path, clip_duration

        except Exception as e:
            logger.error(f"Failed to create clip: {e}")
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