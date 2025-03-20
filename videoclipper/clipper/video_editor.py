"""Video editing functionality for creating highlight clips."""

from typing import List, Optional, Tuple
import random
import logging
import re
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
    
    def _split_text_into_chunks(self, text: str, max_words: int = 5) -> List[str]:
        """Splits text into smaller chunks for better caption timing and readability.
        
        Args:
            text: The text to split into chunks
            max_words: Maximum number of words per chunk (4-6 words per screenshot style)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split into words and create short chunks of 4-6 words
        words = text.split()
        chunks = []
        
        # Process words into small chunks like in the screenshot
        for i in range(0, len(words), max_words):
            # Get a chunk of words (max 4-6 words per chunk like in the example)
            chunk = words[i:i+max_words]
            if chunk:
                # Join the words back into a short phrase
                chunk_text = " ".join(chunk)
                chunks.append(chunk_text)
        
        # If we ended up with no chunks, return the original text as one chunk
        if not chunks:
            return [text]
            
        return chunks

    def _create_caption_clip(self, text: str, duration: float, video_size: Tuple[int, int], 
                        highlight_words: Optional[List[str]] = None) -> TextClip:
        """Create a styled caption clip with word highlighting, matching the style in the example.
        
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
            # Match the large, bold font from the screenshot
            font_size = max(40, int(video_height * 0.07))
            
            # If no highlight words provided, extract them automatically
            if not highlight_words or len(highlight_words) == 0:
                highlight_words = self._extract_keywords(text)
                logger.info(f"Auto-extracted keywords: {highlight_words}")
            
            # Use colors from the screenshot - bright yellow for highlights,
            # white for regular text with black stroke for readability
            highlight_color = (255, 255, 0)  # Bright yellow like in the screenshot
            regular_color = (255, 255, 255)  # White
            
            # Convert highlight words to lowercase for case-insensitive comparison
            highlight_words_lower = [word.lower() for word in highlight_words]
            
            # Check if any words in the text should be highlighted
            # If so, create a colorized version of the text
            colorized_text = text
            has_highlights = False
            
            for word in highlight_words:
                if word.lower() in text.lower():
                    has_highlights = True
                    # Will use the highlighted style
                    break
            
            # Try creating the text clip with correct parameters for MoviePy 2.1.1/2.1.2
            try:
                # In MoviePy 2.1.1/2.1.2, font is the first required parameter
                # Match the bold, prominent style from the screenshot
                caption_clip = TextClip(
                    font='Arial',  # Use Arial font which is more commonly available
                    text=text,
                    font_size=font_size,
                    color=highlight_color if has_highlights else regular_color,
                    stroke_color='black',
                    stroke_width=3,     # Thicker stroke like in the screenshot
                    method='label'
                )
                # Set duration using with_duration instead of set_duration
                caption_clip = caption_clip.with_duration(duration)
                logger.info("Created styled caption with parameters")
            except Exception as e:
                logger.error(f"Failed to create styled caption: {e}")
                
                # Fallback to system font if Arial-Bold fails
                try:
                    caption_clip = TextClip(
                        font='/System/Library/Fonts/Helvetica.ttc',
                        text=text,
                        font_size=font_size,
                        color=highlight_color if has_highlights else regular_color,
                        stroke_color='black',
                        stroke_width=3
                    )
                    caption_clip = caption_clip.with_duration(duration)
                    logger.info("Created caption with system font")
                except Exception as e2:
                    logger.error(f"Failed with system font: {e2}")
                    
                    # Last-resort fallback with absolute minimal parameters
                    try:
                        caption_clip = TextClip(
                            font='Arial',
                            text=text,
                            font_size=font_size
                        )
                        caption_clip = caption_clip.with_duration(duration)
                        logger.info("Created minimal caption with basic parameters")
                    except Exception as e3:
                        logger.error(f"All caption creation attempts failed: {e3}")
                        return None
            
            # In the screenshot, captions don't have a background but have a thick stroke
            # for readability. We'll skip the background for now since the stroke should
            # provide enough contrast.
            
            # Position captions in the lower part of the video, matching the screenshot
            # The screenshot shows captions positioned about 1/3 up from the bottom
            position_y = int(video_height * 0.7)
            
            # Apply final positioning - center horizontally, fixed position vertically
            final_caption = caption_clip.with_position(('center', position_y))
            
            logger.info(f"Created caption clip with size {caption_clip.w}x{caption_clip.h}")
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
                            
                            # Always split text into shorter chunks like in the screenshot (4-6 words)
                            # This matches the style in the reference image 
                            chunks = self._split_text_into_chunks(segment.text)
                            logger.info(f"Split text into {len(chunks)} small chunks (4-6 words each)")
                            
                            caption_clips = []
                            # Each chunk gets a short duration - words should appear briefly
                            # Caption duration in the screenshot is very short (1-2 seconds per phrase)
                            chunk_duration = min(2.0, subclip.duration / len(chunks))
                            
                            for i, chunk_text in enumerate(chunks):
                                # Auto-generate keywords for each chunk
                                # In the screenshot, almost every chunk has a highlighted word
                                chunk_keywords = []
                                
                                # Extract keywords specific to this chunk
                                if highlight_keywords is None:
                                    # Try to highlight at least one word in each chunk if possible
                                    chunk_words = chunk_text.split()
                                    for word in chunk_words:
                                        if len(word) > 3 and word.lower() not in STOPWORDS:
                                            chunk_keywords.append(word)
                                            break
                                    
                                    # If no keywords found, use extracted keywords from whole segment
                                    if not chunk_keywords:
                                        chunk_keywords = self._extract_keywords(chunk_text)
                                else:
                                    # Use provided keywords that appear in this chunk
                                    for word in highlight_keywords:
                                        if word.lower() in chunk_text.lower():
                                            chunk_keywords.append(word)
                                
                                # Calculate timing for this chunk within the subclip
                                # Distribute chunks evenly across the subclip duration
                                chunk_start = i * (subclip.duration / len(chunks))
                                
                                # Create caption clip for this chunk with the screenshot style
                                caption = self._create_caption_clip(
                                    chunk_text,
                                    chunk_duration,
                                    video_size,
                                    chunk_keywords
                                )
                                
                                if caption:
                                    # Start this caption at the calculated time
                                    caption = caption.with_start(chunk_start)
                                    caption_clips.append(caption)
                                    logger.info(f"Created caption for chunk {i+1}/{len(chunks)}: '{chunk_text}'")
                            
                            if caption_clips:
                                # Composite all caption chunks with the video
                                try:
                                    all_clips = [subclip] + caption_clips
                                    subclip = CompositeVideoClip(all_clips, size=(subclip.w, subclip.h))
                                    logger.info(f"Composited {len(caption_clips)} short caption chunks with video")
                                except Exception as e:
                                    logger.error(f"Failed to composite caption chunks: {e}")
                            else:
                                logger.warning("No caption clips were created for this segment")
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