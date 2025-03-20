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
    
    def _split_text_into_chunks(self, text: str, max_chunk_length: int = 60) -> List[str]:
        """Splits text into smaller chunks for better caption timing and readability.
        
        Args:
            text: The text to split into chunks
            max_chunk_length: Maximum length of each chunk in characters
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # First, try to split by sentence
        sentences = []
        # Match sentence boundaries (period, question mark, exclamation point followed by space)
        for sentence in re.split(r'[.!?]\s+', text):
            if sentence:
                sentences.append(sentence.strip())
        
        # If no sentence boundaries, or very short text, return as single chunk
        if len(sentences) <= 1 or len(text) < max_chunk_length:
            # Check if the single sentence is too long
            if len(text) > max_chunk_length:
                # Split by commas or natural pauses
                clauses = []
                for clause in re.split(r'[,;:]\s+', text):
                    if clause:
                        clauses.append(clause.strip())
                
                # If we have multiple clauses, use them
                if len(clauses) > 1:
                    return clauses
                
                # If still too long, just split by character count
                if len(text) > max_chunk_length:
                    chunks = []
                    for i in range(0, len(text), max_chunk_length):
                        chunk = text[i:i+max_chunk_length].strip()
                        if chunk:
                            chunks.append(chunk)
                    return chunks
            
            # If text is short enough or can't be split, return as is
            return [text]
        
        # If we have multiple sentences, merge short ones to avoid too many tiny captions
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would make the chunk too long, start a new chunk
            if len(current_chunk) + len(sentence) + 1 > max_chunk_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # Add to current chunk with a space if needed
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if any
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

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
            font_size = max(36, int(video_height * 0.05))
            
            # If no highlight words provided, extract them automatically
            if not highlight_words or len(highlight_words) == 0:
                highlight_words = self._extract_keywords(text)
                logger.info(f"Auto-extracted keywords: {highlight_words}")
            
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
            
            # Highlight specified words by creating individual clips for each word
            # Convert to lowercase for case-insensitive comparison
            highlight_words_lower = [word.lower() for word in highlight_words]
            
            # Try creating the text clip with correct parameters for MoviePy 2.1.1/2.1.2
            try:
                # In MoviePy 2.1.1/2.1.2, font is the first required parameter
                caption_clip = TextClip(
                    font='Arial',
                    text=text,
                    font_size=font_size,
                    color=text_color,
                    stroke_color='black',
                    stroke_width=2,
                    method='label'
                )
                # Set duration using with_duration instead of set_duration
                caption_clip = caption_clip.with_duration(duration)
                logger.info("Created styled caption with parameters")
            except Exception as e:
                logger.error(f"Failed to create styled caption: {e}")
                
                # Fallback to simpler caption with just basic parameters
                try:
                    caption_clip = TextClip(
                        font='Arial',
                        text=text,
                        font_size=font_size,
                        color=text_color
                    )
                    caption_clip = caption_clip.with_duration(duration)
                    logger.info("Created basic caption with minimal parameters")
                except Exception as e2:
                    logger.error(f"Failed to create basic caption: {e2}")
                    
                    # Last-resort fallback with absolute minimal parameters
                    try:
                        # Try with system font path as last resort
                        caption_clip = TextClip(
                            font='/System/Library/Fonts/Helvetica.ttc',
                            text=text,
                            font_size=font_size
                        )
                        caption_clip = caption_clip.with_duration(duration)
                        logger.info("Created minimal caption with system font")
                    except Exception as e3:
                        logger.error(f"All caption creation attempts failed: {e3}")
                        return None
            
            # Create a dark semi-transparent background
            try:
                margin = int(font_size * 0.7)  # Margin around text
                bg_width = min(video_width * 0.95, caption_clip.w + margin * 2)
                bg_height = caption_clip.h + margin
                
                # Create dark background (use a solid color with opacity)
                bg = ColorClip(
                    size=(int(bg_width), int(bg_height)),
                    color=(0, 0, 0),
                    duration=duration
                )
                # Set opacity for semi-transparency
                bg = bg.with_opacity(0.8)
                
                # Position text on background
                text_on_bg = CompositeVideoClip([
                    bg,
                    caption_clip.with_position(('center', 'center'))
                ], size=(int(bg_width), int(bg_height)))
                
                caption_clip = text_on_bg
                logger.info("Added background to caption")
            except Exception as bg_error:
                logger.warning(f"Could not create background: {bg_error}")
            
            # Position captions in the lower part of the video
            position_y = video_height - caption_clip.h - int(video_height * 0.1)
            
            # Apply final positioning
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
                            
                            # For longer segments, split text into smaller chunks for better readability
                            if subclip.duration > 6.0 and len(segment.text) > 70:
                                # Split text into sentences or phrases
                                chunks = self._split_text_into_chunks(segment.text)
                                logger.info(f"Split text into {len(chunks)} chunks for more readable captions")
                                
                                caption_clips = []
                                chunk_duration = subclip.duration / len(chunks)
                                
                                for i, chunk_text in enumerate(chunks):
                                    # Auto-generate keywords for each chunk if not provided
                                    if highlight_keywords is None:
                                        chunk_keywords = self._extract_keywords(chunk_text)
                                    else:
                                        chunk_keywords = highlight_keywords
                                    
                                    # Calculate start time for this chunk within the subclip
                                    chunk_start = i * chunk_duration
                                    
                                    # Create caption clip for this chunk
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
                                        logger.info(f"Created caption for chunk {i+1}/{len(chunks)}")
                                
                                if caption_clips:
                                    # Composite all caption chunks with the video
                                    try:
                                        all_clips = [subclip] + caption_clips
                                        subclip = CompositeVideoClip(all_clips, size=(subclip.w, subclip.h))
                                        logger.info(f"Composited {len(caption_clips)} caption chunks with video")
                                    except Exception as e:
                                        logger.error(f"Failed to composite caption chunks: {e}")
                                
                            else:
                                # For shorter segments, use a single caption
                                # Auto-generate keywords if not provided
                                if highlight_keywords is None:
                                    segment_keywords = self._extract_keywords(segment.text)
                                    logger.info(f"Auto-extracted keywords: {segment_keywords}")
                                else:
                                    segment_keywords = highlight_keywords
                                    logger.info(f"Using provided keywords: {segment_keywords}")
                                
                                # Create a single caption clip
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