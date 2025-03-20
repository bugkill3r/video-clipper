"""Video editing functionality for creating highlight clips."""

from typing import List, Optional, Tuple, Callable
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
    
    def _split_text_into_chunks(self, text: str, max_words: int = 4) -> List[str]:
        """Splits text into smaller chunks for better caption timing and readability.
        
        Args:
            text: The text to split into chunks
            max_words: Maximum number of words per chunk (3-5 words per screenshot style)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Clean up text - remove extra spaces, normalize punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Split sentence into words
            words = sentence.split()
            
            # For very short sentences (3-4 words), keep them intact
            if len(words) <= max_words:
                chunks.append(" ".join(words))
                continue
            
            # For longer sentences, try to create natural phrases (mimicking speech rhythm)
            # Look for natural breaking points like commas, conjunctions
            natural_breaks = [i for i, word in enumerate(words) 
                            if word.endswith(',') or word.lower() in ['and', 'but', 'or', 'because', 'so']]
            
            if natural_breaks:
                # Use natural breaking points to create chunks
                current_start = 0
                for break_point in natural_breaks:
                    # Only create chunk if it's not too long
                    if break_point - current_start + 1 <= max_words + 1:
                        chunk = words[current_start:break_point + 1]
                        chunks.append(" ".join(chunk))
                        current_start = break_point + 1
                    
                # Add any remaining words
                if current_start < len(words):
                    remaining = words[current_start:]
                    # Process remaining words in small groups
                    for i in range(0, len(remaining), max_words):
                        sub_chunk = remaining[i:i+max_words]
                        if sub_chunk:
                            chunks.append(" ".join(sub_chunk))
            else:
                # No natural breaks, so just split into equal chunks of max_words
                for i in range(0, len(words), max_words):
                    chunk = words[i:i+max_words]
                    if chunk:
                        chunks.append(" ".join(chunk))
        
        # Further refinement - ensure each chunk has an emphasized word if possible
        refined_chunks = []
        for chunk in chunks:
            words = chunk.split()
            has_keyword = False
            
            # Check if this chunk already has a potential keyword (non-stopword)
            for word in words:
                clean_word = word.strip('.,!?;:"\'()[]{}').lower()
                if len(clean_word) > 3 and clean_word not in STOPWORDS:
                    has_keyword = True
                    break
            
            # If this chunk has a keyword, keep it as is
            if has_keyword or len(words) <= 2:  # Very short chunks should stay intact
                refined_chunks.append(chunk)
            else:
                # No keywords in this chunk, so try to combine with another chunk
                # Only combine if we have a previous chunk and the combined length isn't too long
                if refined_chunks and len(refined_chunks[-1].split()) + len(words) <= max_words + 2:
                    refined_chunks[-1] = refined_chunks[-1] + " " + chunk
                else:
                    refined_chunks.append(chunk)
                
        # If we ended up with no chunks, return the original text as one chunk
        if not refined_chunks:
            return [text]
            
        return refined_chunks

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
            font_size = max(45, int(video_height * 0.08))  # Larger font for more impact
            
            # If no highlight words provided, extract them automatically
            if not highlight_words or len(highlight_words) == 0:
                highlight_words = self._extract_keywords(text)
                logger.info(f"Auto-extracted keywords: {highlight_words}")
            
            # Colors from the screenshot - bright yellow for highlights with black stroke,
            # white for regular text with black stroke for readability
            highlight_color = (255, 255, 0)  # Bright yellow like in the screenshot
            regular_color = (255, 255, 255)  # White
            
            # Convert highlight words to lowercase for case-insensitive comparison
            highlight_words_lower = [word.lower() for word in highlight_words]
            
            # Now we'll colorize individual words to match the screenshot exactly
            # Break the text into words to identify which should be highlighted
            words = text.split()
            colorized_words = []
            
            for word in words:
                # Strip punctuation for matching but keep it for display
                clean_word = word.strip('.,!?;:"\'()[]{}')
                
                # Check if this word should be highlighted (case-insensitive)
                should_highlight = False
                for highlight in highlight_words:
                    if clean_word.lower() == highlight.lower():
                        should_highlight = True
                        break
                
                # Add the word with its color coding
                if should_highlight:
                    colorized_words.append((word, highlight_color))
                else:
                    colorized_words.append((word, regular_color))
            
            # Try creating the text clip with advanced styling for MoviePy 2.1.1/2.1.2
            try:
                # In MoviePy 2.1.1/2.1.2, create individual clips for each word
                word_clips = []
                
                # Total width calculation for positioning
                total_width = 0
                word_widths = []
                
                # First pass to create clips and measure total width
                for word_text, word_color in colorized_words:
                    try:
                        word_clip = TextClip(
                            font='Arial-Bold',  # Try with Arial-Bold first for greater impact
                            text=word_text + " ",  # Add space after each word
                            font_size=font_size,
                            color=word_color, 
                            stroke_color='black',
                            stroke_width=4,  # Thicker stroke for better readability
                            method='label'
                        )
                        total_width += word_clip.w
                        word_widths.append(word_clip.w)
                        word_clips.append((word_clip, word_color))
                    except Exception as e:
                        # Fallback to Arial if Arial-Bold fails
                        try:
                            word_clip = TextClip(
                                font='Arial',
                                text=word_text + " ",
                                font_size=font_size,
                                color=word_color,
                                stroke_color='black',
                                stroke_width=4,
                                method='label'
                            )
                            total_width += word_clip.w
                            word_widths.append(word_clip.w)
                            word_clips.append((word_clip, word_color))
                        except Exception as e2:
                            logger.error(f"Failed to create word clip: {e2}")
                            # Skip this word if it fails
                            continue
                
                # Now position each word clip properly
                composite_clips = []
                current_x = int((video_width - total_width) / 2)  # Center the text line
                
                # Position words in a row
                for i, (word_clip, color) in enumerate(word_clips):
                    # Add animation effects based on color - highlighted words get more effects
                    clip_duration = duration
                    
                    # Add slight fade and scale animation for all words (subtle animation)
                    # Scale up and fade in at start for a subtle pop effect
                    word_clip = word_clip.with_duration(clip_duration)
                    
                    # Position the word clip
                    positioned_clip = word_clip.with_position((current_x, int(video_height * 0.7)))
                    
                    # Add extra visual impact for highlighted words
                    if color == highlight_color:
                        # Make highlighted words dynamic and eye-catching with subtle effects
                        # This matches the screenshot where highlighted words stand out
                        
                        # For highlighted words, we apply simple but effective animation
                        # The simple opacity effect is more reliable than complex scaling
                        
                        # Apply a simple fade-in effect for highlighted words
                        # This is more reliable than complex animations
                        try:
                            # Add a slightly quicker fade-in for highlighted words
                            positioned_clip = positioned_clip.with_opacity(
                                lambda t: min(1.0, 2.0 * t) if t < 0.2 else 1.0
                            )
                        except Exception as e:
                            # If effects fail, fallback to basic positioning
                            logger.error(f"Effect application failed: {e}")
                            # Keep the clip without advanced effects
                        
                    composite_clips.append(positioned_clip)
                    current_x += word_widths[i]
                
                # Create a composite clip from all word clips
                if composite_clips:
                    caption_clip = CompositeVideoClip(composite_clips, size=video_size)
                    caption_clip = caption_clip.with_duration(duration)
                    logger.info("Created advanced styled caption with word-by-word highlighting")
                else:
                    raise ValueError("No word clips were created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create advanced caption: {e}")
                
                # Fallback to simpler implementation if word-by-word fails
                try:
                    # Create a single text clip with standard styling
                    caption_clip = TextClip(
                        font='Arial',  # Use Arial font which is more commonly available
                        text=text,
                        font_size=font_size,
                        color=regular_color,
                        stroke_color='black',
                        stroke_width=4,  # Thicker stroke for greater impact
                        method='label'
                    )
                    # Set duration using with_duration
                    caption_clip = caption_clip.with_duration(duration)
                    
                    # Add a fade-in/pulse animation
                    caption_clip = caption_clip.with_opacity(
                        lambda t: min(1.0, 1.5 * t) if t < 0.3 else 1.0
                    )
                    
                    logger.info("Created fallback styled caption")
                except Exception as e2:
                    logger.error(f"Failed with fallback styling: {e2}")
                    
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
            
            # Position captions in the lower part of the video, matching the screenshot
            # The screenshot shows captions positioned about 1/3 up from the bottom
            position_y = int(video_height * 0.7)
            
            # Apply final positioning - center horizontally, fixed position vertically
            # Only needed for the fallback implementation (not for word-by-word)
            if 'composite_clips' not in locals() or not composite_clips:
                caption_clip = caption_clip.with_position(('center', position_y))
            
            logger.info(f"Created caption clip with improved styling")
            return caption_clip
            
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
                                # For natural pacing, add a small gap between each chunk (mimics natural speech)
                                # This creates the effect of captions appearing exactly when words are spoken
                                chunk_gap = min(0.3, subclip.duration / (len(chunks) * 4))  # Small gap between chunks
                                
                                # Time each chunk to appear at the right moment with natural pacing
                                # Use a slightly staggered timing for more natural feel (like in the screenshot)
                                if len(chunks) > 1:
                                    # For multiple chunks, stagger them with small gaps in between
                                    # This matches the cadence of the spoken words better
                                    chunk_start = i * ((subclip.duration - (chunk_gap * len(chunks))) / len(chunks))
                                    # Add progressive gap between chunks for natural pacing
                                    chunk_start += i * chunk_gap
                                else:
                                    # For a single chunk, center it in the subclip
                                    chunk_start = (subclip.duration - chunk_duration) / 2
                                
                                # Create caption clip for this chunk with the screenshot style
                                # Reduce duration slightly for a punchier, more animated feel
                                # This makes the captions appear briefly exactly when words are spoken
                                effective_duration = min(chunk_duration, 
                                                       max(1.0, min(2.0, len(chunk_text.split()) * 0.35)))
                                
                                caption = self._create_caption_clip(
                                    chunk_text,
                                    effective_duration,  # Shorter duration for punchier effect like in screenshot
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