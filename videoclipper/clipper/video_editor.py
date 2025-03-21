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
    
    def _split_text_into_chunks(self, text: str, max_words: int = 5) -> List[str]:
        """Splits text into smaller chunks for better caption timing and readability.
        
        Args:
            text: The text to split into chunks
            max_words: Maximum number of words per chunk (4-6 words per screen as requested)
            
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
            
            # For very short sentences (4-6 words), keep them intact
            if len(words) <= max_words + 1:  # Allow up to 6 words per chunk
                chunks.append(" ".join(words))
                continue
            
            # For longer sentences, try to create natural 2-line phrases
            # Look for natural breaking points like commas, conjunctions
            natural_breaks = [i for i, word in enumerate(words) 
                            if word.endswith(',') or word.lower() in ['and', 'but', 'or', 'because', 'so']]
            
            if natural_breaks:
                # Use natural breaking points to create chunks
                current_start = 0
                for break_point in natural_breaks:
                    # Only create chunk if it's not too long (max 6 words)
                    if break_point - current_start + 1 <= max_words + 1:
                        chunk = words[current_start:break_point + 1]
                        chunks.append(" ".join(chunk))
                        current_start = break_point + 1
                    
                # Add any remaining words
                if current_start < len(words):
                    remaining = words[current_start:]
                    # Process remaining words in groups of 4-6 words
                    for i in range(0, len(remaining), max_words):
                        sub_chunk = remaining[i:i+max_words]
                        if sub_chunk:
                            chunks.append(" ".join(sub_chunk))
            else:
                # No natural breaks, so split into chunks of 4-6 words
                # Try to keep a good rhythm with alternating lengths
                i = 0
                while i < len(words):
                    # Create varied chunk sizes (4, 5, or 6 words) for better rhythm
                    if i + max_words + 1 <= len(words):  # Can fit a full chunk
                        # Determine chunk size (4, 5, or 6) based on the sentence structure
                        # Use a smaller chunk if the next word is a natural break point
                        for chunk_size in range(max_words + 1, max_words - 1, -1):
                            if i + chunk_size < len(words) and (words[i + chunk_size - 1].endswith(',') or 
                               words[i + chunk_size - 1].endswith('.') or 
                               words[i + chunk_size - 1].endswith('?') or
                               words[i + chunk_size - 1].endswith('!')):
                                chunks.append(" ".join(words[i:i+chunk_size]))
                                i += chunk_size
                                break
                        else:
                            # No natural break found, use standard size (5 words)
                            chunks.append(" ".join(words[i:i+max_words]))
                            i += max_words
                    else:
                        # Final chunk with remaining words
                        chunks.append(" ".join(words[i:]))
                        i = len(words)
        
        # Ensure each chunk has at least one significant word that can be highlighted
        refined_chunks = []
        for chunk in chunks:
            words = chunk.split()
            
            # Only process chunks that have enough words
            if len(words) > 0:
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
                    if refined_chunks and len(refined_chunks[-1].split()) + len(words) <= max_words + 1:
                        refined_chunks[-1] = refined_chunks[-1] + " " + chunk
                    else:
                        refined_chunks.append(chunk)
            else:
                # Skip empty chunks
                continue
                
        # If we ended up with no chunks, return the original text as one chunk
        if not refined_chunks:
            return [text]
            
        return refined_chunks

    def _create_caption_clip(self, text: str, duration: float, video_size: Tuple[int, int], 
                        highlight_words: Optional[List[str]] = None) -> TextClip:
        """Create a styled caption clip with word highlighting like the reference screenshot.
        
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
            
            # Styling parameters - make text readable but not too large
            font_size = min(int(video_height * 0.08), 35)  # More reasonable font size for better readability
            
            try:
                # PRO APPROACH: Create individual word clips with colored text for highlighting
                words = text.split()
                word_clips = []
                
                # Find highlight words
                highlight_indices = []
                
                # Try to match with provided keywords first
                if highlight_words and len(highlight_words) > 0:
                    for i, word in enumerate(words):
                        clean_word = word.strip('.,!?;:"\'()[]{}').lower()
                        if any(hw.lower() in clean_word for hw in highlight_words):
                            highlight_indices.append(i)
                
                # If no highlight words matched, try to find significant words
                if not highlight_indices and len(words) > 2:
                    for i, word in enumerate(words):
                        clean_word = word.strip('.,!?;:"\'()[]{}').lower()
                        if len(clean_word) > 3 and clean_word.lower() not in STOPWORDS:
                            highlight_indices.append(i)
                            # Just highlight one significant word if no keywords matched
                            break
                
                # Log highlighted words with their colors
                if highlight_indices:
                    highlighted_words = []
                    for i, idx in enumerate(highlight_indices):
                        color = "YELLOW" if i % 2 == 0 else "GREEN"
                        highlighted_words.append(f"{words[idx]} ({color})")
                    logger.info(f"Words to highlight: {highlighted_words}")
                
                # Create clips for each word
                word_clips = []
                word_sizes = []
                spacing = int(font_size * 0.3)  # Space between words
                
                for i, word in enumerate(words):
                    # Determine if this word should be highlighted
                    is_highlight = (i in highlight_indices)
                    
                    # Choose color - alternate between yellow and green for highlights
                    if is_highlight:
                        highlight_index = highlight_indices.index(i) if i in highlight_indices else 0
                        word_color = "#FFFF00" if highlight_index % 2 == 0 else "#00FF00"  # Yellow/Green alternating
                    else:
                        word_color = "white"
                    
                    # Add space to all words except the last one
                    display_word = word + (" " if i < len(words) - 1 else "")
                    
                    # Create word clip
                    word_clip = TextClip(
                        text=display_word,
                        font='Arial',
                        font_size=font_size,
                        color=word_color,
                        stroke_color="black", 
                        stroke_width=1 if is_highlight else 2,  # Lighter stroke for highlighted words to look better
                        method="label"
                    ).with_duration(duration)
                    
                    # Store the clip and its size
                    word_clips.append(word_clip)
                    word_sizes.append((word_clip.w, word_clip.h))
                
                # Calculate the total width and maximum height
                total_width = sum(w for w, h in word_sizes)
                max_height = max(h for w, h in word_sizes) if word_sizes else 0
                
                # Create a background for the text
                bg_width = int(min(total_width * 1.1, video_width * 0.85))
                bg_height = int(max_height * 1.3)
                
                # Create background clip
                bg = ColorClip(
                    size=(bg_width, bg_height),
                    color=(0, 0, 0),
                    duration=duration
                ).with_opacity(0.9)  # More opaque background
                
                # Position words horizontally on the background
                composite_clips = [bg]
                x_position = (bg_width - total_width) // 2  # Center text horizontally
                
                for word_clip, (w, h) in zip(word_clips, word_sizes):
                    # Position word at the right x position and center vertically
                    positioned_clip = word_clip.with_position((x_position, (bg_height - h) // 2))
                    composite_clips.append(positioned_clip)
                    x_position += w  # Move to the position for the next word
                
                # Create composite of background and word clips
                caption_composite = CompositeVideoClip(
                    composite_clips,
                    size=(bg_width, bg_height)
                ).with_duration(duration)
                
                # Position the whole caption at the bottom of the screen
                positioned_caption = caption_composite.with_position(
                    ("center", int(video_height * 0.88))  # Position lower on screen
                )
                
                logger.info("Created professional caption with word-by-word highlighting")
                return positioned_caption
                
            except Exception as e:
                logger.error(f"Failed to create word-by-word highlighted caption: {e}")
                
                # FALLBACK: Create a single text clip with highlight markers
                try:
                    words = text.split()
                    highlight_indices = []
                    
                    # Find words to highlight
                    if highlight_words and len(highlight_words) > 0:
                        for i, word in enumerate(words):
                            clean_word = word.strip('.,!?;:"\'()[]{}').lower()
                            if any(hw.lower() in clean_word for hw in highlight_words):
                                highlight_indices.append(i)
                    
                    # If no matches, try to find significant words
                    if not highlight_indices and len(words) > 2:
                        for i, word in enumerate(words):
                            clean_word = word.strip('.,!?;:"\'()[]{}').lower()
                            if len(clean_word) > 3 and clean_word.lower() not in STOPWORDS:
                                highlight_indices.append(i)
                                break
                    
                    # Create a marked version of the text for highlighting
                    if highlight_indices:
                        for idx in sorted(highlight_indices, reverse=True):
                            words[idx] = f"**{words[idx]}**"
                    
                    marked_text = " ".join(words)
                    clean_text = text
                    
                    # Create the caption
                    caption_text = TextClip(
                        text=clean_text,
                        font='Arial',
                        font_size=font_size,
                        color="white",
                        stroke_color="black", 
                        stroke_width=2,
                        method="label"
                    )
                    
                    # Create a background for the caption
                    bg_width = int(min(caption_text.w * 1.1, video_width * 0.85))
                    bg_height = int(caption_text.h * 1.3)
                    
                    bg = ColorClip(
                        size=(bg_width, bg_height),
                        color=(0, 0, 0),
                        duration=duration
                    ).with_opacity(0.9)
                    
                    # Position caption text on background
                    text_on_bg = CompositeVideoClip(
                        [bg, caption_text.with_position("center")],
                        size=(bg_width, bg_height)
                    )
                    
                    # Position the whole caption at the bottom of the screen
                    caption_with_bg = text_on_bg.with_position(
                        ("center", int(video_height * 0.88))
                    ).with_duration(duration)
                    
                    logger.info("Created fallback caption with basic text")
                    return caption_with_bg
                    
                except Exception as e2:
                    logger.error(f"Failed to create fallback caption: {e2}")
                    
                    # SUPER FALLBACK: Just basic text with no styling
                    try:
                        caption_clip = TextClip(
                            text=text,
                            font="Arial",
                            font_size=font_size,
                            color="white",
                            stroke_color="black",
                            stroke_width=3,
                            method="label" 
                        )
                        
                        # Set duration and position
                        caption_clip = caption_clip.with_duration(duration)
                        caption_clip = caption_clip.with_position(
                            ("center", int(video_height * 0.85))
                        )
                        
                        logger.info("Created simple fallback caption")
                        return caption_clip
                        
                    except Exception as e3:
                        logger.error(f"Failed to create super fallback caption: {e3}")
                        return None
            
        except Exception as e:
            logger.error(f"Caption creation failed: {str(e)}")
            return None
    
    def create_clip(
        self, segments: List[Segment], output_path: str, max_duration: Optional[float] = None,
        viral_style: bool = True,  # Enable viral-style edits
        add_captions: bool = True,  # Add captions to the clip
        highlight_keywords: Optional[List[str]] = None,  # Words to highlight in captions
        force_algorithm: bool = False,  # Force use of sophisticated timing algorithm
        target_duration: Optional[float] = None  # Target duration, fallback to max_duration if not provided
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
                            # More precise timing for captions to align with speech
                            # Calculate exact duration and timing for each chunk
                            
                            # Use a precise speech rate estimate
                            total_words = len(segment.text.split())
                            speech_rate = total_words / subclip.duration if subclip.duration > 0 else 2.0
                            
                            # Ensure reasonable speech rate (not too fast or slow)
                            if speech_rate > 3.0:  # Too fast
                                speech_rate = 2.5  # More natural pace
                            elif speech_rate < 1.0:  # Too slow
                                speech_rate = 1.5  # Reasonable minimum
                                
                            # Calculate total words to distribute
                            total_chunk_words = sum(len(chunk.split()) for chunk in chunks)
                            
                            # Distribute video duration based on word count
                            chunk_durations = []
                            for chunk in chunks:
                                words_in_chunk = len(chunk.split())
                                # Each chunk gets a proportion of total time based on its word count
                                # Minimum 1.5 seconds per chunk to ensure readability
                                proportion = words_in_chunk / max(1, total_chunk_words)
                                duration = max(1.5, proportion * subclip.duration)
                                chunk_durations.append(min(duration, subclip.duration / 2))  # Cap at half the subclip duration
                            
                            # Distribute chunks evenly across the entire subclip
                            # Use more precise timing to match speech
                            chunk_start_times = []
                            total_duration = sum(chunk_durations)
                            
                            # Distribute chunks evenly if their total duration is less than clip duration
                            if total_duration < subclip.duration:
                                spacing = (subclip.duration - total_duration) / max(1, len(chunks) - 1)
                                current_time = 0
                                
                                for duration in chunk_durations:
                                    chunk_start_times.append(current_time)
                                    current_time += duration + spacing
                            else:
                                # If chunks would be too long, scale them down proportionally
                                scale_factor = 0.95 * subclip.duration / total_duration
                                chunk_durations = [d * scale_factor for d in chunk_durations]
                                
                                current_time = 0
                                small_gap = 0.1  # Small gap between chunks
                                
                                for duration in chunk_durations:
                                    chunk_start_times.append(current_time)
                                    current_time += duration + small_gap
                            
                            # Process each chunk and create caption clips
                            for i, chunk_text in enumerate(chunks):
                                # Choose appropriate keywords for this chunk
                                chunk_keywords = []
                                
                                # Extract keywords specific to this chunk
                                if highlight_keywords is None:
                                    # Try to find significant words in this chunk
                                    chunk_words = chunk_text.split()
                                    for word in chunk_words:
                                        clean_word = word.strip('.,!?;:"\'()[]{}')
                                        if len(clean_word) > 3 and clean_word.lower() not in STOPWORDS:
                                            chunk_keywords.append(clean_word)
                                            break
                                    
                                    # If no keywords found, use extracted keywords from whole segment
                                    if not chunk_keywords:
                                        chunk_keywords = self._extract_keywords(chunk_text)
                                else:
                                    # Use provided keywords that appear in this chunk
                                    for word in highlight_keywords:
                                        if word.lower() in chunk_text.lower():
                                            chunk_keywords.append(word)
                                
                                # Create caption clip with exactly one highlighted word (if possible)
                                caption = self._create_caption_clip(
                                    chunk_text,
                                    chunk_durations[i],  # Use calculated duration for better timing
                                    video_size,
                                    chunk_keywords
                                )
                                
                                if caption:
                                    # Start this caption at the calculated time using chunk_start_times
                                    caption = caption.with_start(chunk_start_times[i])
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