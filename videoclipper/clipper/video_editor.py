"""Video editing functionality for creating highlight clips."""

from typing import List, Optional, Tuple, Callable
import random
import logging
import re
import numpy as np
import uuid
import os
import sys

# Ensure tools directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import audio analyzer if possible
try:
    from tools.audio_analyzer import AudioAnalyzer
except ImportError:
    # Fallback if not available
    AudioAnalyzer = None

from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.VideoClip import TextClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

# Try to import moviepy fx using multiple approaches
try:
    from moviepy.video.fx.all import resize, crop, fadein, fadeout, crossfadein
except ImportError:
    try:
        from moviepy.video.fx.resize import resize
        from moviepy.video.fx.crop import crop
        from moviepy.video.fx.fadein import fadein
        from moviepy.video.fx.fadeout import fadeout
        from moviepy.video.fx.crossfadein import crossfadein
    except ImportError:
        # Define fallback functions if imports fail
        def resize(clip, width=None, height=None):
            return clip
        def crop(clip, x1=0, y1=0, width=None, height=None):
            return clip
        def fadein(clip, duration=1.0):
            return clip
        def fadeout(clip, duration=1.0):
            return clip
        def crossfadein(clip, duration=1.0):
            return clip

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


from dataclasses import dataclass

@dataclass
class CaptionConfig:
    """Centralized configuration for caption styling and timing"""
    font: str = 'Arial-Bold'
    font_size: int = 42
    color: str = '#FFFFFF'
    bg_color: str = 'rgba(0, 0, 0, 0.8)'
    highlight_colors: Tuple[str] = ('#FFD700', '#00FF00')  # Gold and Green
    stroke_color: str = '#000000'
    stroke_width: int = 2
    position: Tuple[str, float] = ('center', 0.85)  # (horizontal, vertical %)
    max_words: int = 8
    padding: Tuple[int, int] = (20, 10)  # (horizontal, vertical)
    line_spacing: int = 10
    fade_duration: float = 0.15

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
        self.caption_config = CaptionConfig()  # Initialize caption config

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
                    # Create varied chunk sizes (4, 5, or 6) for better rhythm
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

    def _calculate_caption_timing(self, audio_path: str, text: str) -> List[Tuple[float, float]]:
        """Calculate caption timing using audio analysis and speech rhythm estimation.

        Args:
            audio_path: Path to the audio file (or source video path if no separate audio)
            text: The text to align with the audio

        Returns:
            List of tuples containing the start and end times for each chunk of text
        """
        # First split text into readable caption chunks
        text_chunks = self._split_text_into_chunks(text)
        
        # If no chunks, return empty timing
        total_chunks = len(text_chunks)
        if not total_chunks:
            return []
        
        # Try to use audio analysis for better timing if we have a valid audio path
        use_audio_analysis = False
        audio_file = None
        
        if audio_path and os.path.exists(audio_path):
            # We have audio file - try to use it for better timing
            use_audio_analysis = True
            audio_file = audio_path
        elif self.video_path and os.path.exists(self.video_path):
            # No audio file but we have the video - use that instead
            use_audio_analysis = True
            audio_file = self.video_path
            
        if use_audio_analysis:
            try:
                logger.info(f"Using advanced audio analysis for precise caption timing")
                
                # Try to extract or analyze audio characteristics
                import numpy as np
                
                try:
                    # Try using MoviePy to analyze audio amplitude
                    from moviepy.editor import AudioFileClip
                    
                    # Load audio
                    audio_clip = AudioFileClip(audio_file)
                    
                    # Get total audio duration
                    audio_duration = audio_clip.duration
                    logger.info(f"Audio duration: {audio_duration:.2f}s")
                    
                    # Calculate average words per second based on text and audio duration
                    words = text.split()
                    total_words = len(words)
                    words_per_second = total_words / audio_duration if audio_duration > 0 else 3.0
                    
                    # Adjust for reasonable speaking pace (2-4 words per second is typical)
                    words_per_second = min(max(words_per_second, 2.0), 4.0)
                    logger.info(f"Estimated speaking pace: {words_per_second:.1f} words/second")
                    
                    # Try to detect pauses by analyzing audio amplitude
                    # Sample audio at 10Hz for amplitude analysis (analyzes every 0.1 seconds)
                    sampling_rate = 10
                    sample_count = int(audio_duration * sampling_rate)
                    
                    # Sample audio amplitude at regular intervals
                    amplitude_samples = []
                    for i in range(sample_count):
                        t = i / sampling_rate
                        if t < audio_duration:
                            try:
                                # Get audio frame at this time point (average abs amplitude)
                                amplitude = np.abs(audio_clip.get_frame(t)).mean()
                                amplitude_samples.append((t, amplitude))
                            except Exception:
                                # Skip this sample if extraction fails
                                continue
                                
                    # Detect potential pauses (where amplitude drops significantly)
                    if amplitude_samples:
                        # Calculate mean and threshold for pause detection
                        mean_amplitude = np.mean([amp for _, amp in amplitude_samples])
                        pause_threshold = mean_amplitude * 0.3  # 30% of average amplitude
                        
                        # Find potential pause points
                        pause_points = [t for t, amp in amplitude_samples if amp < pause_threshold]
                        logger.info(f"Detected {len(pause_points)} potential pauses in audio")
                        
                        # Group close pause points together
                        grouped_pauses = []
                        current_group = []
                        
                        for t in sorted(pause_points):
                            if not current_group or t - current_group[-1] < 0.3:  # Group pauses less than 0.3s apart
                                current_group.append(t)
                            else:
                                if current_group:
                                    # Use the middle of the group as the pause point
                                    grouped_pauses.append(sum(current_group) / len(current_group))
                                current_group = [t]
                                
                        # Add the last group
                        if current_group:
                            grouped_pauses.append(sum(current_group) / len(current_group))
                            
                        logger.info(f"Identified {len(grouped_pauses)} distinct pause points")
                        
                        # Now use these pause points to improve caption timing
                        # Use an algorithm that places captions between pauses
                        if grouped_pauses:
                            # Calculate timing based on text chunks and pause points
                            timing = []
                            pause_idx = 0
                            text_position = 0  # Track position in total text
                            
                            for i, chunk in enumerate(text_chunks):
                                # Calculate start time - either at a pause or based on text position
                                if i == 0:
                                    # First chunk starts at beginning
                                    start_time = 0.0
                                elif pause_idx < len(grouped_pauses):
                                    # Use next pause point as a boundary
                                    start_time = grouped_pauses[pause_idx]
                                    pause_idx += 1
                                else:
                                    # Distribute remaining chunks evenly
                                    start_time = text_position / total_words * audio_duration
                                
                                # Update text position
                                text_position += len(chunk.split())
                                
                                # Calculate end time based on words in this chunk
                                words_in_chunk = len(chunk.split())
                                chunk_duration = max(1.0, words_in_chunk / words_per_second)
                                
                                # Ensure reasonable duration (at least 1s, not too long)
                                chunk_duration = min(max(chunk_duration, 1.0), 5.0)
                                
                                # Add timing entry
                                end_time = min(start_time + chunk_duration, audio_duration)
                                timing.append((start_time, end_time))
                                
                            logger.info(f"Created audio-informed timing for {len(text_chunks)} chunks")
                            return timing
                except Exception as e:
                    logger.warning(f"Audio analysis failed: {e}, falling back to text-based timing")
            except Exception as outer_e:
                logger.warning(f"Error in audio-based timing: {outer_e}")
        
        # Fallback to improved text-based timing if audio analysis fails or isn't available
        logger.info("Using improved text-based timing for captions")
        
        # More sophisticated estimation of reading speed and appropriate display times
        # - Shorter phrases need more time per word than longer ones
        # - Account for complexity of language
        # - Ensure minimum caption display time
        
        # Estimate total duration needed based on speaking rate and content
        total_words = sum(len(chunk.split()) for chunk in text_chunks)
        avg_words_per_second = 2.5  # Average speaking rate (adjust if known)
        
        # Calculate natural-sounding non-uniform timing
        timing = []
        current_time = 0.0
        
        for chunk in text_chunks:
            words = chunk.split()
            word_count = len(words)
            
            # Adjust timing based on chunk length
            # - Short chunks (1-3 words) need more time per word
            # - Medium chunks (4-6 words) need standard time
            # - Longer chunks need less time per word but longer overall
            if word_count <= 3:
                # Short phrases need more time per word
                seconds_per_word = 0.7
            elif word_count <= 6:
                # Medium phrases at standard pace
                seconds_per_word = 0.5
            else:
                # Longer phrases slightly faster per word
                seconds_per_word = 0.45
            
            # Check for complex words or punctuation which need more time
            complex_word_count = sum(1 for word in words if len(word) > 7)
            punctuation_count = sum(1 for c in chunk if c in '.,:;?!')
            
            # Add time for complexity factors
            complexity_factor = 1.0 + (complex_word_count * 0.1) + (punctuation_count * 0.2)
            
            # Calculate duration with adjustments (ensure minimum display time)
            this_chunk_duration = max(1.2, word_count * seconds_per_word * complexity_factor)
            
            # Cap at reasonable maximum (avoid overly long captions)
            this_chunk_duration = min(this_chunk_duration, 6.0)
            
            # Add this chunk's timing
            timing.append((current_time, current_time + this_chunk_duration))
            current_time += this_chunk_duration
            
        logger.info(f"Created enhanced text-based timing for {len(text_chunks)} chunks in {current_time:.2f}s")
        return timing

    def create_clip(
        self, segments: List[Segment], output_path: str, max_duration: Optional[float] = None,
        min_segment_duration: Optional[float] = None,  # Minimum segment duration
        max_segment_duration: Optional[float] = None,  # Maximum segment duration
        viral_style: bool = True,  # Enable viral-style edits
        add_captions: bool = True,  # Add captions to the clip
        highlight_keywords: Optional[List[str]] = None,  # Words to highlight in captions
        force_algorithm: bool = False,  # Force use of sophisticated timing algorithm
        target_duration: Optional[float] = None,  # Target duration, fallback to max_duration if not provided
        vertical_format: bool = False,  # Create vertical format video (1080x1920) for shorts/reels
        clip_tag: Optional[str] = None  # Optional tag to identify this specific clip
    ) -> Tuple[str, float]:
        """Create a video clip from the given segments.

        Args:
            segments: List of segments to include in the clip
            output_path: Path to save the output video
            max_duration: Maximum duration of the output clip in seconds
            min_segment_duration: Minimum duration of each segment in seconds
            max_segment_duration: Maximum duration of each segment in seconds
            viral_style: Whether to apply viral-style effects
            add_captions: Whether to add captions to the clips
            highlight_keywords: Words to highlight in captions

        Returns:
            Tuple of (output_path, duration)

        Raises:
            VideoProcessingError: If clip creation fails
        """
        try:
            video = self._load_video()
            
            # Skip audio extraction and Whisper alignment - too slow for shorts/reels
            # Just use existing segments with simpler timing
            aligned_segments = []
            for segment in segments:
                if hasattr(segment, 'text') and segment.text:
                    # Check if segment already has caption timing
                    if hasattr(segment, 'caption_timing') and segment.caption_timing:
                        aligned_segments.append(segment)
                    else:
                        # Calculate simple timing directly - no audio processing needed
                        caption_timing = self._calculate_caption_timing("", segment.text)
                        aligned_segment = Segment(
                            start=segment.start,
                            end=segment.end,
                            text=segment.text,
                            caption_timing=caption_timing,
                            score=segment.score,
                            segment_type=segment.segment_type,
                            metadata=segment.metadata
                        )
                        aligned_segments.append(aligned_segment)
                else:
                    aligned_segments.append(segment)

            output_path = validate_output_path(output_path)

            if not aligned_segments:
                raise VideoProcessingError("No segments provided for clip creation")

            # Sort segments by start time to ensure chronological order
            aligned_segments.sort(key=lambda x: x.start)
            
            # Select segments with more relaxed criteria for shorts/reels
            selected_segments = []
            last_end_time = -5  # Initialize to negative value to ensure first segment is always included
            
            # For vertical format, we need shorter segments with less spacing
            min_spacing = 3 if vertical_format else 10
            
            for segment in aligned_segments:
                start = max(0, segment.start)
                end = min(self._duration or float('inf'), segment.end)
                
                # Skip segments that are too close to the previous one (with adjustable spacing)
                if start < last_end_time + min_spacing and selected_segments:
                    continue
                
                if start < end:
                    selected_segments.append(segment)
                    last_end_time = end
                    
                    # For shorts/reels we need more segments
                    if len(selected_segments) >= (8 if vertical_format else 5):
                        break
            
            # If we don't have enough segments, be less picky about spacing
            if len(selected_segments) < 2:
                logger.info("Not enough segments with spacing criteria, using all valid segments")
                selected_segments = []
                for segment in aligned_segments:
                    start = max(0, segment.start)
                    end = min(self._duration or float('inf'), segment.end)
                    if start < end:
                        selected_segments.append(segment)

            # Extract the relevant subclips with professional transitions and pacing
            subclips = []
            transition_duration = 0.5  # Half-second transitions between segments
            
            # Determine if this is a narrative-style clip (has lots of text)
            narrative_style = any(hasattr(segment, 'text') and segment.text for segment in selected_segments)
            
            # Group segments to avoid jumpy editing - segments that are close should be combined
            # This creates a more flowing edit with fewer jarring cuts
            grouped_segments = []
            current_group = []
            
            # Parameter for what's considered "close enough" to group
            grouping_threshold = 2.0  # 2 seconds or less gets grouped
            
            # Group consecutive segments that are close together
            for i, segment in enumerate(selected_segments):
                if not current_group:
                    current_group.append(segment)
                else:
                    prev_segment = current_group[-1]
                    # If this segment is close to the previous one, add to group
                    if segment.start - prev_segment.end <= grouping_threshold:
                        current_group.append(segment)
                    else:
                        # Start a new group
                        if current_group:
                            grouped_segments.append(current_group)
                        current_group = [segment]
                        
                # Don't forget the last group
                if i == len(selected_segments) - 1 and current_group:
                    grouped_segments.append(current_group)
            
            logger.info(f"Grouped {len(selected_segments)} segments into {len(grouped_segments)} cohesive sequences")
            
            # Now process each group into a subclip with proper transitions
            for group_idx, segment_group in enumerate(grouped_segments):
                # Get the entire span of this group
                group_start = segment_group[0].start
                group_end = segment_group[-1].end
                
                # Ensure boundaries are within video duration
                start = max(0, group_start)
                end = min(self._duration or float('inf'), group_end)
                
                if start < end:
                    # Add padding for transitions
                    padding_start = min(0.7, start) if start > 0 else 0
                    padding_end = min(0.7, (self._duration or float('inf')) - end) if end < (self._duration or float('inf')) else 0
                    
                    # Use expanded boundaries for smoother transitions
                    expanded_start = start - padding_start
                    expanded_end = end + padding_end
                    
                    # Extract the clip with expanded boundaries
                    group_clip = video.subclipped(expanded_start, expanded_end)
                    
                    # Choose transition style based on content and position
                    # First clip: fade in
                    if group_idx == 0 and hasattr(group_clip, 'fadein'):
                        try:
                            # Gentle fade in for first clip
                            group_clip = group_clip.fadein(0.8)  # Slightly longer fade in for opener
                            logger.info("Added opening fade-in effect")
                        except Exception as e:
                            logger.warning(f"Could not add opening fade effect: {e}")
                    
                    # Middle clips: crossfade based on content
                    elif group_idx > 0 and group_idx < len(grouped_segments) - 1 and hasattr(group_clip, 'fadein'):
                        try:
                            # For narrative style, use shorter transitions to maintain flow
                            trans_duration = 0.4 if narrative_style else 0.6
                            group_clip = group_clip.fadein(trans_duration)
                        except Exception as e:
                            logger.warning(f"Could not add middle transition effect: {e}")
                    
                    # Last clip: fade out nicely
                    if group_idx == len(grouped_segments) - 1 and hasattr(group_clip, 'fadeout'):
                        try:
                            # Longer, more gentle fade out for ending
                            group_clip = group_clip.fadeout(1.0)  # 1-second fade out for conclusion
                            logger.info("Added closing fade-out effect")
                        except Exception as e:
                            logger.warning(f"Could not add closing fade effect: {e}")
                    
                    # Don't add the clip yet - we may need to add captions first
                    logger.info(f"Processing group clip {group_idx+1} from {start:.2f}s to {end:.2f}s, duration: {group_clip.duration:.2f}s")
                    
                    # Add captions for this grouped segment clip
                    caption_clips = []
                    
                    # Process captions for each segment in this group
                    for seg_idx, segment in enumerate(segment_group):
                        if add_captions and hasattr(segment, 'text') and segment.text:
                            try:
                                # Get video dimensions
                                video_size = (group_clip.w, group_clip.h)
                                logger.info(f"Adding captions to segment {segment.start:.1f}-{segment.end:.1f}")
                                
                                # Always split text into shorter chunks (4-6 words)
                                chunks = self._split_text_into_chunks(segment.text)
                                logger.info(f"Split text into {len(chunks)} small chunks (4-6 words each)")
                                
                                # Calculate relative position within the group clip
                                # This is necessary because caption timing is relative to clip start
                                rel_start = max(0, segment.start - expanded_start)
                                rel_end = min(group_clip.duration, segment.end - expanded_start)
                                rel_duration = rel_end - rel_start
                                
                                # Ensure we have a valid duration
                                if rel_duration <= 0:
                                    logger.warning(f"Segment {segment.start:.1f}-{segment.end:.1f} is outside the clip bounds, skipping captions")
                                    continue
                                
                                # More precise timing for captions to align with speech
                                # Calculate exact duration and timing for each chunk
                                segment_caption_clips = []
                                
                                # Use a precise speech rate estimate
                                total_words = len(segment.text.split())
                                speech_rate = total_words / rel_duration if rel_duration > 0 else 2.0
                                
                                # Ensure reasonable speech rate (not too fast or slow)
                                if speech_rate > 3.0:  # Too fast
                                    speech_rate = 2.5  # More natural pace
                                elif speech_rate < 1.0:  # Too slow
                                    speech_rate = 1.5  # Reasonable minimum
                                    
                                # Calculate total words to distribute
                                total_chunk_words = sum(len(chunk.split()) for chunk in chunks)
                                
                                # Distribute segment duration based on word count
                                chunk_durations = []
                                for chunk in chunks:
                                    words_in_chunk = len(chunk.split())
                                    # Each chunk gets a proportion of total time based on its word count
                                    # Minimum 1.5 seconds per chunk to ensure readability
                                    proportion = words_in_chunk / max(1, total_chunk_words)
                                    duration = max(1.5, proportion * rel_duration)
                                    chunk_durations.append(min(duration, rel_duration / 2))  # Cap at half the segment duration
                                
                                # Distribute chunks evenly across the segment within the group clip
                                # Use more precise timing to match speech
                                chunk_start_times = []
                                total_chunk_duration = sum(chunk_durations)
                                
                                # Scale timing to fit within the segment's timeframe in the group clip
                                scale_factor = 0.95 * rel_duration / total_chunk_duration
                                chunk_durations = [d * scale_factor for d in chunk_durations]
                                
                                # Calculate start times for each chunk within the segment
                                # These times must be relative to the group clip's start, not just the segment
                                chunk_start_times = []
                                current_time = rel_start  # Start at the segment's relative position
                                small_gap = self.caption_config.fade_duration if hasattr(self, 'caption_config') else 0.15
                                
                                for duration in chunk_durations:
                                    chunk_start_times.append(current_time)
                                    current_time += duration + small_gap
                                
                                # Process each chunk and create caption clips for this segment
                                for i, chunk_text in enumerate(chunks):
                                    if i >= len(chunk_durations) or i >= len(chunk_start_times):
                                        logger.warning(f"Inconsistent chunks/durations count, skipping chunk {i+1}")
                                        continue
                                        
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
                                        segment_caption_clips.append(caption)
                                        logger.info(f"Created caption for chunk {i+1}/{len(chunks)}: '{chunk_text}'")
                                
                                # Add this segment's captions to the overall caption clips
                                caption_clips.extend(segment_caption_clips)
                                
                            except Exception as e:
                                # If caption fails for this segment, just continue
                                logger.error(f"Caption addition failed for segment {segment.start:.1f}-{segment.end:.1f}: {str(e)}")
                                
                    # Add all captions to the group clip
                    if caption_clips:
                        # Composite all caption chunks with the group clip
                        try:
                            all_clips = [group_clip] + caption_clips
                            group_clip = CompositeVideoClip(all_clips, size=(group_clip.w, group_clip.h))
                            logger.info(f"Composited {len(caption_clips)} caption chunks with group clip")
                        except Exception as e:
                            logger.error(f"Failed to composite caption chunks: {e}")
                    else:
                        logger.warning("No caption clips were created for this group")
                    
                    # Add the group clip to our subclips list
                    subclips.append(group_clip)
                    logger.info(f"Added group clip {group_idx+1} to final sequence")

            if not subclips:
                logger.error("No valid subclips could be created, using fallback method")
                # Fallback: create at least one subclip from the video
                try:
                    # Use first 10 seconds of video as fallback
                    fallback_duration = min(10.0, video.duration if video.duration else 10.0)
                    fallback_clip = video.subclipped(0, fallback_duration)
                    subclips = [fallback_clip]
                    logger.info(f"Created fallback clip of {fallback_duration} seconds")
                except Exception as fallback_error:
                    raise VideoProcessingError(f"Failed to create even fallback clip: {fallback_error}")

            # Concatenate the subclips with professional transitions between them
            logger.info(f"Concatenating {len(subclips)} subclips with professional transitions")
            
            try:
                # For professional transitions, use varied transition styles
                # Varied transitions help maintain viewer interest and look more professional
                
                # Define a list of transition types to alternate through
                transition_styles = ['crossfade', 'fade_to_black', 'fade_with_flash', 'none']
                transitions_clips = []
                
                # Process subclips with transitions
                for i, clip in enumerate(subclips):
                    # Determine transition style for this clip - rotate through styles
                    transition_style = transition_styles[i % len(transition_styles)]
                    
                    # Short clips need special handling
                    if clip.duration < 1.0:
                        # Just use the clip with no transition if it's too short
                        transitions_clips.append(clip)
                        continue
                    
                    # Apply transition based on style
                    if transition_style == 'crossfade' and i > 0:
                        # Standard crossfade - varies by position in sequence
                        # Shorter at beginning, longer towards end for storytelling
                        trans_duration = min(0.4 + (i * 0.05), 0.7)  # Gradually increase duration
                        try:
                            if hasattr(clip, 'crossfadein'):
                                transitions_clips.append(clip.crossfadein(trans_duration))
                            else:
                                transitions_clips.append(clip.fadein(trans_duration))
                        except Exception:
                            # Fallback if transition fails
                            transitions_clips.append(clip)
                            
                    elif transition_style == 'fade_to_black' and i > 0:
                        # Create fade to black, then fade in
                        try:
                            # Apply fade out to previous clip if possible
                            if i > 0 and i-1 < len(transitions_clips) and hasattr(transitions_clips[i-1], 'fadeout'):
                                transitions_clips[i-1] = transitions_clips[i-1].fadeout(0.5)
                            
                            # Add fade in to current clip
                            if hasattr(clip, 'fadein'):
                                transitions_clips.append(clip.fadein(0.6))  # Slightly longer for dramatic effect
                            else:
                                transitions_clips.append(clip)
                        except Exception:
                            transitions_clips.append(clip)
                            
                    elif transition_style == 'fade_with_flash' and i > 0:
                        # Create a stylistic quick-fade transition (mimics camera flash)
                        try:
                            if hasattr(clip, 'fadein'):
                                # Very quick fade for stylistic effect
                                flash_clip = clip.fadein(0.2)
                                transitions_clips.append(flash_clip)
                            else:
                                transitions_clips.append(clip)
                        except Exception:
                            transitions_clips.append(clip)
                    else:
                        # No transition - direct cut for variety
                        # Sometimes a clean cut is more effective, especially for action
                        transitions_clips.append(clip)
                
                # Combine clips with the applied transitions
                final_clip = concatenate_videoclips(transitions_clips, method="compose")
                logger.info("Created clip with professional varied transitions between segments")
            except Exception as e:
                logger.warning(f"Could not create professional transitions: {e}, using standard concatenation")
                # Fallback to standard concatenation
                final_clip = concatenate_videoclips(subclips)
            
            # Professional duration handling - no looping, create properly paced clips
            if target_duration is not None:
                actual_duration = final_clip.duration
                logger.info(f"Final clip duration: {actual_duration}s, target: {target_duration}s")
                
                # If clip is too short, NEVER loop or repeat content - that's unprofessional
                # Instead, find additional segments from the source video
                if actual_duration < target_duration * 0.8:  # If less than 80% of target
                    logger.info(f"Clip too short ({actual_duration}s), finding additional segments to reach target")
                    
                    # Calculate how much more time we need
                    additional_time_needed = target_duration - actual_duration
                    logger.info(f"Need {additional_time_needed:.1f}s of additional content")
                    
                    try:
                        # Store current clip as we'll need to concatenate with new content
                        current_clip = final_clip
                        
                        # Find additional segments from the video, avoiding used segments
                        avoid_times = [(sub.start, sub.end) for sub in subclips]
                        video_duration = self._duration or 0
                        
                        # Create filler segments from unused parts of the video
                        filler_segments = []
                        
                        # Strategy 1: Try to find interesting moments adjacent to our current clips
                        # Look for segments just before or after our existing segments
                        for seg_start, seg_end in avoid_times:
                            # Look before this segment - buffer of 5-10 seconds
                            pre_start = max(0, seg_start - 15)
                            pre_end = max(0, seg_start - 1)
                            
                            if pre_end > pre_start + 2:  # At least 2 seconds
                                # Check if this segment overlaps with any segments we're avoiding
                                if not any(start <= pre_end and end >= pre_start for start, end in avoid_times):
                                    filler_segments.append((pre_start, pre_end, 0.6))  # Medium score
                            
                            # Look after this segment - buffer of 5-10 seconds
                            post_start = min(video_duration, seg_end + 1)
                            post_end = min(video_duration, seg_end + 15)
                            
                            if post_end > post_start + 2:  # At least 2 seconds
                                # Check if this segment overlaps with any segments we're avoiding
                                if not any(start <= post_end and end >= post_start for start, end in avoid_times):
                                    filler_segments.append((post_start, post_end, 0.6))  # Medium score
                        
                        # Strategy 2: Sample evenly across the video
                        if not filler_segments or sum(end-start for start, end, _ in filler_segments) < additional_time_needed:
                            # Create 10 segments evenly distributed across video
                            segment_count = 10
                            for i in range(segment_count):
                                seg_start = (video_duration * i) / segment_count
                                seg_end = seg_start + min(10, video_duration / segment_count * 0.8)
                                
                                # Check if this segment overlaps with any segments we're avoiding
                                if not any(start <= seg_end and end >= seg_start for start, end in avoid_times):
                                    filler_segments.append((seg_start, seg_end, 0.5))  # Lower score
                        
                        # Sort by score (highest first)
                        filler_segments.sort(key=lambda x: -x[2])
                        
                        # Create subclips for additional segments
                        additional_clips = []
                        additional_duration = 0
                        
                        for start, end, _ in filler_segments:
                            # Only add if we still need more time
                            if additional_duration >= additional_time_needed:
                                break
                                
                            # Add some padding for smooth transitions
                            padding = 0.5
                            clip_start = max(0, start - padding)
                            clip_end = min(video_duration, end + padding)
                            
                            # Extract the clip
                            try:
                                new_clip = self._video.subclipped(clip_start, clip_end)
                                
                                # Add fade transitions
                                if hasattr(new_clip, 'fadein') and hasattr(new_clip, 'fadeout'):
                                    new_clip = new_clip.fadein(0.5).fadeout(0.5)
                                
                                additional_clips.append(new_clip)
                                additional_duration += new_clip.duration
                                logger.info(f"Added segment {clip_start:.1f}-{clip_end:.1f} ({new_clip.duration:.1f}s)")
                            except Exception as e:
                                logger.warning(f"Could not add segment {start:.1f}-{end:.1f}: {e}")
                                continue
                        
                        # If we found additional content, add it to the clip
                        if additional_clips:
                            # Create the final clip by combining current and additional clips
                            # Use crossfades between clips for smooth transitions
                            all_clips = [current_clip] + additional_clips
                            final_clip = concatenate_videoclips(all_clips, method="compose")
                            logger.info(f"Extended clip with {len(additional_clips)} new segments to {final_clip.duration:.1f}s")
                        else:
                            logger.warning("Could not find additional segments, using original clip")
                    except Exception as e:
                        logger.error(f"Error adding more content: {e}")
                        # Keep original clip, don't loop
                
                # If clip is too long, intelligently trim it to match target duration
                elif actual_duration > target_duration * 1.1:  # If more than 110% of target
                    logger.info(f"Clip too long ({actual_duration}s), intelligently trimming to target duration")
                    
                    # Keep the beginning and end intact, trim from the middle if possible
                    if actual_duration > target_duration + 10:  # If we need to trim more than 10 seconds
                        # Keep first 40% and last 40% intact
                        keep_start_duration = target_duration * 0.4
                        keep_end_duration = target_duration * 0.6
                        
                        try:
                            # Extract start and end portions
                            start_clip = final_clip.subclipped(0, keep_start_duration)
                            end_clip = final_clip.subclipped(actual_duration - keep_end_duration, actual_duration)
                            
                            # Add crossfade between them
                            if hasattr(end_clip, 'crossfadein'):
                                end_clip = end_clip.crossfadein(1.0)
                            
                            # Combine the clips
                            final_clip = concatenate_videoclips([start_clip, end_clip], method="compose")
                            logger.info(f"Trimmed middle portion, new duration: {final_clip.duration:.1f}s")
                        except Exception as e:
                            logger.error(f"Failed to trim middle: {e}, using straight cut")
                            # Fall back to simpler trim
                            final_clip = final_clip.subclipped(0, target_duration)
                    else:
                        # Just trim the end for minimal adjustment
                        final_clip = final_clip.subclipped(0, target_duration)
                        
                    logger.info(f"Trimmed clip duration to {final_clip.duration:.1f}s")
            
            # Handle vertical format for shorts/reels if requested
            if vertical_format:
                try:
                    logger.info("Creating vertical format video (1080x1920) for shorts/reels")
                    
                    # Original dimensions
                    orig_w, orig_h = final_clip.size
                    
                    # Calculate scaling and cropping for vertical format
                    if orig_w > orig_h:
                        # For landscape videos, crop to focus on center portion and then resize
                        crop_width = int(orig_h * 9 / 16)  # 9:16 aspect ratio from height
                        crop_x1 = int((orig_w - crop_width) / 2)  # Center crop
                        
                        # Crop to square-ish center area
                        cropped_clip = crop(final_clip, x1=crop_x1, width=crop_width)
                        
                        # Resize to 1080x1920
                        final_clip = resize(cropped_clip, height=1920, width=1080)
                    else:
                        # For portrait videos, just resize to 1080x1920 maintaining aspect ratio
                        # and add black bars if needed
                        new_height = 1920
                        new_width = min(1080, int(orig_w * new_height / orig_h))
                        
                        # Resize while maintaining aspect ratio
                        resized_clip = resize(final_clip, height=new_height, width=new_width)
                        
                        # Create a black background of 1080x1920
                        bg = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=resized_clip.duration)
                        
                        # Composite the resized clip centered on the background
                        final_clip = CompositeVideoClip([bg, resized_clip.with_position("center")], size=(1080, 1920))
                    
                    logger.info(f"Converted to vertical format: 1080x1920")
                except Exception as resize_error:
                    logger.error(f"Failed to convert to vertical format: {resize_error}")
                    logger.info("Continuing with original format")
                    # Continue with original format rather than failing
            
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
        min_segment_duration: Optional[float] = None,  # Minimum segment duration
        max_segment_duration: Optional[float] = None,  # Maximum segment duration
        top_segments: Optional[int] = None,
        vertical_format: bool = False  # Create vertical format video (1080x1920) for shorts/reels
    ) -> Tuple[str, float]:
        """Create a highlight clip from the most interesting segments.

        Args:
            segments: List of segments to consider for the highlight
            output_path: Path to save the output video
            max_duration: Maximum duration of the highlight clip in seconds
            min_segment_duration: Minimum duration of each segment in seconds
            max_segment_duration: Maximum duration of each segment in seconds
            top_segments: Optional number of top segments to include
            vertical_format: Whether to create a vertical format video (1080x1920) for shorts/reels

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
                min_segment_duration=min_segment_duration,  # Use provided parameter
                max_segment_duration=max_segment_duration,  # Use provided parameter
            )

            # Process segments
            processed_segments = selector.process_segments(segments)

            # Select top segments
            selected_segments = selector.select_top_segments(
                processed_segments, max_duration, top_segments
            )

            # Create the clip with vertical format if requested
            return self.create_clip(
                selected_segments, 
                output_path, 
                min_segment_duration=min_segment_duration,
                max_segment_duration=max_segment_duration,
                vertical_format=vertical_format
            )

        except Exception as e:
            raise VideoProcessingError(f"Failed to create highlight clip: {e}")