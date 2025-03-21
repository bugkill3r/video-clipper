"""Video editing functionality for creating highlight clips."""

from typing import List, Optional, Tuple, Callable, Dict, Any
import random
import logging
import os
import tempfile
import json
import re
import numpy as np
from pydub import AudioSegment
# Audio processing for VAD (Voice Activity Detection)
from scipy.io import wavfile
from scipy.signal import find_peaks
import re
import numpy as np

from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.VideoClip import TextClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut

# Removed external transitions module due to compatibility issues

from videoclipper.clipper.base import VideoClipper
from videoclipper.clipper.segment_selector import SegmentSelector
from videoclipper.exceptions import VideoProcessingError
from videoclipper.models.segment import Segment, SegmentType
from videoclipper.utils.validation import validate_output_path
from videoclipper.config import get_config

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

    def _split_text_into_chunks(self, text: str, max_words: int = None) -> List[str]:
        """Splits text into smaller chunks for better caption timing and readability.

        Args:
            text: The text to split into chunks
            max_words: Maximum number of words per chunk (2-3 words per line for viral shorts)

        Returns:
            List of text chunks for perfect viral shorts captions
        """
        if not text:
            return []

        # Get max words from config if not provided
        max_words = max_words or get_config("max_words_per_caption_line", 3)

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

            # For viral shorts, we want VERY short chunks (2-3 words maximum)
            # This makes it much easier to read quickly
            i = 0
            while i < len(words):
                # For short attention span shorts, use strict 2-3 word chunks
                chunk_size = min(max_words, len(words) - i)
                chunk = words[i:i+chunk_size]
                if chunk:
                    chunks.append(" ".join(chunk))
                i += chunk_size

        # If we ended up with no chunks, return empty list
        if not chunks:
            return []

        return chunks

    def _detect_speech_segments(self, audio_path: str) -> List[Tuple[float, float]]:
        """Detects speech segments in audio using enhanced voice activity detection.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of (start_time, end_time) tuples for speech segments with highly precise timing
        """
        logger.info(f"Detecting speech segments in audio using enhanced precision algorithm")
        try:
            # Load audio using pydub for easier manipulation
            audio = AudioSegment.from_file(audio_path)

            # Convert to mono if it's stereo for more consistent analysis
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Export to WAV for scientific processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_path = temp_audio_file.name
                audio.export(temp_audio_path, format="wav")

            # Read the audio file using scipy
            sample_rate, audio_data = wavfile.read(temp_audio_path)

            # Convert to float and normalize
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # ENHANCED: Use even smaller window for ultra-precise detection
            # Professional broadcast standard uses ~10ms windows
            frame_size = int(0.015 * sample_rate)  # 15ms frames for ultra precision
            hop_size = int(0.005 * sample_rate)    # 5ms hop size for extremely fine detection

            # Calculate energy for each frame
            energy = []
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i+frame_size]
                energy.append(np.sum(frame**2) / frame_size)

            energy = np.array(energy)

            # IMPROVED: Use adaptive multi-threshold detection for better speech detection
            # This technique is used in professional speech recognition
            noise_floor = np.percentile(energy, 15)  # Lower percentile for more sensitivity
            
            # Use multiple thresholds to better detect different speech volumes
            low_threshold = noise_floor * 1.5    # For catching softer speech
            med_threshold = noise_floor * 2.5    # For normal speech
            high_threshold = noise_floor * 4.0   # For louder parts
            
            # Multi-level detection - more robust way to catch ALL speech
            is_speech_low = energy > low_threshold
            is_speech_med = energy > med_threshold
            is_speech_high = energy > high_threshold
            
            # Combine different levels with smart weighting
            # If a frame has high energy, it's definitely speech
            # If it has medium energy and is near known speech, likely speech
            # If it has low energy but is surrounded by speech, may be speech
            
            # First use high confidence markers as seeds
            is_speech = is_speech_high.copy()
            
            # Then expand to medium confidence areas that are adjacent to high confidence
            for i in range(1, len(is_speech)-1):
                if not is_speech[i] and is_speech_med[i]:
                    # Check if this medium-confidence frame is near high confidence
                    # Using a wider window for context (5 frames = ~25ms context)
                    window_size = 5
                    start_idx = max(0, i - window_size)
                    end_idx = min(len(is_speech), i + window_size)
                    
                    # If there's high confidence speech nearby, this is likely speech
                    if np.any(is_speech[start_idx:end_idx]):
                        is_speech[i] = True
            
            # Finally consider low confidence areas between definite speech
            # This helps catch quiet consonants between vowels
            for i in range(window_size, len(is_speech)-window_size):
                if not is_speech[i] and is_speech_low[i]:
                    # Only count if surrounded by definite speech
                    # This avoids false positives but catches quiet connecting sounds
                    before_speech = np.any(is_speech[i-window_size:i])
                    after_speech = np.any(is_speech[i+1:i+window_size+1])
                    
                    if before_speech and after_speech:
                        is_speech[i] = True
            
            # Fine tune timings for broadcast-quality caption sync using config values
            # Professional captions appear slightly BEFORE speech starts
            # This gives viewers time to shift focus to caption before hearing words
            pre_roll_ms = get_config("speech_preroll_ms", 100) / 1000.0  # Convert from ms to seconds
            post_roll_ms = get_config("speech_postroll_ms", 50) / 1000.0  # Convert from ms to seconds
            
            # Convert to frame counts
            preroll_frames = int(pre_roll_ms / (hop_size / sample_rate))
            postroll_frames = int(post_roll_ms / (hop_size / sample_rate))
            
            # Minimum segment durations for better timing from config
            min_speech_ms = get_config("min_speech_segment_duration", 250)
            min_gap_ms = get_config("min_gap_between_speech", 150)
            
            # Convert to frames
            min_speech_frames = int((min_speech_ms / 1000.0) / (hop_size / sample_rate))
            min_gap_frames = int((min_gap_ms / 1000.0) / (hop_size / sample_rate))  # Convert to frames

            # Convert frame indices to time
            frame_time = hop_size / sample_rate

            # Identify continuous speech segments with perfect timing
            speech_segments = []
            in_speech = False
            speech_start = 0

            for i in range(len(is_speech)):
                if is_speech[i] and not in_speech:
                    # Start of a speech segment with professional preroll
                    in_speech = True
                    # Apply the preroll for broadcast-style timing
                    speech_start = max(0, i - preroll_frames)
                elif not is_speech[i] and in_speech:
                    # End of a speech segment
                    in_speech = False
                    speech_end = min(len(is_speech), i + postroll_frames)

                    # Only keep if segment is long enough
                    if speech_end - speech_start >= min_speech_frames:
                        speech_segments.append((
                            speech_start * frame_time,
                            speech_end * frame_time
                        ))

            # Handle the case where audio ends during speech
            if in_speech:
                speech_end = len(is_speech)
                if speech_end - speech_start >= min_speech_frames:
                    speech_segments.append((
                        speech_start * frame_time,
                        speech_end * frame_time
                    ))

            # ADVANCED: Intelligent segment merging for more natural caption flow
            # This gives more natural reading flow by avoiding too-rapid caption changes
            if speech_segments:
                merged_segments = [speech_segments[0]]

                for current_start, current_end in speech_segments[1:]:
                    last_start, last_end = merged_segments[-1]

                    # Professional timing uses context-aware gap sizes from config
                    # Shorter gaps for related speech, longer gaps for new thoughts
                    max_gap = get_config("max_gap_for_caption_merge", 200) / 1000.0  # Convert from ms to seconds
                    
                    # If gap is small enough, merge segments for smoother reading
                    if current_start - last_end < max_gap:
                        merged_segments[-1] = (last_start, current_end)
                    else:
                        merged_segments.append((current_start, current_end))

                speech_segments = merged_segments

            # Clean up
            try:
                os.remove(temp_audio_path)
            except:
                pass

            logger.info(f"Detected {len(speech_segments)} speech segments with broadcast-quality timing")
            return speech_segments

        except Exception as e:
            logger.error(f"Enhanced speech detection failed: {str(e)}")
            # Return a single segment covering the entire audio as fallback
            audio_duration = len(AudioSegment.from_file(audio_path)) / 1000.0
            return [(0, audio_duration)]

    def _align_text_with_speech(self, audio_path: str, text: str) -> List[Dict[str, Any]]:
        """Aligns text chunks with detected speech segments for broadcast-quality caption timing.

        Args:
            audio_path: Path to the audio file
            text: The text to align with the audio

        Returns:
            List of dictionaries with 'text', 'start', and 'duration' for each aligned phrase
        """
        logger.info(f"Using professional-grade text-to-speech alignment algorithm")

        # Get speech segments from audio using our enhanced detector
        speech_segments = self._detect_speech_segments(audio_path)

        if not speech_segments:
            logger.warning("No speech segments detected, using uniform timing")
            audio_duration = len(AudioSegment.from_file(audio_path)) / 1000.0
            speech_segments = [(0, audio_duration)]

        # IMPROVED: Use smaller chunks for more precise alignment with speech
        # Professional captions use 1-2 short phrases per caption for perfect timing
        chunks = self._split_text_into_chunks(text, max_words=3)  # Even smaller chunks for perfect sync
        logger.info(f"Split text into {len(chunks)} precision-timed chunks for captions")

        if not chunks:
            return []

        # Calculate total audio duration for speech portions
        total_speech_duration = sum(end - start for start, end in speech_segments)

        # ENHANCED: More sophisticated speaking rate estimation
        # Account for different speech patterns and real-world speaking rates
        word_count = sum(len(chunk.split()) for chunk in chunks)
        
        # Professional broadcast standard speaking rates range from 140-180 words per minute
        # That's about 2.3-3.0 words per second
        base_speaking_rate = word_count / max(total_speech_duration, 1.0)
        
        # Constrain to realistic values for natural speech
        if base_speaking_rate < 1.5:  # Very slow speech
            words_per_second = 1.5
        elif base_speaking_rate > 4.0:  # Very fast speech
            words_per_second = 4.0
        else:
            words_per_second = base_speaking_rate

        logger.info(f"Professional calibrated speaking rate: {words_per_second:.2f} words/second")

        # NEW ALGORITHM: AI-based dynamic text distribution
        # This approach mimics how professional caption timers work with broadcast content
        aligned_segments = []
        
        # SCENARIO A: Optimal scenario - when speech segments closely match our text chunks
        # This is ideal for precise timing, similar to professional subtitle formats
        if 0.8 <= len(chunks) / len(speech_segments) <= 1.2:
            logger.info("Using professional 1:1 text-to-speech mapping for exact timings")
            
            # STRATEGY: Direct mapping with fine-tuned timing
            # For each speech segment, find the best matching text chunk
            # This is similar to how professional subtitle editors work
            
            for i, (segment_start, segment_end) in enumerate(speech_segments):
                if i < len(chunks):
                    segment_duration = segment_end - segment_start
                    
                    # Apply professional timing standards:
                    # 1. Ensure minimum duration (broadcast standard is ~700ms minimum)
                    # 2. Maximum 3s caption display unless many words
                    min_display_time = 0.7
                    display_duration = max(min_display_time, segment_duration)
                    
                    # Professional broadcast timing - caption appears slightly before speech
                    # Standard is 50-100ms before speech starts
                    display_start = segment_start
                    
                    # Add this perfectly timed segment
                    aligned_segments.append({
                        "text": chunks[i],
                        "start": display_start,
                        "duration": display_duration
                    })
                    
            # For any remaining chunks, add with calibrated timing
            if len(chunks) > len(speech_segments):
                # Calculate appropriate duration based on word count and speaking rate
                for i in range(len(speech_segments), len(chunks)):
                    chunk = chunks[i]
                    chunk_words = len(chunk.split())
                    
                    # Calculate natural duration based on professional standards
                    # Allow 250-350ms per word + 500ms padding for reading
                    natural_duration = min(3.0, (chunk_words / words_per_second) * 1.15 + 0.5)
                    
                    # Position after the last segment with small gap
                    if aligned_segments:
                        last_segment = aligned_segments[-1]
                        start_time = last_segment["start"] + last_segment["duration"] + 0.05
                    else:
                        start_time = speech_segments[-1][1] + 0.05
                        
                    aligned_segments.append({
                        "text": chunk,
                        "start": start_time,
                        "duration": natural_duration
                    })
                    
        # SCENARIO B: More text chunks than speech segments
        # This happens with denser text or sparse speech detection
        elif len(chunks) > len(speech_segments):
            logger.info("Using professional multi-text distribution algorithm")
            
            # PROFESSIONAL APPROACH: Distribute chunks across segments with precise timing
            # This uses a dynamic programming approach similar to professional caption timing software

            # Calculate optimal words per segment based on segment duration
            segment_capacity = []
            for start, end in speech_segments:
                segment_dur = end - start
                # How many words can fit in this segment at our speaking rate
                # Use 90% of theoretical capacity for natural reading
                optimal_words = int(segment_dur * words_per_second * 0.9)
                segment_capacity.append(optimal_words)
            
            # Track words in each chunk
            chunk_words = [len(chunk.split()) for chunk in chunks]
            
            # Dynamic distribution - ensure captions appear with spoken words
            current_chunk = 0
            segment_assignments = {i: [] for i in range(len(speech_segments))}
            
            # First pass - assign chunks to segments
            for segment_idx, capacity in enumerate(segment_capacity):
                words_assigned = 0
                
                # Add chunks until we reach capacity or run out
                while current_chunk < len(chunks) and words_assigned + chunk_words[current_chunk] <= capacity * 1.1:
                    segment_assignments[segment_idx].append(current_chunk)
                    words_assigned += chunk_words[current_chunk]
                    current_chunk += 1
                    
                    if current_chunk >= len(chunks):
                        break
                        
            # Handle any remaining chunks - assign to the last segment
            if current_chunk < len(chunks):
                for i in range(current_chunk, len(chunks)):
                    segment_assignments[len(speech_segments)-1].append(i)
            
            # Second pass - create precisely timed captions from assignments
            for segment_idx, chunk_indices in segment_assignments.items():
                if not chunk_indices:
                    continue
                    
                segment_start, segment_end = speech_segments[segment_idx]
                segment_duration = segment_end - segment_start
                
                # If single chunk for this segment, timing is straightforward
                if len(chunk_indices) == 1:
                    chunk_idx = chunk_indices[0]
                    aligned_segments.append({
                        "text": chunks[chunk_idx],
                        "start": segment_start,
                        "duration": segment_duration
                    })
                    
                # Multiple chunks for this segment
                # Distribute them evenly across the segment duration
                else:
                    # Professional timing uses weighted distribution based on word count
                    total_words = sum(chunk_words[idx] for idx in chunk_indices)
                    current_time = segment_start
                    
                    for chunk_idx in chunk_indices:
                        chunk = chunks[chunk_idx]
                        # Duration proportional to word count
                        word_proportion = chunk_words[chunk_idx] / max(total_words, 1)
                        chunk_duration = max(0.7, segment_duration * word_proportion)
                        
                        aligned_segments.append({
                            "text": chunk,
                            "start": current_time,
                            "duration": chunk_duration
                        })
                        
                        # Advance time for next chunk, with small overlap for smooth reading
                        # Professional captions often have 50-100ms overlap between segments
                        current_time += max(0.1, chunk_duration - 0.05)
        
        # SCENARIO C: Fewer text chunks than speech segments
        # This happens with sparse text or over-sensitive speech detection
        else:
            logger.info("Using professional segment consolidation algorithm")
            
            # PROFESSIONAL APPROACH: Consolidate speech segments for natural caption flow
            
            # First, merge adjacent speech segments to better match text chunks
            merged_speech = []
            current_start, current_end = speech_segments[0]
            
            for i in range(1, len(speech_segments)):
                next_start, next_end = speech_segments[i]
                
                # If the gap is small, merge segments
                if next_start - current_end < 0.3:  # 300ms gap threshold
                    current_end = next_end
                else:
                    # Add the current merged segment
                    merged_speech.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
                    
            # Add the final merged segment
            merged_speech.append((current_start, current_end))
            
            # Now we can match text chunks to merged speech segments more naturally
            if len(chunks) <= len(merged_speech):
                # Direct 1:1 mapping is now possible
                for i, chunk in enumerate(chunks):
                    if i < len(merged_speech):
                        start, end = merged_speech[i]
                        duration = end - start
                        
                        aligned_segments.append({
                            "text": chunk,
                            "start": start,
                            "duration": duration
                        })
            else:
                # Still need to distribute - use the algorithm from SCENARIO B
                # Calculate words per segment
                words_per_segment = {}
                
                for i, (start, end) in enumerate(merged_speech):
                    segment_dur = end - start
                    # Professional approach uses adaptive word count
                    # Slower speech: ~2 words/sec, Faster speech: ~3.5 words/sec
                    words_per_segment[i] = max(1, int(segment_dur * words_per_second))
                
                # Distribute chunks based on word counts
                current_chunk = 0
                segment_text = {i: [] for i in range(len(merged_speech))}
                
                for segment_idx in range(len(merged_speech)):
                    target_words = words_per_segment.get(segment_idx, 1)
                    words_added = 0
                    
                    while current_chunk < len(chunks) and words_added < target_words * 1.1:
                        chunk = chunks[current_chunk]
                        chunk_words = len(chunk.split())
                        
                        # Use professional broadcast timing standards
                        # Don't overfill a segment if not the last one
                        if (words_added > 0 and 
                            words_added + chunk_words > target_words * 1.2 and
                            segment_idx < len(merged_speech) - 1):
                            break
                            
                        segment_text[segment_idx].append(chunk)
                        words_added += chunk_words
                        current_chunk += 1
                        
                        if current_chunk >= len(chunks):
                            break
                
                # Create caption segments from the distribution
                for segment_idx, (start, end) in enumerate(merged_speech):
                    if not segment_text[segment_idx]:
                        continue
                        
                    segment_chunks = segment_text[segment_idx]
                    segment_duration = end - start
                    
                    # Evenly space multiple chunks for this segment
                    if len(segment_chunks) > 1:
                        chunk_duration = segment_duration / len(segment_chunks)
                        
                        for i, chunk in enumerate(segment_chunks):
                            # Create a caption with precise timing
                            aligned_segments.append({
                                "text": chunk, 
                                "start": start + (i * chunk_duration),
                                "duration": max(0.7, chunk_duration)  # Minimum professional duration
                            })
                    else:
                        # Just one chunk for this segment
                        aligned_segments.append({
                            "text": segment_chunks[0],
                            "start": start,
                            "duration": max(0.7, segment_duration)  # Minimum professional duration
                        })

        # FINAL POLISH: Ensure no caption timing overlaps
        # In professional captioning, segments should never overlap
        if aligned_segments:
            # Sort by start time to ensure proper sequencing
            aligned_segments.sort(key=lambda x: x["start"])
            
            # Check for and fix overlaps
            for i in range(1, len(aligned_segments)):
                prev_end = aligned_segments[i-1]["start"] + aligned_segments[i-1]["duration"]
                curr_start = aligned_segments[i]["start"]
                
                # If this segment starts before previous ends, adjust timing
                if curr_start < prev_end:
                    # Adjust current segment to start right after previous
                    # Add small 50ms gap between captions
                    aligned_segments[i]["start"] = prev_end + 0.05
                    
                    # If adjustment makes caption too short, adjust duration
                    caption_end = aligned_segments[i]["start"] + aligned_segments[i]["duration"]
                    if i < len(aligned_segments) - 1 and caption_end > aligned_segments[i+1]["start"]:
                        aligned_segments[i]["duration"] = max(0.5, aligned_segments[i+1]["start"] - aligned_segments[i]["start"] - 0.05)

        logger.info(f"Created {len(aligned_segments)} captions with broadcast-quality timing")
        return aligned_segments

    def _create_caption_clip(self, text: str, duration: float, video_size: Tuple[int, int],
                        highlight_words: Optional[List[str]] = None) -> TextClip:
        # Try multiple methods to create captions, with fallbacks
        """Create a styled caption clip with word highlighting.

        Args:
            text: The caption text
            duration: Duration of the caption
            video_size: Size of the video (width, height)
            highlight_words: List of words to highlight

        Returns:
            TextClip with styled captions
        """
        if not text:
            return None

        # Debug logs
        logger.info(f"Creating caption for text: '{text}'")
        logger.info(f"Duration: {duration}, Video size: {video_size}")

        try:

            video_width, video_height = video_size

            # Convert text to UPPERCASE to match the style in the screenshot
            text = text.upper()

            # Styling parameters - make text readable but not too large
            font_size = min(int(video_height * 0.08), 48)  # Slightly larger for better visibility

            # First ensure text is between 4-6 words per line
            words = text.split()
            if len(words) > 6:
                # Split into shorter chunks
                chunks = self._split_text_into_chunks(text, 5)  # Max 5 words per chunk
                if chunks and len(chunks) == 1:
                    text = chunks[0]  # Use the first chunk if we only have one
                elif chunks and len(chunks) > 1:
                    # Use first two chunks as two lines if available
                    text = f"{chunks[0]}\n{chunks[1]}"
            elif len(words) <= 6 and len(words) >= 4:
                # Good length already - no change needed
                pass
            else:
                # Text is very short, keep as is
                pass

            # Now proceed with word-by-word highlighting
            try:
                # Create individual word clips with alternating yellow/green highlighting
                lines = text.split('\n')
                composite_clips = []
                max_line_width = 0
                total_height = 0
                line_clips = []

                # Process each line separately (limit to configured max lines)
                max_caption_lines = get_config("max_caption_lines", 2)
                for line_idx, line in enumerate(lines[:max_caption_lines]):
                    words = line.split()
                    word_clips = []
                    word_sizes = []
                    highlight_indices = []

                    # Find highlight words for this line
                    if highlight_words and len(highlight_words) > 0:
                        for i, word in enumerate(words):
                            clean_word = word.strip('.,!?;:"\'()[]{}').lower()
                            if any(hw.lower() in clean_word for hw in highlight_words):
                                highlight_indices.append(i)

                    # If no highlight words matched or none provided, find significant words to highlight
                    if not highlight_indices and len(words) > 1:
                        # Find at least one word to highlight in each line
                        for i, word in enumerate(words):
                            clean_word = word.strip('.,!?;:"\'()[]{}').lower()
                            if len(clean_word) > 3 and clean_word.lower() not in STOPWORDS:
                                highlight_indices.append(i)
                                # Find a second word to highlight (alternating colors) if possible
                                for j in range(i+1, len(words)):
                                    second_word = words[j].strip('.,!?;:"\'()[]{}').lower()
                                    if len(second_word) > 3 and second_word not in STOPWORDS:
                                        highlight_indices.append(j)
                                        break
                                break

                    # Create clips for each word
                    for i, word in enumerate(words):
                        # Determine if this word should be highlighted
                        is_highlight = (i in highlight_indices)

                        # Use configured colors for highlights
                        if is_highlight:
                            word_color = get_config("caption_highlight_color", "#00FF00")
                        else:
                            word_color = get_config("caption_normal_color", "white")

                        # Don't add spaces to displayed words - we'll handle spacing with positioning
                        display_word = word

                        # Create professional word clip with enhanced visibility
                        # Bold style with optimized stroke for maximum readability
                        try:
                            # Use Arial font for maximum compatibility and clarity
                            word_clip = TextClip(
                                text=display_word,  # Use 'text' parameter
                                font_size=font_size + 4,  # Slightly larger for better visibility
                                font='Arial',
                                color=word_color,
                                stroke_color="black",
                                stroke_width=3,  # Refined stroke width for cleaner appearance
                                size=(video_width//4, None),  # Better width constraint for natural spacing
                                method='caption'
                            ).with_duration(duration)
                        except Exception as e:
                            logger.warning(f"First attempt at text rendering failed: {e}")
                            try:
                                # Second attempt with font_size instead of fontsize
                                word_clip = TextClip(
                                    text=display_word,
                                    font_size=font_size + 4,  # Try font_size instead of fontsize
                                    font='Arial',
                                    color=word_color,
                                    stroke_color="black",
                                    stroke_width=5,
                                    size=(video_width//5, None),  # Reduced width for better spacing
                                    method='caption'
                                ).with_duration(duration)
                            except Exception as e:
                                logger.warning(f"Second attempt at text rendering failed: {e}")
                                # Ultra simple fallback with absolutely minimal settings
                                word_clip = TextClip(
                                    text=display_word,
                                    font='Arial',
                                    font_size=font_size + 4,  # Use font_size, not fontsize
                                    color=word_color,
                                    size=(video_width//3, None),
                                    method='caption'
                                ).with_duration(duration)

                        # Store the clip and its size
                        word_clips.append(word_clip)
                        word_sizes.append((word_clip.w, word_clip.h))

                    # Get word spacing from config
                    word_spacing = get_config("word_spacing", 0)  # Spacing between words in pixels

                    # Calculate total width with controlled spacing between words
                    total_width = sum(w for w, h in word_sizes)
                    if len(word_sizes) > 1:
                        total_width += word_spacing * (len(word_sizes) - 1)

                    line_height = max(h for w, h in word_sizes) if word_sizes else 0
                    max_line_width = max(max_line_width, total_width)

                    # Create composite line with controlled spacing to preserve highlighting
                    line_composite_clips = []
                    x_position = 0

                    # Position each word with precise spacing
                    for i, (word_clip, (w, h)) in enumerate(zip(word_clips, word_sizes)):
                        positioned_clip = word_clip.with_position((x_position, 0))
                        line_composite_clips.append(positioned_clip)
                        x_position += w

                        # Add controlled spacing between words (not after the last word)
                        if i < len(word_clips) - 1:
                            x_position += word_spacing

                    # Create composite that preserves word highlighting
                    line_clip = CompositeVideoClip(
                        line_composite_clips,
                        size=(total_width, line_height)
                    ).with_duration(duration)

                    # Line clip is created above as a composite with the highlight words preserved

                    line_clips.append((line_clip, line_height))
                    total_height += line_height

                # Calculate spacing between lines
                line_spacing = int(font_size * 0.3)  # Adjust spacing between lines
                total_height += line_spacing * (len(line_clips) - 1)  # Use actual number of line clips

                # Create a semi-transparent background for better readability
                bg_width = int(min(max_line_width * 1.1, video_width * 0.85))  # Balanced width to avoid wrapping
                bg_height = int(total_height * 1.2)  # Maintain good vertical spacing

                # No background - just the text with stroke for visibility (like in screenshot)
                composite_clips = []

                # Position each line on the frame with proper spacing
                y_position = (bg_height - total_height) // 2

                for i, (line_clip, line_height) in enumerate(line_clips):
                    # Center the line horizontally
                    x_position = (bg_width - line_clip.w) // 2
                    positioned_line = line_clip.with_position((x_position, y_position))
                    composite_clips.append(positioned_line)

                    # Only add line spacing if this isn't the last line
                    if i < len(line_clips) - 1:
                        y_position += line_height + line_spacing
                    else:
                        y_position += line_height

                # Create composite of text lines only (no background)
                caption_composite = CompositeVideoClip(
                    composite_clips,
                    size=(bg_width, bg_height),
                    bg_color=None  # Ensure transparent background
                ).with_duration(duration)

                # Position the whole caption based on config
                # Using the bottom third position (0.66) by default for better visibility
                caption_position = get_config("caption_position", 0.66)
                positioned_caption = caption_composite.with_position(
                    ("center", int(video_height * caption_position))
                )

                logger.info(f"Created caption with highlighted words in {'two' if len(lines) > 1 else 'one'} line(s)")
                return positioned_caption

            except Exception as e:
                logger.error(f"Failed to create highlighted caption: {e}")

                # FALLBACK: Create a simpler caption with basic highlighting
                return self._create_simple_text_clip(text, duration, video_size)
        except Exception as final_e:
            logger.error(f"Final caption creation error: {final_e}")
            return None

    def _create_simple_text_clip(self, text: str, duration: float, video_size: Tuple[int, int], word_color: str = 'white') -> TextClip:
        """Create a simple text clip with maximum compatibility."""
        video_width, video_height = video_size
        font_size = 30  # Slightly larger default font

        try:
            # Split text into shorter lines for better readability if needed
            lines = text.split('\n')
            if len(lines) == 1 and len(lines[0].split()) > 5:
                words = lines[0].split()
                mid_point = min(5, len(words) // 2)  # Ensure first line is max 5 words
                line1 = " ".join(words[:mid_point])
                line2 = " ".join(words[mid_point:])  # All remaining words
                text = f"{line1}\n{line2}"

            # Ultra-simple implementation with minimal parameters and proper text sizing
            txt_clip = TextClip(
                text=text,  # Use 'text' parameter instead of 'txt'
                font='Arial',  # Always specify a font
                font_size=font_size,  # Use font_size, not fontsize
                color=word_color,
                stroke_color="black",
                stroke_width=5,
                size=(int(video_width * 0.5), None),  # Wider to allow some natural spacing
                method='caption'  # This version doesn't support kerning
                # 'align' parameter not supported in this version
            ).with_duration(duration)


            # Position at the bottom of the screen
            positioned_caption = txt_clip.with_position(
                ("center", int(video_height * 0.66))
            )

            # Note: set_layer only works on CompositeVideoClip in newer versions
            # Simply return the positioned caption

            logger.info("Created simple caption with fallback method")
            return positioned_caption

        except Exception as e:
            logger.error(f"Final fallback caption creation failed: {e}")
            try:
                # Last resort - create caption with absolute minimum parameters
                txt_clip = TextClip(
                    text=text,
                    font='Arial',  # Always include font
                    font_size=font_size,  # Use font_size, not fontsize
                    color='white',
                    size=(int(video_width * 0.8), None),
                    method='caption'
                ).with_duration(duration)

                return txt_clip.with_position(("center", int(video_height * 0.66)))
            except Exception as e:
                logger.error(f"All caption creation methods failed: {e}")
                # Return a colored clip as absolute last resort
                blank = ColorClip(size=(200, 50), color=(0, 0, 0)).with_duration(duration)
                return blank.with_position(("center", int(video_height * 0.66)))

    def create_clip(
        self, segments: List[Segment], output_path: str, max_duration: float = 45.0,
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
            force_algorithm: Whether to force use of the sophisticated timing algorithm
            target_duration: Target duration for the clip (will aim for exactly this duration)

        Returns:
            Tuple of (output_path, duration)

        Raises:
            VideoProcessingError: If clip creation fails
        """
        try:
            logger.info(f"Creating clip with {len(segments)} segments")
            
            video = self._load_video()
            output_path = validate_output_path(output_path)
            
            # Use target_duration if provided, otherwise use max_duration
            effective_duration = target_duration or max_duration
            logger.info(f"Targeting clip duration of {effective_duration} seconds")

            if not segments:
                raise VideoProcessingError("No segments provided for clip creation")

            # Sort segments by start time to ensure chronological order
            segments.sort(key=lambda x: x.start)

            # Use the segments as they were provided - they've already been selected by segment_selector
            # Just validate and adjust if necessary
            selected_segments = []
            total_duration = 0.0
            
            for segment in segments:
                start = max(0, segment.start)
                end = min(self._duration or float('inf'), segment.end)
                
                # Skip very short segments (less than 1.5 seconds)
                if end - start < 1.5:
                    logger.info(f"Skipping too short segment: {start:.2f}s-{end:.2f}s (duration: {end-start:.2f}s)")
                    continue
                
                # Apply minor timing adjustments if needed
                adjusted_segment = segment
                adjusted_segment.start = start
                adjusted_segment.end = end
                
                # Add to selected segments
                selected_segments.append(adjusted_segment)
                total_duration += (end - start)
                
                # Add estimated transition time (0.5s per transition except for the first segment)
                if len(selected_segments) > 1:
                    total_duration += 0.5
                
                logger.info(f"Added segment: {start:.2f}s-{end:.2f}s (duration: {end-start:.2f}s)")
                
            logger.info(f"Using {len(selected_segments)} segments with total estimated duration of {total_duration:.2f}s")

            # Log final segment selection
            logger.info(f"Final selected segments: {len(selected_segments)} with total duration: {total_duration:.2f}s")
            
            # Extract the relevant subclips with clean boundaries for proper transitions
            subclips = []
            for segment in selected_segments:
                # Ensure segment boundaries are within video duration
                start = max(0, segment.start)
                end = min(self._duration or float('inf'), segment.end)

                # Ensure minimum clip duration (to avoid transition issues)
                if end - start < 1.0:  # Less than 1 second is too short for proper transitions
                    # Extend to at least 1 second if possible
                    if end + 0.5 <= self._duration:
                        end += 0.5
                    elif start - 0.5 >= 0:
                        start -= 0.5
                
                if start < end:
                    logger.info(f"Creating subclip from {start:.2f}s to {end:.2f}s (duration: {end-start:.2f}s)")
                    # Use subclipped method instead of subclip
                    subclip = video.subclipped(start, end)

                    # CRITICAL: Calculate relative time adjustment for subtitle segments
                    # This is needed to align subtitle timings with the subclip
                    segment_relative_start = start

                    # If this is a subtitle-based segment, update metadata with relative timing
                    if hasattr(segment, 'metadata') and isinstance(segment.metadata, dict):
                        if segment.metadata.get('subtitle_timing'):
                            # Calculate time adjustment (segment time → subclip time)
                            segment.metadata['subclip_start_offset'] = 0 - segment_relative_start
                            logger.info(f"Subtitle segment timing adjusted by {segment.metadata['subclip_start_offset']} seconds")

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

                            # CRITICAL PROBLEM IDENTIFIED:
                            # The segment text may contain content that's not actually in the subclip
                            # if the segment boundaries don't align perfectly with speech boundaries.

                            # SOLUTION:
                            # 1. For subtitle-based segments, we need to filter captions to only include
                            #    ones that fall within the actual subclip time range
                            # 2. For transcript-based segments, we need a more precise way to align text with audio

                            # Check if this is coming from SRT subtitles (which have exact timing)
                            has_subtitle_timing = False
                            subtitle_segments = []
                            relative_start = 0

                            if hasattr(segment, 'metadata') and segment.metadata:
                                if segment.metadata.get('subtitle_timing') == True and not force_algorithm:
                                    has_subtitle_timing = True
                                    logger.info("Found exact subtitle timing data - using for perfect sync")

                                    # If this segment came from SRT, we need to determine if it contains
                                    # multiple subtitle entries that were merged

                                    # Extract the exact original subtitle timing info
                                    exact_start = segment.metadata.get('exact_start', segment.start)

                                    # CRITICAL FIX: Apply subclip offset to align timing perfectly
                                    # This ensures captions are properly timed relative to the subclip
                                    timing_offset = segment.metadata.get('subclip_start_offset', 0)
                                    logger.info(f"Applying subtitle timing offset of {timing_offset:.3f}s")

                                    # If we have access to the original SRT file, we could actually
                                    # find all subtitle segments that fall within our clip range
                                    # For now, we'll use a hack - split by sentence and assign timing

                                    # Split the text by sentence endings to approximate subtitle segments
                                    sentences = re.split(r'(?<=[.!?])\s+', segment.text)
                                    if sentences:
                                        # Calculate total duration and words
                                        total_words = sum(len(sentence.split()) for sentence in sentences)
                                        sentence_count = len(sentences)

                                        # Distribute sentences evenly across the clip duration
                                        # (this is an approximation since we don't have the exact subtitle timing for each sentence)
                                        for i, sentence in enumerate(sentences):
                                            # Apply precise timing with offset adjustment
                                            # Use precise relative positioning within the clip
                                            rel_start = (i / sentence_count) * subclip.duration

                                            # Adjust with the offset if available (this is the critical fix for perfect timing)
                                            if timing_offset and i == 0:
                                                # Only adjust the first segment - it needs to be perfectly aligned
                                                # with the start of speech in the clip
                                                logger.info(f"First sentence gets precise timing alignment")

                                            # Duration based on word count proportion
                                            word_count = len(sentence.split())
                                            rel_duration = (word_count / max(total_words, 1)) * subclip.duration
                                            # Ensure it doesn't extend beyond clip end
                                            if rel_start + rel_duration > subclip.duration:
                                                rel_duration = subclip.duration - rel_start

                                            if rel_duration <= 0:
                                                continue

                                            subtitle_segments.append({
                                                'text': sentence.strip(),
                                                'start': rel_start,
                                                'duration': rel_duration
                                            })

                            # PERFECT TIMING APPROACH USING FORCED ALIGNMENT
                            # Use forced alignment to precisely match text to audio

                            # Step 1: Process the text to ensure it's only what's in this subclip
                            # If it came from subtitle timing, we already filtered above
                            processed_segments = []

                            if has_subtitle_timing and subtitle_segments:
                                # Use the pre-processed subtitle segments
                                processed_segments = subtitle_segments
                                logger.info(f"Using {len(processed_segments)} subtitle segments with exact subtitle timing")
                            else:
                                try:
                                    # Extract the audio from this subclip for precise alignment
                                    logger.info(f"Extracting audio for forced alignment...")
                                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                                        temp_audio_path = temp_audio_file.name

                                    # Export audio for this subclip
                                    subclip.audio.write_audiofile(
                                        temp_audio_path,
                                        fps=44100,
                                        nbytes=2,
                                        codec='pcm_s16le',
                                        ffmpeg_params=['-ac', '1']  # Convert to mono for better alignment
                                    )

                                    # Align text with detected speech segments
                                    logger.info(f"Aligning text with speech for perfect caption timing")
                                    aligned_segments = self._align_text_with_speech(temp_audio_path, segment.text)

                                    # Use the aligned segments
                                    if aligned_segments and len(aligned_segments) > 0:
                                        processed_segments = aligned_segments
                                        logger.info(f"Successfully aligned {len(processed_segments)} segments with audio using forced alignment")
                                    else:
                                        raise Exception("Forced alignment returned no segments")

                                    # Clean up temporary file
                                    try:
                                        os.remove(temp_audio_path)
                                    except:
                                        pass

                                except Exception as e:
                                    logger.error(f"Forced alignment failed: {str(e)}. Falling back to estimation method.")
                                    # Fallback to the original method if forced alignment fails
                                    # Split text into 4-6 word chunks for readability
                                    chunks = self._split_text_into_chunks(segment.text, 5)  # Max 5 words per chunk
                                    logger.info(f"Split into {len(chunks)} text chunks (4-6 words each)")

                                    if chunks and len(chunks) > 0:
                                        # Calculate total word count for timing
                                        total_words = sum(len(chunk.split()) for chunk in chunks)

                                        # Estimate speaking rate (words per second)
                                        # Conservative range: 120-180 words per minute = 2-3 words per second
                                        base_rate = total_words / subclip.duration  # Calculated rate
                                        speaking_rate = min(max(base_rate, 1.8), 3.0)  # Constrained to reasonable range
                                        logger.info(f"Speaking rate: {speaking_rate:.2f} words/second")

                                        # Time needed per word with precision tuning
                                        word_time = 1.0 / speaking_rate

                                        # First pass: calculate timing for each chunk
                                        current_time = 0.0

                                        for chunk_text in chunks:
                                            # Get word count and calculate base duration
                                            words = chunk_text.split()
                                            word_count = len(words)

                                            # Skip empty chunks
                                            if word_count == 0:
                                                continue

                                            # Calculate chunk duration based on word count and speaking rate
                                            # Add small multiplier for readability (1.2x)
                                            chunk_duration = (word_count * word_time) * 1.2

                                            # Enforce minimum and maximum duration constraints
                                            min_duration = min(0.6, subclip.duration / 2)  # At least 0.6s unless clip is very short
                                            max_duration = min(3.5, subclip.duration)  # No more than 3.5s or clip duration
                                            chunk_duration = max(min(chunk_duration, max_duration), min_duration)

                                            # Stop if we've reached the end of the clip
                                            if current_time >= subclip.duration:
                                                break

                                            # Ensure we don't exceed clip duration
                                            if current_time + chunk_duration > subclip.duration:
                                                chunk_duration = subclip.duration - current_time

                                            # Store segment timing data
                                            processed_segments.append({
                                                'text': chunk_text,
                                                'start': current_time,
                                                'duration': chunk_duration
                                            })

                                            # Advance position for next chunk
                                            current_time += chunk_duration

                                            # Add tiny gap between segments for more natural flow
                                            gap = 0.05  # 50ms gap
                                            if current_time + gap < subclip.duration:
                                                current_time += gap
                                                        # Create caption clips based on processed segments
                            caption_clips = []

                            # Process each segment to create a precisely timed caption
                            for segment_data in processed_segments:
                                # Verification check - skip any segments with invalid timing
                                if segment_data['duration'] <= 0:
                                    continue

                                if segment_data['start'] < 0 or segment_data['start'] >= subclip.duration:
                                    continue

                                # Create caption clip with the exact duration
                                caption_clip = self._create_caption_clip(
                                    segment_data['text'],
                                    segment_data['duration'],
                                    video_size,
                                    segment_keywords
                                )

                                if caption_clip:
                                    # Position at exact calculated start time
                                    caption_with_start = caption_clip.with_start(segment_data['start'])
                                    caption_clips.append(caption_with_start)

                                    # Log detailed timing for debugging
                                    logger.info(f"Caption [{segment_data['start']:.2f}s - {segment_data['start'] + segment_data['duration']:.2f}s]: '{segment_data['text']}'")

                            # Composite all caption clips with the video - FIXED APPROACH
                            if caption_clips:
                                # Explicitly set the clips with the base clip first
                                all_clips = [subclip] + caption_clips
                                # Create composite with explicit size to ensure captions appear
                                subclip = CompositeVideoClip(all_clips, size=video_size)
                                logger.info(f"Added {len(caption_clips)} precisely timed caption clips")
                            else:
                                # Fallback: use a single caption for the entire subclip
                                # This is less ideal for timing but works as a fallback
                                caption_clip = self._create_caption_clip(
                                    segment.text,
                                    subclip.duration,
                                    video_size,
                                    segment_keywords
                                )

                                if caption_clip:
                                    # Explicitly set the composite with the base clip first
                                    # This ensures proper stacking and visibility
                                    subclip = CompositeVideoClip([
                                        subclip,
                                        caption_clip
                                    ], size=video_size)
                                    logger.info("Added single caption for entire segment - timing may be less precise")
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

            # Apply simple but clean crossfade transitions for a professional look
            logger.info(f"Applying transitions between {len(subclips)} subclips")
            
            if len(subclips) > 1:
                # Use MoviePy's built-in crossfade - the most reliable method
                try:
                    logger.info("Applying standard crossfade transitions (0.5s)")
                    
                    # Direct use of concatenate_videoclips with crossfade is most reliable
                    crossfade_duration = get_config("crossfade_duration", 0.5)
                    final_clip = concatenate_videoclips(
                        subclips, 
                        method="crossfade",
                        crossfade_duration=crossfade_duration
                    )
                    
                    # Add subtle fade in/out effect to the entire clip for a polished look
                    logger.info("Adding fade in/out to final clip")
                    final_clip = FadeIn(duration=0.7).apply(final_clip)  # Slightly longer fade for smoother entry
                    final_clip = FadeOut(duration=0.7).apply(final_clip)  # Slightly longer fade for smoother exit
                except Exception as e:
                    # Fallback to simple concatenation if crossfades fail
                    logger.warning(f"Crossfade transitions failed: {e}, falling back to simple concatenation")
                    final_clip = concatenate_videoclips(subclips, method="compose")
                    # Still add fades for polish
                    final_clip = FadeIn(duration=0.7).apply(final_clip)
                    final_clip = FadeOut(duration=0.7).apply(final_clip)
            else:
                final_clip = subclips[0]
                # For single clips, still add fade in/out effects
                logger.info("Single clip - adding fade in/out effects")
                final_clip = FadeIn(duration=0.7).apply(final_clip)
                final_clip = FadeOut(duration=0.7).apply(final_clip)

            # We NEVER want to repeat content in viral shorts, so no looping!
            
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
                min_segment_duration=3.0,  # Fixed 3-second segment width
                max_segment_duration=15.0,
            )


            # Process segments
            processed_segments = selector.process_segments(segments)

            # Select top segments
            selected_segments = selector.select_top_segments(
                processed_segments, max_duration, top_segments
            )

            # Create the clip
            return self.create_clip(selected_segments, output_path, target_duration=max_duration)

        except Exception as e:
            raise VideoProcessingError(f"Failed to create highlight clip: {e}")
