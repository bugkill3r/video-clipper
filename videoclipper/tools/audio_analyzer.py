"""
Audio analysis utilities for improved caption timing and segment detection.
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)

try:
    from moviepy.editor import AudioFileClip
except ImportError:
    logger.warning("MoviePy not available for audio analysis")

class AudioAnalyzer:
    """Analyzes audio to detect pauses, energy levels, and speech patterns."""
    
    def __init__(self, audio_path: str = None, video_path: str = None):
        """Initialize the audio analyzer.
        
        Args:
            audio_path: Path to the audio file (optional)
            video_path: Path to the video file (optional, used if no audio_path)
        """
        self.audio_path = audio_path
        self.video_path = video_path
        self.audio_clip = None
        self.duration = 0
        self.sampling_rate = 0
        self._amplitude_samples = []
        
    def load_audio(self) -> bool:
        """Load the audio file or extract audio from video.
        
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            # Try audio path first
            if self.audio_path and os.path.exists(self.audio_path):
                self.audio_clip = AudioFileClip(self.audio_path)
            # Fall back to video path
            elif self.video_path and os.path.exists(self.video_path):
                self.audio_clip = AudioFileClip(self.video_path)
            else:
                logger.error("No valid audio or video path provided")
                return False
                
            # Store basic properties
            self.duration = self.audio_clip.duration
            self.sampling_rate = self.audio_clip.fps
            
            logger.info(f"Loaded audio with duration: {self.duration:.2f}s, rate: {self.sampling_rate}Hz")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return False
            
    def detect_speech_pauses(self, window_size: float = 0.1, threshold_factor: float = 0.3) -> List[float]:
        """Detect potential pause points in speech.
        
        Args:
            window_size: Size of analysis window in seconds
            threshold_factor: Amplitude threshold as factor of mean amplitude
            
        Returns:
            List of timestamps where pauses likely occur
        """
        if not self.audio_clip:
            if not self.load_audio():
                return []
        
        # Sample audio at regular intervals
        try:
            # Determine number of samples
            num_samples = int(self.duration / window_size)
            amplitude_samples = []
            
            # Sample audio amplitude
            for i in range(num_samples):
                t = i * window_size
                if t < self.duration:
                    try:
                        # Get audio frame and calculate average amplitude
                        frame = self.audio_clip.get_frame(t)
                        amplitude = np.abs(frame).mean()
                        amplitude_samples.append((t, amplitude))
                    except Exception as e:
                        logger.warning(f"Error sampling audio at {t:.2f}s: {e}")
                        continue
            
            # Store for later use
            self._amplitude_samples = amplitude_samples
            
            # Find pause points (where amplitude drops significantly)
            if amplitude_samples:
                # Calculate mean amplitude for reference
                mean_amplitude = np.mean([amp for _, amp in amplitude_samples])
                threshold = mean_amplitude * threshold_factor
                
                # Identify points below threshold as potential pauses
                pause_points = [t for t, amp in amplitude_samples if amp < threshold]
                
                # Group close pause points (often a single pause will have multiple low samples)
                grouped_pauses = []
                current_group = []
                
                # Sort pause points by time
                for t in sorted(pause_points):
                    if not current_group or t - current_group[-1] < window_size * 3:
                        # Group pause points close to each other
                        current_group.append(t)
                    else:
                        # End current group and start a new one
                        if current_group:
                            # Use the middle of the group as the canonical pause point
                            grouped_pauses.append(sum(current_group) / len(current_group))
                        current_group = [t]
                
                # Don't forget the last group
                if current_group:
                    grouped_pauses.append(sum(current_group) / len(current_group))
                
                logger.info(f"Detected {len(grouped_pauses)} pause points in audio")
                return grouped_pauses
                
            return []
            
        except Exception as e:
            logger.error(f"Error detecting pauses: {e}")
            return []
    
    def analyze_speech_rate(self, text: str) -> float:
        """Estimate speech rate based on text length and audio duration.
        
        Args:
            text: The text being spoken
            
        Returns:
            Estimated words per second
        """
        if not self.audio_clip:
            if not self.load_audio():
                # Default to typical speech rate if audio unavailable
                return 2.5
        
        try:
            # Count words in text
            words = text.split()
            word_count = len(words)
            
            # Calculate words per second
            if word_count > 0 and self.duration > 0:
                words_per_second = word_count / self.duration
                
                # Adjust to reasonable range (humans typically speak 2-4 words/sec)
                words_per_second = min(max(words_per_second, 2.0), 4.0)
                
                logger.info(f"Estimated speech rate: {words_per_second:.2f} words/second")
                return words_per_second
            else:
                return 2.5  # Default reasonable value
                
        except Exception as e:
            logger.error(f"Error analyzing speech rate: {e}")
            return 2.5
    
    def get_amplitude_profile(self) -> List[Tuple[float, float]]:
        """Get the amplitude profile of the audio.
        
        Returns:
            List of (time, amplitude) tuples
        """
        if not self._amplitude_samples:
            if not self.audio_clip:
                if not self.load_audio():
                    return []
            
            # Sample at 10Hz if not already done
            window_size = 0.1
            self.detect_speech_pauses(window_size)
        
        return self._amplitude_samples
        
    def close(self):
        """Clean up resources."""
        if self.audio_clip:
            try:
                self.audio_clip.close()
            except Exception:
                pass