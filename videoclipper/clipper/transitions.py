"""Professional video transitions for VideoClipper.

This module provides clean transitions that make clips flow together seamlessly.
"""

import logging
import numpy as np
from typing import Optional, Callable, Tuple, List

# Import moviepy classes consistent with the project style
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.VideoClip import VideoClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

logger = logging.getLogger(__name__)

def apply_crossfade_transition(clip1: VideoClip, clip2: VideoClip, duration: float = 0.5) -> VideoClip:
    """Apply a smooth crossfade transition between two clips.
    
    Args:
        clip1: First video clip (ending clip)
        clip2: Second video clip (starting clip)
        duration: Transition duration in seconds
        
    Returns:
        A composite video clip with the transition applied
    """
    # Ensure the transition duration is not longer than either clip
    duration = min(duration, clip1.duration, clip2.duration)
    
    # Create the transition
    # Instead of using subclip which is not available for CompositeVideoClip,
    # we'll use the original clips directly
    
    # Create a crossfade effect
    clip1_fadeout = clip1.crossfadeout(duration)
    clip2_fadein = clip2.crossfadein(duration)
    
    # Overlap the clips
    result = CompositeVideoClip([
        clip1_fadeout.with_start(0),
        clip2_fadein.with_start(clip1.duration - duration)
    ])
    
    return result

def apply_fade_transition(clip: VideoClip, fade_in: bool = True, fade_out: bool = True, 
                         fade_duration: float = 0.5) -> VideoClip:
    """Apply fade in and/or fade out to a clip.
    
    Args:
        clip: The video clip to apply fades to
        fade_in: Whether to apply fade in
        fade_out: Whether to apply fade out
        fade_duration: Fade duration in seconds
        
    Returns:
        Video clip with fades applied
    """
    # Ensure the fade duration is not longer than the clip
    fade_duration = min(fade_duration, clip.duration / 2)
    
    # Apply fades
    result = clip
    if fade_in:
        result = result.fadein(fade_duration)
    if fade_out:
        result = result.fadeout(fade_duration)
    
    return result

def apply_slide_transition(clip1: VideoClip, clip2: VideoClip, direction: str = 'left', 
                          duration: float = 0.75) -> VideoClip:
    """Apply a slide transition between two clips.
    
    Args:
        clip1: First video clip (ending clip)
        clip2: Second video clip (starting clip)
        direction: Direction of the slide ('left', 'right', 'up', 'down')
        duration: Transition duration in seconds
        
    Returns:
        A composite video clip with the transition applied
    """
    # Ensure the transition duration is not longer than either clip
    duration = min(duration, clip1.duration, clip2.duration)
    
    w, h = clip1.size
    
    # Define the slide function based on direction
    def get_slide_position(t):
        progress = min(1, t / duration)
        if direction == 'left':
            clip1_pos = (-w * progress, 0)
            clip2_pos = (w * (1 - progress), 0)
        elif direction == 'right':
            clip1_pos = (w * progress, 0)
            clip2_pos = (-w * (1 - progress), 0)
        elif direction == 'up':
            clip1_pos = (0, -h * progress)
            clip2_pos = (0, h * (1 - progress))
        elif direction == 'down':
            clip1_pos = (0, h * progress)
            clip2_pos = (0, -h * (1 - progress))
        else:
            # Default to left
            clip1_pos = (-w * progress, 0)
            clip2_pos = (w * (1 - progress), 0)
        return clip1_pos, clip2_pos
    
    # Create the transition
    def make_frame(t):
        clip1_pos, clip2_pos = get_slide_position(t)
        
        # Create a composite with the clips at their positions
        comp = CompositeVideoClip([
            clip1.with_position(clip1_pos),
            clip2.with_position(clip2_pos)
        ], size=(w, h))
        
        return comp.get_frame(t)
    
    # Create the transition clip
    transition = VideoClip(make_frame, duration=duration)
    
    # Simplified approach - create a direct crossfade instead of using subclips
    result = CompositeVideoClip([
        clip1.with_start(0).with_duration(clip1.duration),
        transition.with_start(clip1.duration - duration),
        clip2.with_start(clip1.duration).with_duration(clip2.duration)
    ])
    
    return result

def get_random_professional_transition(seed: Optional[int] = None) -> Tuple[Callable, dict]:
    """Get a random professional transition function and its parameters.
    
    This provides variety in the transitions while maintaining professional quality.
    
    Args:
        seed: Optional random seed for reproducibility
        
    Returns:
        Tuple of (transition_function, parameters_dict)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define transition options with appropriate parameters
    transitions = [
        (apply_crossfade_transition, {'duration': 0.5}),
        (apply_crossfade_transition, {'duration': 0.75}),
        (apply_slide_transition, {'direction': 'left', 'duration': 0.75}),
        (apply_slide_transition, {'direction': 'right', 'duration': 0.75}),
    ]
    
    # Select a random transition
    transition_idx = np.random.randint(0, len(transitions))
    return transitions[transition_idx]

def combine_clips_with_transitions(clips: List[VideoClip]) -> VideoClip:
    """Combine multiple clips with simple, clean transitions between them.
    
    Args:
        clips: List of video clips to combine
        
    Returns:
        A single video clip with all clips combined using transitions
    """
    if not clips:
        logger.warning("No clips provided to combine.")
        return None
    
    if len(clips) == 1:
        return clips[0]
    
    # Use the official moviepy concatenation with crossfadein/crossfadeout
    # This is the most reliable approach for proper transitions
    logger.info("Using reliable standard crossfade transitions")
    
    try:
        # Simple, reliable approach: use concatenate_videoclips with crossfade transition
        # Use a very simple crossfade - this is the most robust approach
        from videoclipper.config import get_config
        
        # Get crossfade duration from config
        crossfade_duration = get_config("crossfade_duration", 0.5)
        
        return concatenate_videoclips(
            clips, 
            method="crossfade",  # Official crossfade method
            crossfade_duration=crossfade_duration
        )
    except Exception as e:
        logger.warning(f"Crossfade transition failed: {e}, falling back to simple concatenation")
        # Absolutely simple concatenation as fallback
        return concatenate_videoclips(clips, method="compose")
