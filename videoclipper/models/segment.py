"""
Segment model representing a video segment with metadata.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class SegmentType(Enum):
    """Types of video segments for analysis."""
    SCENE_CHANGE = auto()
    ENERGY_PEAK = auto()
    CUSTOM = auto()
    SPEECH = auto()


@dataclass(frozen=False, eq=True)
class Segment:
    """A segment of video with metadata for analysis and selection."""
    
    start: float  # Changed from start_time for compatibility with analyzer code
    end: float    # Changed from end_time for compatibility with analyzer code
    
    # Overall score of the segment (higher is more interesting)
    score: float = 0.5  # Default score
    
    # Field for segment type 
    segment_type: Optional[SegmentType] = None  # Will be set to SegmentType enum value
    
    # Optional fields for metadata
    text: Optional[str] = None  # Description or transcript of the segment
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    # Caption timing information (list of start/end time tuples for caption chunks)
    caption_timing: Optional[List[tuple]] = None
    
    def __hash__(self):
        """Make Segment hashable based on start/end times which are its unique identifiers."""
        return hash((self.start, self.end))
    
    def __eq__(self, other):
        """Two segments are equal if they have the same start and end times."""
        if not isinstance(other, Segment):
            return False
        return (self.start, self.end) == (other.start, other.end)
    
    @property
    def duration(self) -> float:
        """Get the duration of this segment in seconds."""
        return self.end - self.start
        
    def merge(self, other: 'Segment') -> 'Segment':
        """Merge this segment with another, taking the union of their time spans."""
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        # Use the higher score between the two segments
        score = max(self.score, other.score)
        
        # Merge text content intelligently to avoid duplicated content
        merged_text = None
        if self.text and other.text:
            # If one text is contained within the other, use the longer one
            if self.text in other.text:
                merged_text = other.text
            elif other.text in self.text:
                merged_text = self.text
            else:
                # If texts are very different, concatenate with a marker
                # But only if the result won't be too long
                if len(self.text) + len(other.text) < 500:  # Reasonable limit
                    merged_text = f"{self.text} ... {other.text}"
                else:
                    # Use the higher-scored segment's text, or the longer one if scores are equal
                    merged_text = self.text if self.score >= other.score else other.text
        else:
            # Use whatever text is available
            merged_text = self.text or other.text
        
        # For caption_timing, use self's caption_timing if available, otherwise use other's
        caption_timing = self.caption_timing if self.caption_timing else other.caption_timing
        
        # Merge tags without duplicates
        merged_tags = list(set(self.tags + other.tags))
        
        return Segment(
            start=start,
            end=end,
            score=score,
            segment_type=self.segment_type,
            text=merged_text,
            tags=merged_tags,
            metadata={**other.metadata, **self.metadata},  # Merge metadata, with self taking precedence
            caption_timing=caption_timing
        )