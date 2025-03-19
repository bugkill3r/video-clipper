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


@dataclass
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
        
        return Segment(
            start=start,
            end=end,
            score=score,
            segment_type=self.segment_type,
            text=self.text,
            metadata={**other.metadata, **self.metadata}  # Merge metadata, with self taking precedence
        )