"""
Segment model representing a video segment with metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Segment:
    """A segment of video with metadata for analysis and selection."""
    
    start_time: float
    end_time: float
    
    # Scores from different analyzers (higher is more interesting)
    audio_energy_score: float = 0.0
    scene_change_score: float = 0.0
    content_score: float = 0.0
    
    # Combined score (weighted average of individual scores)
    overall_score: float = 0.0
    
    # Metadata
    transcript: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get the duration of this segment in seconds."""
        return self.end_time - self.start_time