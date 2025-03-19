"""
Video model representing a video file with metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from videoclipper.models.segment import Segment


@dataclass
class Video:
    """A video file with metadata and segments."""
    
    path: str
    
    # Video properties
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    
    # Metadata
    title: Optional[str] = None
    description: Optional[str] = None
    
    # Analysis results
    segments: List[Segment] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def filename(self) -> str:
        """Get the filename without the directory."""
        return Path(self.path).name