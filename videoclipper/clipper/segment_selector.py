"""Segment selection and processing for video clipping."""

import logging
from typing import List, Optional

from videoclipper.models.segment import Segment
from videoclipper.config import get_config

# Set up logging
logger = logging.getLogger(__name__)


class SegmentSelector:
    """Selects and processes segments for inclusion in the final highlight video."""

    def __init__(
        self,
        video_duration: float,
        min_segment_duration: float = None,
        max_segment_duration: float = None,
    ) -> None:
        """Initialize the segment selector.

        Args:
            video_duration: Duration of the source video in seconds
            min_segment_duration: Minimum allowed segment duration in seconds
            max_segment_duration: Maximum allowed segment duration in seconds
        """
        self.video_duration = video_duration
        self.min_segment_duration = min_segment_duration or get_config("min_segment_duration", 3.0)
        self.max_segment_duration = max_segment_duration or get_config("max_segment_duration", 45.0)

    def process_segments(self, segments: List[Segment]) -> List[Segment]:
        """Process a list of segments to prepare for clipping.

        This includes:
        1. Removing segments that are too short
        2. Truncating segments that are too long
        3. Merging overlapping segments
        4. Sorting by score and then by position

        Args:
            segments: List of segments to process

        Returns:
            Processed list of segments ready for clipping
        """
        if not segments:
            return []

        # Filter out segments that are too short
        filtered_segments = [
            segment for segment in segments
            if segment.duration >= self.min_segment_duration
        ]

        # Truncate segments that are too long
        for segment in filtered_segments:
            if segment.duration > self.max_segment_duration:
                # Keep the middle portion of the segment
                excess = segment.duration - self.max_segment_duration
                segment.start += excess / 2
                segment.end -= excess / 2

        # Sort segments by start time for merging
        filtered_segments.sort(key=lambda x: x.start)

        # Merge overlapping segments
        merged_segments = []
        for segment in filtered_segments:
            # If this is our first segment or it doesn't overlap with the last merged segment
            if not merged_segments or segment.start >= merged_segments[-1].end:
                merged_segments.append(segment)
            else:
                # Merge with the previous segment if this one has a higher score
                if segment.score > merged_segments[-1].score:
                    merged_segments[-1] = segment.merge(merged_segments[-1])
                # Otherwise, just extend the previous segment if needed
                elif segment.end > merged_segments[-1].end:
                    merged_segments[-1].end = segment.end

        # Final sort by score (descending) and then by position (ascending)
        merged_segments.sort(key=lambda x: (-x.score, x.start))

        return merged_segments

    def select_top_segments(
        self, segments: List[Segment], max_duration: float, top_count: Optional[int] = None,
        min_spacing: float = None
    ) -> List[Segment]:
        """Select top segments based on score and duration constraints.

        Args:
            segments: List of segments to select from
            max_duration: Maximum total duration of selected segments
            top_count: Optional number of top segments to include
            min_spacing: Minimum time gap between selected segments to avoid repetition

        Returns:
            List of selected top segments
        """
        if not segments:
            return []

        # Use config value for min_spacing if not provided
        min_spacing = min_spacing or get_config("min_spacing_between_segments", 10.0)
        
        # Sort segments by score (highest to lowest)
        segments_by_score = sorted(segments, key=lambda x: -x.score)
        
        # Divide video into zones to find more diverse segments
        video_duration = self.video_duration
        num_zones = get_config("num_video_zones", 12)
        num_zones = min(num_zones, max(1, len(segments) // 5))  # Adjust based on segment count
        zone_size = video_duration / max(1, num_zones)  # Prevent division by zero
        
        # Group segments by zone to ensure diverse selection
        segments_by_zone = {i: [] for i in range(num_zones)}
        
        # Distribute segments into zones
        for segment in segments_by_score:
            if zone_size > 0:
                zone_idx = int(segment.start / zone_size)
                zone_idx = min(zone_idx, num_zones - 1)  # Ensure valid zone
            else:
                zone_idx = 0
            segments_by_zone[zone_idx].append(segment)
        
        # First pass: select highest-scoring segment from each zone
        selected_segments = []
        total_duration = 0.0
        zones_covered = set()
        min_zone_duration = max_duration / num_zones  # Aim for balanced distribution
        
        # Try to get at least one segment from each zone first
        for zone_idx in range(num_zones):
            zone_segments = segments_by_zone[zone_idx]
            if not zone_segments:
                continue
                
            # Take the highest scoring segment from this zone
            segment = zone_segments[0]
            
            # Skip if this would exceed our max duration
            if total_duration + segment.duration > max_duration:
                # But if we haven't selected any segments yet, take this one
                if not selected_segments:
                    selected_segments.append(segment)
                    total_duration += segment.duration
                    zones_covered.add(zone_idx)
                continue
                
            # Skip if too close to already selected segments
            too_close = False
            for selected in selected_segments:
                if abs(segment.start - selected.start) < min_spacing:
                    too_close = True
                    break
                    
            if too_close:
                # Try next segment in this zone if available
                continue_outer = False
                for next_segment in zone_segments[1:2]:  # Try just the next segment
                    if all(abs(next_segment.start - selected.start) >= min_spacing for selected in selected_segments):
                        segment = next_segment
                        too_close = False
                        break
                if too_close:
                    continue
            
            # Add this segment
            selected_segments.append(segment)
            total_duration += segment.duration
            zones_covered.add(zone_idx)
            
            # If we've reached our duration target, stop
            if total_duration >= max_duration:
                break
            
            # If we have a top_count and we've reached it, stop
            if top_count is not None and len(selected_segments) >= top_count:
                break
        
        # Second pass: If we still have room, add more segments from each zone
        # Sort zones by how much we've already taken from them (prioritize underrepresented zones)
        if total_duration < max_duration and (top_count is None or len(selected_segments) < top_count):
            # Calculate how much more duration we need
            remaining_duration = max_duration - total_duration
            
            # Try to fill the remaining duration with segments from zones that aren't fully represented
            for zone_idx in range(num_zones):
                # Skip zones we've already covered
                if zone_idx in zones_covered:
                    continue
                    
                zone_segments = segments_by_zone[zone_idx]
                for segment in zone_segments:
                    # Skip if this would exceed our max duration
                    if segment.duration > remaining_duration:
                        continue
                        
                    # Skip if too close to already selected segments
                    if any(abs(segment.start - selected.start) < min_spacing for selected in selected_segments):
                        continue
                    
                    # Add this segment
                    selected_segments.append(segment)
                    total_duration += segment.duration
                    remaining_duration -= segment.duration
                    zones_covered.add(zone_idx)
                    
                    # If we've reached our duration target, stop
                    if remaining_duration <= 0:
                        break
                    
                    # If we have a top_count and we've reached it, stop
                    if top_count is not None and len(selected_segments) >= top_count:
                        break
                        
                # Break outer loop if we've reached our targets
                if remaining_duration <= 0 or (top_count is not None and len(selected_segments) >= top_count):
                    break
            
        # Third pass: If we still need more segments, take any remaining high-scoring segments
        if total_duration < max_duration * 0.8:  # If we're below 80% of target duration, add more
            remaining_duration = max_duration - total_duration
            
            # Try segments we haven't used yet, regardless of zone
            unused_segments = [s for s in segments_by_score if s not in selected_segments]
            
            for segment in unused_segments:
                # Skip if this would exceed our max duration
                if segment.duration > remaining_duration:
                    continue
                    
                # Use a reduced spacing requirement for the final pass to find more segments
                reduced_spacing = min_spacing * 0.5
                if any(abs(segment.start - selected.start) < reduced_spacing for selected in selected_segments):
                    continue
                
                # Add this segment
                selected_segments.append(segment)
                total_duration += segment.duration
                remaining_duration -= segment.duration
                
                # If we've reached our duration target, stop
                if remaining_duration <= 0:
                    break
                
                # If we have a top_count and we've reached it, stop
                if top_count is not None and len(selected_segments) >= top_count:
                    break
        
        # Sort selected segments by start time for the final sequence
        selected_segments.sort(key=lambda x: x.start)
        
        # Calculate total duration for logging
        total_duration = sum(seg.duration for seg in selected_segments)
        logger.info(f"Selected {len(selected_segments)} segments with total duration: {total_duration:.2f}s")
        
        return selected_segments