"""Segment selection and processing for video clipping."""

from typing import List, Optional

from videoclipper.models.segment import Segment


class SegmentSelector:
    """Selects and processes segments for inclusion in the final highlight video."""

    def __init__(
        self,
        video_duration: float,
        min_segment_duration: float = 3.0,
        max_segment_duration: float = 15.0,
    ) -> None:
        """Initialize the segment selector.

        Args:
            video_duration: Duration of the source video in seconds
            min_segment_duration: Minimum allowed segment duration in seconds
            max_segment_duration: Maximum allowed segment duration in seconds
        """
        self.video_duration = video_duration
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration

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
        self, segments: List[Segment], max_duration: float, top_count: Optional[int] = None
    ) -> List[Segment]:
        """Select top segments based on score and duration constraints.

        Args:
            segments: List of segments to select from
            max_duration: Maximum total duration of selected segments
            top_count: Optional number of top segments to include

        Returns:
            List of selected top segments
        """
        if not segments:
            return []

        # If top_count is specified, just take that many top segments
        if top_count is not None:
            return sorted(segments[:min(top_count, len(segments))], key=lambda x: x.start)

        # Otherwise, take as many segments as will fit within max_duration
        selected_segments = []
        total_duration = 0.0

        for segment in segments:
            if total_duration + segment.duration <= max_duration:
                selected_segments.append(segment)
                total_duration += segment.duration
            else:
                break

        # Sort selected segments by start time for the final sequence
        selected_segments.sort(key=lambda x: x.start)

        return selected_segments
