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
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text segments to avoid repetition.
        
        Returns a value between 0.0 (completely different) and 1.0 (identical).
        
        Args:
            text1: First text segment
            text2: Second text segment
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Enhanced similarity calculation with more sophisticated checks
        if not text1 or not text2:
            return 0.0
            
        # Normalize and tokenize
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Quick exact match check
        if text1 == text2:
            return 1.0
            
        # Check for substring containment (indicates high similarity)
        # If one text completely contains the other, they're very similar
        if text1 in text2 or text2 in text1:
            # Calculate relative size to determine similarity
            longer = max(len(text1), len(text2))
            shorter = min(len(text1), len(text2))
            if longer > 0:
                # If the shorter text is >70% of the longer text, consider them very similar
                if shorter / longer > 0.7:
                    return 0.9  # Very high similarity
                # If >50%, still fairly similar
                elif shorter / longer > 0.5:
                    return 0.7  # High similarity
            return 0.6  # Moderate similarity for substring containment
        
        # Tokenize into words
        words1 = text1.split()
        words2 = text2.split()
        
        # Handle phrase-level matching
        # Extract phrases (2-3 word sequences) from each text
        phrases1 = set()
        phrases2 = set()
        
        # Generate 2-word phrases
        for i in range(len(words1) - 1):
            phrases1.add(words1[i] + ' ' + words1[i+1])
        for i in range(len(words2) - 1):
            phrases2.add(words2[i] + ' ' + words2[i+1])
            
        # Check phrase overlap
        phrase_intersection = len(phrases1.intersection(phrases2))
        if phrase_intersection > 0:
            # Significant phrase overlap means similar content
            phrase_union = len(phrases1.union(phrases2))
            phrase_similarity = phrase_intersection / phrase_union if phrase_union > 0 else 0.0
            if phrase_similarity > 0.3:  # If 30% of phrases match
                return max(0.6, phrase_similarity)  # At least 0.6 similarity
        
        # Convert to sets for standard word-level comparison
        words1 = set(words1)
        words2 = set(words2)
        
        # Remove common stopwords - extended list for better filtering
        stopwords = {'the', 'and', 'that', 'this', 'with', 'for', 'from', 'but', 'not', 
                     'are', 'is', 'was', 'were', 'have', 'has', 'had', 'you', 'your', 
                     'they', 'their', 'our', 'it', 'a', 'an', 'i', 'me', 'my', 'we', 'us',
                     'so', 'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why',
                     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some',
                     'of', 'in', 'to', 'by', 'at', 'on', 'off', 'out', 'over', 'under'}
        
        # Filter stopwords
        content_words1 = words1 - stopwords
        content_words2 = words2 - stopwords
        
        if not content_words1 or not content_words2:
            # If no content words left after stopword removal,
            # fall back to raw word comparison but with reduced weight
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return (intersection / union if union > 0 else 0.0) * 0.6  # Apply discount factor
        
        # Calculate Jaccard similarity on content words (more meaningful)
        intersection = len(content_words1.intersection(content_words2))
        union = len(content_words1.union(content_words2))
        
        return intersection / union if union > 0 else 0.0

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
        min_spacing: float = None, force_selection: bool = False
    ) -> List[Segment]:
        """Select top segments based on score and duration constraints.

        Args:
            segments: List of segments to select from
            max_duration: Maximum total duration of selected segments
            top_count: Optional number of top segments to include
            min_spacing: Minimum time gap between selected segments to avoid repetition
            force_selection: If True, force use of these exact segments without zone-based filtering

        Returns:
            List of selected top segments
        """
        if not segments:
            return []

        # Use config value for min_spacing if not provided
        min_spacing = min_spacing or get_config("min_spacing_between_segments", 10.0)
        
        # If force_selection is True, we use provided segments with professional editing approach
        # Always use professional editing approach for high-quality results
        if True:
            logger.info("Using enhanced professional segment selection")
            
            # First, sort segments by position to understand the timeline
            segments_by_position = sorted(segments, key=lambda x: x.start)
            
            # Calculate the total video section duration this group spans
            if segments_by_position:
                section_start = segments_by_position[0].start
                section_end = segments_by_position[-1].end
                section_duration = section_end - section_start
                
                # Divide the section into beginning, middle, and end for proper storytelling
                beginning_bound = section_start + (section_duration * 0.3)  # First 30%
                ending_bound = section_end - (section_duration * 0.3)      # Last 30%
                
                # Group segments into beginning, middle, and end
                beginning_segments = [s for s in segments_by_position if s.start <= beginning_bound]
                middle_segments = [s for s in segments_by_position if beginning_bound < s.start < ending_bound]
                ending_segments = [s for s in segments_by_position if s.start >= ending_bound]
                
                # Sort each group by score so we get the best segments from each part
                beginning_segments.sort(key=lambda x: -x.score)
                middle_segments.sort(key=lambda x: -x.score)
                ending_segments.sort(key=lambda x: -x.score)
                
                # Now build a professional sequence with proper pacing
                # We want to allocate around 25% to beginning, 50% to middle, 25% to end
                target_begin_duration = max_duration * 0.25
                target_middle_duration = max_duration * 0.5
                target_end_duration = max_duration * 0.25
                
                # Select key segments from each section
                selected_segments = []
                total_duration = 0.0
                
                # Function to add segments from a section up to a target duration
                def add_segments_from_section(segment_list, target_dur, prev_selected):
                    section_duration = 0.0
                    section_selected = []
                    
                    for segment in segment_list:
                        # Skip if too close to already selected segments
                        if any(abs(segment.start - s.start) < min_spacing for s in prev_selected):
                            continue
                            
                        # Skip very similar segments (if text is available)
                        if hasattr(segment, 'text') and segment.text:
                            if any(hasattr(s, 'text') and s.text and 
                                   self._calculate_text_similarity(segment.text, s.text) > 0.7
                                   for s in prev_selected):
                                continue
                        
                        # Add this segment
                        section_selected.append(segment)
                        section_duration += segment.duration
                        
                        # Stop if we've reached our target
                        if section_duration >= target_dur:
                            break
                    
                    return section_selected
                
                # Add beginning segments - set the scene
                beginning_selected = add_segments_from_section(
                    beginning_segments, target_begin_duration, []
                )
                selected_segments.extend(beginning_selected)
                total_duration += sum(s.duration for s in beginning_selected)
                
                # Add middle segments - the meat of the clip
                middle_selected = add_segments_from_section(
                    middle_segments, target_middle_duration, selected_segments
                )
                selected_segments.extend(middle_selected)
                total_duration += sum(s.duration for s in middle_selected)
                
                # Add ending segments - conclude the story
                end_selected = add_segments_from_section(
                    ending_segments, target_end_duration, selected_segments
                )
                selected_segments.extend(end_selected)
                total_duration += sum(s.duration for s in end_selected)
                
                # If we're still way under our target duration, add more high-scoring segments
                if total_duration < max_duration * 0.7:
                    # Sort all segments by score and add the best unused ones
                    all_by_score = sorted(segments, key=lambda x: -x.score)
                    remaining_duration = max_duration - total_duration
                    
                    for segment in all_by_score:
                        if segment not in selected_segments:
                            # Skip if too close to existing segments
                            if any(abs(segment.start - s.start) < min_spacing for s in selected_segments):
                                continue
                                
                            # Skip if too similar to existing
                            if hasattr(segment, 'text') and segment.text:
                                if any(hasattr(s, 'text') and s.text and 
                                       self._calculate_text_similarity(segment.text, s.text) > 0.7
                                       for s in selected_segments):
                                    continue
                            
                            selected_segments.append(segment)
                            total_duration += segment.duration
                            
                            if total_duration >= max_duration * 0.85:
                                break
            else:
                # Fallback if no segments in position groups
                selected_segments = sorted(segments, key=lambda x: x.start)[:min(5, len(segments))]
            
            # Sort final segments by position for a chronological story
            selected_segments.sort(key=lambda x: x.start)
            
            logger.info(f"Professional selection: {len(selected_segments)} segments, {sum(s.duration for s in selected_segments):.1f}s")
            return selected_segments
        
        # Standard zone-based algorithm if not using forced selection
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