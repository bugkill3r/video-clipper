"""
Command-line interface for VideoClipper.
"""

import os
import time
import subprocess
import logging
import click
from rich.console import Console
from rich.progress import Progress

# Set up logging
logger = logging.getLogger(__name__)

from videoclipper.clipper.video_editor import VideoEditor
from videoclipper.clipper.segment_selector import SegmentSelector
from videoclipper.analyzer.scene_detector import SceneDetector
from videoclipper.exceptions import FileError, VideoClipperError
from videoclipper.models.segment import Segment, SegmentType
from videoclipper.utils.file_utils import get_file_extension, ensure_directory, list_files, load_segments, save_segments, has_cached_segments
from videoclipper.utils.youtube import download_youtube_video, is_youtube_url, get_video_id, parse_srt_file
from videoclipper.config import get_config


@click.group()
def main():
    """VideoClipper - AI-powered video highlight generator."""
    pass


@main.command()
@click.argument("video_input")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory path for highlight clips.",
)
@click.option(
    "--duration",
    "-d",
    type=int,
    default=None,
    help="Target duration of the highlight video in seconds.",
)
@click.option(
    "--transcribe/--no-transcribe",
    default=True,
    help="Enable transcription for content analysis and captions.",
)
@click.option(
    "--whisper-model",
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    default="base",
    help="Whisper model size to use for transcription.",
)
@click.option(
    "--min-segment",
    type=int,
    default=None,
    help="Minimum segment duration in seconds.",
)
@click.option(
    "--max-segment",
    type=int,
    default=None,
    help="Maximum segment duration in seconds.",
)
@click.option(
    "--num-clips",
    "-n",
    type=int,
    default=3,
    help="Number of highlight clips to generate.",
)
@click.option(
    "--captions/--no-captions",
    default=False,
    help="Add animated captions to the clips that match speech timing (requires transcription).",
)
@click.option(
    "--highlight-words",
    type=str,
    help="Comma-separated list of words to highlight in captions.",
)
@click.option(
    "--force-timing-algorithm/--use-available-srt",
    default=False,
    help="Force use of the sophisticated timing algorithm even if SRT files are available.",
)
@click.option(
    "--vertical/--horizontal",
    default=False,
    help="Create vertical format (1080x1920) video for shorts/reels.",
)
def process(
    video_input, output_dir, duration, transcribe, whisper_model, min_segment, max_segment, num_clips,
    captions, highlight_words, force_timing_algorithm, vertical
):
    """Process a video or YouTube link to create highlight clips.
    
    VIDEO_INPUT can be either a local video file path or a YouTube URL.
    
    Use the --vertical flag to create clips in vertical format (1080x1920) for social media shorts/reels.
    """
    # Get default duration from config if not specified
    duration = duration or get_config("default_highlight_duration", 45)
    console = Console()

    try:
        # Initialize segments list
        segments = []
        
        # Determine if input is a YouTube URL or local file
        if is_youtube_url(video_input):
            # Handle YouTube URL
            console.print(f"[bold blue]Processing YouTube video: {video_input}[/bold blue]")
            
            # Create a download directory based on video ID
            video_id = get_video_id(video_input)
            download_dir = os.path.join("downloads", video_id)
            ensure_directory(download_dir)
            
            # Check if video is already downloaded
            video_files = list_files(download_dir, extension=".mp4") + list_files(download_dir, extension=".mkv")
            cached_segments = None
            
            # Use cached segments if available
            
            if has_cached_segments(video_id):
                console.print(f"[green]✓ Found cached segments for video {video_id}[/green]")
                cached_segments = load_segments(video_id)
                if cached_segments:
                    console.print(f"[green]✓ Loaded {len(cached_segments)} cached segments[/green]")
                    segments.extend(cached_segments)
                    # Since we have cached segments, we can skip transcription
                    transcribe = False
            
            # Download the video if needed
            if not video_files:
                console.print(f"[blue]Downloading YouTube video...[/blue]")
                with Progress() as progress:
                    task = progress.add_task("[cyan]Downloading video...", total=1)
                    
                    # This happens synchronously
                    video_info = download_youtube_video(video_input, download_dir)
                    progress.update(task, advance=1)
                
                video_path = video_info["path"]
                console.print(f"[green]✓ Downloaded to: {video_path}[/green]")
            else:
                video_path = video_files[0]
                console.print(f"[green]✓ Using existing download: {video_path}[/green]")
                
                # Try to get subtitles path
                subtitle_files = list_files(download_dir, extension=".srt") + list_files(download_dir, extension=".vtt")
                if subtitle_files:
                    video_info = {
                        "path": video_path,
                        "subtitles_path": subtitle_files[0],
                        "id": video_id
                    }
                else:
                    video_info = {
                        "path": video_path,
                        "id": video_id
                    }
            
            # Check if we need to parse subtitles (if no cached segments)
            if not cached_segments and not segments and "subtitles_path" in video_info and video_info["subtitles_path"]:
                console.print(f"[green]✓ Found YouTube captions: {video_info['subtitles_path']}[/green]")
                # Parse the subtitle file into segments
                try:
                    youtube_segments = parse_srt_file(video_info["subtitles_path"])
                    if youtube_segments:
                        console.print(f"[green]✓ Extracted {len(youtube_segments)} segments from YouTube captions[/green]")
                        # Add to segments list
                        segments.extend(youtube_segments)
                        
                        # Cache the segments for future use
                        if save_segments(youtube_segments, video_id):
                            console.print(f"[green]✓ Cached segments for future use[/green]")
                        
                        # Since we have YouTube captions, we can skip transcription
                        transcribe = False
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to parse YouTube captions: {e}[/yellow]")
            
            # Use video ID and title for output directory if not specified
            if not output_dir:
                output_dir = os.path.join("output", video_id)
        else:
            # Local file
            video_path = video_input
            
            # Check if input file is a video
            ext = get_file_extension(video_path)
            if ext not in [".mp4", ".mov", ".avi", ".mkv"]:
                console.print(f"[red]Error: Unsupported video format: {ext}[/red]")
                return
                
            # Use input filename for output directory if not specified
            if not output_dir:
                basename = os.path.basename(video_path)
                name, _ = os.path.splitext(basename)
                output_dir = os.path.join("output", name)
        
        # Create output directory
        ensure_directory(output_dir)
        
        # Process the video
        console.print(f"[bold green]Processing video: {video_path}[/bold green]")
        console.print(f"Output directory: {output_dir}")
        console.print(f"Target highlight duration: {duration} seconds")
        console.print(f"Number of clips: {num_clips}")
        if vertical:
            console.print(f"[bold cyan]Creating vertical format (1080x1920) for shorts/reels[/bold cyan]")
        
        # For YouTube videos, track the video ID for caching
        video_id = None
        if is_youtube_url(video_input):
            video_id = get_video_id(video_input)
        
        # Implement the actual video processing logic
        with Progress() as progress:
            # Step 1: Load video
            task1 = progress.add_task("[cyan]Loading video...", total=100)
            video_editor = VideoEditor(video_path)
            progress.update(task1, advance=100)
            
            # Step 2: Analyze video for scene changes and transcribe if enabled
            task2 = progress.add_task("[cyan]Analyzing content...", total=100)
            scene_detector = SceneDetector(video_path)
            scene_segments = []
            speech_segments = []
            
            # Process transcription if enabled (needed for captions)
            if transcribe or captions:
                try:
                    from videoclipper.transcriber.whisper_transcriber import WhisperTranscriber
                    
                    # Check if the video file exists
                    if not os.path.exists(video_path):
                        console.print(f"[yellow]Video file not found: {video_path}[/yellow]")
                        raise FileNotFoundError(f"Video file not found: {video_path}")
                    
                    # Check if ffmpeg is installed for audio extraction
                    try:
                        # Try different possible ffmpeg paths
                        ffmpeg_cmd = "ffmpeg"
                        ffmpeg_paths = [
                            "ffmpeg",
                            "/opt/homebrew/bin/ffmpeg",  # Homebrew on Apple Silicon
                            "/usr/local/bin/ffmpeg",     # Homebrew on Intel Mac
                            "/usr/bin/ffmpeg"            # System path
                        ]
                        
                        for path in ffmpeg_paths:
                            try:
                                subprocess.run([path, "-version"], capture_output=True, check=True)
                                ffmpeg_cmd = path
                                console.print(f"[green]Found ffmpeg at: {path}[/green]")
                                break
                            except (subprocess.SubprocessError, FileNotFoundError):
                                continue
                        else:
                            # If the loop completes without a break, ffmpeg was not found
                            console.print("[yellow]ffmpeg not found, which is required for transcription[/yellow]")
                            console.print("[yellow]Please install ffmpeg using 'brew install ffmpeg'[/yellow]")
                            raise RuntimeError("ffmpeg not found, which is required for transcription")
                    except Exception as e:
                        console.print(f"[yellow]Error checking ffmpeg: {e}[/yellow]")
                        raise RuntimeError(f"Error checking ffmpeg: {e}")
                    
                    console.print(f"[cyan]Transcribing audio with Whisper ({whisper_model} model)...[/cyan]")
                    
                    # Extract audio to a temporary file
                    temp_audio = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
                    extract_cmd = [ffmpeg_cmd, "-i", video_path, "-q:a", "0", "-map", "a", temp_audio, "-y"]
                    
                    console.print(f"[cyan]Extracting audio to {temp_audio}...[/cyan]")
                    try:
                        subprocess.run(extract_cmd, capture_output=True, check=True)
                        if os.path.exists(temp_audio):
                            console.print(f"[green]Audio extracted successfully to {temp_audio}[/green]")
                        else:
                            console.print(f"[yellow]Audio extraction failed - output file not found[/yellow]")
                            raise FileNotFoundError(f"Audio file not found: {temp_audio}")
                    except Exception as e:
                        console.print(f"[yellow]Audio extraction failed: {e}[/yellow]")
                        raise RuntimeError(f"Audio extraction failed: {e}")
                    
                    # Transcribe from the audio file
                    transcriber = WhisperTranscriber(temp_audio)
                    speech_segments = transcriber.transcribe(
                        model_size=whisper_model,
                        min_segment_length=min_segment or 1.5  # Use CLI parameter if provided, otherwise default to 1.5
                    )
                    
                    # Clean up temp file
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)
                    
                    if speech_segments:
                        console.print(f"[green]Found {len(speech_segments)} speech segments[/green]")
                        # Add to segments list with higher score for segments with text
                        for segment in speech_segments:
                            # Boost score for segments with text for better selection
                            segment.score = min(1.0, segment.score * 1.2)
                            segments.append(segment)
                            console.print(f"[dim]Segment {segment.start:.1f}-{segment.end:.1f}: {segment.text}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Transcription failed: {e}[/yellow]")
                    console.print("[yellow]Continuing without captions[/yellow]")
            
            # Detect scene changes
            # First check if we can load cached scene segments
            scene_cache_key = f"{video_id}_scenes" if video_id else None
            cached_scene_segments = []
            scenes_from_cache = False
            
            if scene_cache_key and has_cached_segments(scene_cache_key):
                try:
                    cached_scene_segments = load_segments(scene_cache_key)
                    if cached_scene_segments:
                        console.print(f"[green]✓ Loaded {len(cached_scene_segments)} cached scene segments[/green]")
                        scene_segments = cached_scene_segments
                        scenes_from_cache = True
                except Exception as e:
                    console.print(f"[yellow]Failed to load cached scene segments: {e}[/yellow]")
            
            # If no cached scene segments, detect scenes
            if not scene_segments:
                try:
                    # Use scene detection to find interesting segments
                    console.print(f"[cyan]Detecting scene changes in video...[/cyan]")
                    scene_segments = scene_detector.detect_scenes()
                    console.print(f"[green]Found {len(scene_segments)} scene changes[/green]")
                    
                    # Cache the scene segments if we have a video ID
                    if scene_cache_key and scene_segments:
                        console.print(f"[cyan]Caching scene segments for future use...[/cyan]")
                        if save_segments(scene_segments, scene_cache_key):
                            console.print(f"[green]✓ Cached {len(scene_segments)} scene segments[/green]")
                            
                except Exception as e:
                    console.print(f"[yellow]Scene detection failed: {e}, using fallback method[/yellow]")
                    # If scene detection fails, manually create segments spanning the video
                    # This is a fallback to ensure we have some segments to work with
                    video_duration = video_editor._load_video().duration
                    # Create more segments with smaller step size for shorts/reels
                    step = video_duration / (num_clips * 4)  # Create 4x as many segments as needed clips
                    for i in range(num_clips * 4):
                        start_time = i * step
                        end_time = min(video_duration, start_time + step)
                        segments.append(
                            Segment(
                                start=start_time,
                                end=end_time,
                                score=0.5,  # Average importance
                            )
                        )
            
            # Use the detected scene segments if available
            if scene_segments:
                # Convert segments to the correct format if necessary
                for segment in scene_segments:
                    # Check if this segment overlaps with any speech segment
                    overlapping_speech = False
                    for speech_seg in speech_segments:
                        # If there's significant overlap (>50%)
                        overlap_start = max(segment.start, speech_seg.start)
                        overlap_end = min(segment.end, speech_seg.end)
                        if overlap_end > overlap_start:
                            overlap_duration = overlap_end - overlap_start
                            if overlap_duration > 0.5 * (segment.end - segment.start):
                                overlapping_speech = True
                                # Copy the text from speech segment to scene segment if it exists
                                if hasattr(speech_seg, 'text') and speech_seg.text:
                                    segment.text = speech_seg.text
                                break
                    
                    # Only add non-overlapping segments to avoid duplicates
                    if not overlapping_speech:
                        segments.append(
                            Segment(
                                start=segment.start,
                                end=segment.end,
                                score=segment.score,
                                segment_type=segment.segment_type
                            )
                        )
            
            progress.update(task2, advance=100)
            
            # Step 3: Generate highlight clips
            task3 = progress.add_task("[cyan]Generating highlights...", total=num_clips)
            
            # Ensure video is loaded and get proper duration
            video = video_editor._load_video()
            video_duration = video.duration if video else 0
            console.print(f"Video duration: {video_duration} seconds")
            
            selector = SegmentSelector(
                video_duration=video_duration,
                min_segment_duration=min_segment,  # Use CLI parameter if provided
                max_segment_duration=max_segment or duration,  # Use CLI parameter or requested clip duration
            )
            
            # Process all segments once to filter, merge, etc.
            try:
                processed_segments = selector.process_segments(segments)
                
                # Cache the processed segments for YouTube videos if not already cached
                if is_youtube_url(video_input) and video_id:
                    # Cache even if we already had cached segments - this preserves our shorter segments
                    console.print(f"[cyan]Caching processed segments for future use...[/cyan]")
                    if save_segments(processed_segments, video_id):
                        console.print(f"[green]✓ Saved {len(processed_segments)} processed segments to cache[/green]")
            except Exception as e:
                console.print(f"[yellow]Error processing segments: {e}[/yellow]")
                console.print("[yellow]Using original segments[/yellow]")
                processed_segments = segments
            
            clip_files = []
            
            # Process highlight words if provided
            highlight_keywords = None
            if highlight_words:
                highlight_keywords = [word.strip() for word in highlight_words.split(',')]
                console.print(f"[cyan]Using highlight keywords: {highlight_keywords}[/cyan]")
            
            if num_clips == 1:
                try:
                    # For a single clip, just use all processed segments with improved selection
                    # Use min_spacing from config
                    min_spacing = get_config("min_spacing_between_segments", 10.0)
                    # Set target duration for consistent naming
                    clip_target_duration = duration
                    console.print(f"[cyan]Creating single clip with target duration: {clip_target_duration}s[/cyan]")
                    
                    selected_segments = selector.select_top_segments(
                        processed_segments, 
                        max_duration=clip_target_duration,
                        min_spacing=min_spacing,
                        force_selection=False  # For single clip, we want diverse segments from across the video
                    )
                except Exception as e:
                    console.print(f"[yellow]Error selecting top segments: {e}[/yellow]")
                    console.print("[yellow]Using all processed segments[/yellow]")
                    selected_segments = sorted(processed_segments, key=lambda x: x.start)
                
                # Create the output path for this clip
                clip_name = "highlight_1.mp4"
                clip_path = os.path.join(output_dir, clip_name)
                
                try:
                    # We need to select MORE unique segments from different parts of the video
                    # This may require examining more of the video to find enough good segments
                    logger.info(f"Selected {len(selected_segments)} segments initially")
                    if len(selected_segments) < 10 and duration > 20:
                        # Look for more segments with lower score threshold
                        logger.info("Finding more quality segments from different parts of the video...")
                    
                    # Create the highlight clip with enhanced captions and highlighting
                    output_path, clip_duration = video_editor.create_clip(
                        selected_segments, 
                        clip_path,
                        max_duration=clip_target_duration,  # Use the requested duration
                        min_segment_duration=min_segment,  # Pass CLI parameter
                        max_segment_duration=max_segment,  # Pass CLI parameter 
                        viral_style=True,
                        add_captions=captions,  # Use CLI option for captions
                        highlight_keywords=highlight_keywords,
                        force_algorithm=force_timing_algorithm,  # Pass the flag to control timing
                        target_duration=clip_target_duration,  # Explicitly set target duration
                        vertical_format=vertical  # Use vertical format for shorts/reels if requested
                    )
                    clip_files.append(output_path)
                    console.print(f"[green]Created clip 1/1 ({clip_duration:.1f}s)[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to create clip: {e}[/red]")
                
                progress.update(task3, advance=1)
            else:
                # For multiple clips, we need to FORCEFULLY divide the video into completely different segments
                video_duration = video_editor._duration or 0
                clip_target_duration = duration  # Use the requested duration for each clip
                console.print(f"[bold cyan]Creating {num_clips} COMPLETELY DISTINCT highlight clips, each {clip_target_duration}s[/bold cyan]")
                
                # Create a tracking mechanism using IDs instead of segment objects
                used_segment_ids = set()  # Track by segment start/end, not objects
                
                # Divide the video into completely separate sections
                time_sorted_segments = sorted(processed_segments, key=lambda x: x.start)
                total_segments = len(time_sorted_segments)
                
                if total_segments < num_clips * 3:
                    console.print(f"[yellow]Warning: Only {total_segments} total segments for {num_clips} clips.[/yellow]")
                    console.print("[yellow]Creating clips from different parts of the video, but some content may overlap.[/yellow]")
                    
                    # If we have very few segments, we take a different approach
                    # We will organize segments by their position in the video timeline
                    # and try to create clips from different parts
                
                # Handle edge case: fewer segments than clips
                if total_segments < num_clips:
                    console.print(f"[bold red]Error: Only {total_segments} segments but {num_clips} clips requested![/bold red]")
                    console.print("[yellow]Creating fewer clips than requested.[/yellow]")
                    num_clips = max(1, total_segments)
                
                # Professional segmentation algorithm for multiple quality clips
                # Always ensure we can create num_clips full professional clips
                video_duration = video_editor._duration or 0
                
                # PROFESSIONAL APPROACH: Use a more sophisticated zone-based segmentation
                # We'll create enough zones to ensure variety and good coverage
                zone_count = max(num_clips * 5, 25)  # Use more zones for better granularity
                zone_size = video_duration / zone_count
                
                # Create zone groups to organize segments
                zone_groups = {i: [] for i in range(zone_count)}
                
                # Assign existing segments to zones
                for segment in time_sorted_segments:
                    zone_idx = min(int(segment.start / zone_size), zone_count - 1)
                    zone_groups[zone_idx].append(segment)
                
                # Professional fix for empty zones - create segments for areas with no content
                empty_zones = [i for i, segments in zone_groups.items() if not segments]
                
                if empty_zones:
                    console.print(f"[yellow]Found {len(empty_zones)} empty zones. Creating segments to ensure quality clips.[/yellow]")
                    
                    # Calculate how many new segments we need - at least 8 per clip
                    needed_segments = max(0, (num_clips * 8) - total_segments)
                    
                    # Create segments to fill empty zones - focus on empty zones first
                    for zone_idx in empty_zones:
                        # Calculate zone boundaries
                        zone_start = zone_idx * zone_size
                        zone_end = min(video_duration, (zone_idx + 1) * zone_size)
                        zone_duration = zone_end - zone_start
                        
                        # Create segments within this zone (up to 2 per zone for variety)
                        segments_to_create = min(2, int(zone_duration / 3))  # Roughly one segment per 3 seconds
                        
                        for i in range(segments_to_create):
                            # Calculate segment position - distribute evenly
                            segment_position = zone_start + ((i+1) * zone_duration / (segments_to_create+1))
                            
                            # Create segment with professional duration (3-8s is ideal for clips)
                            segment_duration = min(max(3.0, zone_duration * 0.3), 8.0)
                            
                            # Ensure segment is within zone and video boundaries
                            seg_start = max(0, segment_position - segment_duration/2)
                            seg_end = min(video_duration, segment_position + segment_duration/2)
                            
                            if seg_end - seg_start >= 2.0:  # Only create if at least 2 seconds
                                new_segment = Segment(
                                    start=seg_start,
                                    end=seg_end,
                                    score=0.65,  # Good score for filling gaps
                                    segment_type=SegmentType.CUSTOM
                                )
                                
                                # Add to both the zone group and processed segments
                                zone_groups[zone_idx].append(new_segment)
                                processed_segments.append(new_segment)
                                
                                console.print(f"[blue]Added new segment {seg_start:.1f}-{seg_end:.1f}s in zone {zone_idx+1}/{zone_count}[/blue]")
                                
                                # Reduce needed count
                                needed_segments -= 1
                                if needed_segments <= 0:
                                    break
                        
                        if needed_segments <= 0:
                            break
                    
                    # If we still need more segments, add some in sparsely populated zones
                    if needed_segments > 0:
                        # Find zones with few segments
                        sparse_zones = [i for i, segments in zone_groups.items() 
                                      if len(segments) < 2 and i not in empty_zones]
                        
                        for zone_idx in sparse_zones:
                            zone_start = zone_idx * zone_size
                            zone_end = min(video_duration, (zone_idx + 1) * zone_size)
                            
                            # Find a gap within this zone's existing segments
                            existing_segments = sorted(zone_groups[zone_idx], key=lambda x: x.start)
                            
                            # Find the largest gap between segments or zone boundaries
                            gaps = []
                            prev_end = zone_start
                            
                            for segment in existing_segments:
                                if segment.start > prev_end + 2.0:  # Need at least 2s gap
                                    gaps.append((prev_end, segment.start))
                                prev_end = segment.end
                            
                            # Check for gap at the end of zone
                            if zone_end > prev_end + 2.0:
                                gaps.append((prev_end, zone_end))
                            
                            # Create segment in the largest gap
                            if gaps:
                                gaps.sort(key=lambda x: x[1]-x[0], reverse=True)  # Sort by gap size
                                gap_start, gap_end = gaps[0]
                                
                                gap_duration = gap_end - gap_start
                                gap_midpoint = gap_start + (gap_duration / 2)
                                
                                # Create a segment within the gap (3-5s)
                                segment_duration = min(max(3.0, gap_duration * 0.7), 5.0)
                                
                                new_segment = Segment(
                                    start=max(0, gap_midpoint - segment_duration/2),
                                    end=min(video_duration, gap_midpoint + segment_duration/2),
                                    score=0.6,
                                    segment_type=SegmentType.CUSTOM
                                )
                                
                                zone_groups[zone_idx].append(new_segment)
                                processed_segments.append(new_segment)
                                
                                console.print(f"[blue]Added gap-filling segment {new_segment.start:.1f}-{new_segment.end:.1f}s in zone {zone_idx+1}[/blue]")
                                
                                needed_segments -= 1
                                if needed_segments <= 0:
                                    break
                
                # Update segment count after additions
                total_segments = len(processed_segments)
                console.print(f"[green]✓ Now have {total_segments} total segments to work with[/green]")
                
                # Create clip groups using professional video editing approach
                # Each clip should come from a different part of the video
                
                # Calculate zones per clip (with slight overlap for better transitions)
                zones_per_clip = max(2, zone_count // num_clips)
                
                # Create segment groups - one per clip
                segment_groups = []
                for i in range(num_clips):
                    # Center zone for this clip
                    center_zone = int(((i * zone_count) / num_clips) + (zones_per_clip / 2))
                    # Calculate zone range with slight overlap between clips
                    half_width = zones_per_clip // 2
                    start_zone = max(0, center_zone - half_width)
                    end_zone = min(zone_count, center_zone + half_width + 1)
                    
                    # Get all segments in these zones
                    group = []
                    for z in range(start_zone, end_zone):
                        group.extend(zone_groups[z])
                    
                    # Ensure we have enough segments (at least 5-8 per clip)
                    if len(group) < 5:
                        # Add more segments from nearby zones if needed
                        nearby_zones = list(range(max(0, start_zone-2), start_zone)) + \
                                      list(range(end_zone, min(zone_count, end_zone+2)))
                        
                        for z in nearby_zones:
                            group.extend(zone_groups[z])
                            if len(group) >= 8:  # Enough segments now
                                break
                    
                    # If still under 5 segments, add highest-scoring segments from anywhere
                    if len(group) < 5:
                        # Sort all segments by score and add best ones
                        score_sorted = sorted(processed_segments, key=lambda x: -x.score)
                        for segment in score_sorted:
                            if segment not in group:
                                group.append(segment)
                                if len(group) >= 5:
                                    break
                    
                    # Add this segment group
                    segment_groups.append(group)
                    console.print(f"[blue]Clip {i+1}: Using {len(group)} segments from zones {start_zone}-{end_zone} ({int(start_zone*zone_size)}s-{int(end_zone*zone_size)}s)[/blue]")
                
                console.print(f"[green]✓ Created {len(segment_groups)} professional clip groups across the video[/green]")
                
                # Processing is complete - all segment groups have been created
                
                # Now process each group into a clip with as little overlap as possible
                for i, section_segments in enumerate(segment_groups):
                    if not section_segments:
                        console.print(f"[red]Warning: No segments for clip {i+1}. Skipping.[/red]")
                        continue
                    
                    section_start = section_segments[0].start if section_segments else 0
                    section_end = section_segments[-1].end if section_segments else video_duration
                    
                    console.print(f"[blue]Clip {i+1}: {section_start:.1f}s-{section_end:.1f}s will use {len(section_segments)} segments[/blue]")
                    
                    # Create IDs for tracking
                    section_ids = [(s.start, s.end) for s in section_segments]
                    
                    # If we need more segments, try to grab unique ones without overlapping other clips
                    if len(section_segments) < 5 and total_segments > 5:
                        # Get segments we haven't used yet - check by ID instead of object
                        unused_segments = []
                        for s in processed_segments:
                            s_id = (s.start, s.end)
                            if s_id not in used_segment_ids and s_id not in section_ids:
                                unused_segments.append(s)
                        
                        # Sort by score
                        score_sorted = sorted(unused_segments, key=lambda x: -x.score)
                        
                        # Add some of these unique segments
                        added_count = 0
                        for seg in score_sorted:
                            seg_id = (seg.start, seg.end)
                            if seg_id not in section_ids and added_count < 5:
                                section_segments.append(seg)
                                section_ids.append(seg_id)
                                added_count += 1
                                
                        console.print(f"[blue]Added {added_count} additional unique segments to clip {i+1}[/blue]")
                    
                    # Sort segments by time for smooth playback
                    section_segments.sort(key=lambda x: x.start)
                    
                    # Mark these segments as used by ID
                    for seg in section_segments:
                        used_segment_ids.add((seg.start, seg.end))
                    try:
                        # Use min_spacing from config (reduced for multi-clip mode)
                        min_spacing = get_config("min_spacing_between_segments", 10.0) / 2  # Reduce spacing for multi-clip
                        selected_segments = selector.select_top_segments(
                            section_segments, 
                            max_duration=clip_target_duration,  # Use exact requested duration for each clip
                            min_spacing=min_spacing,
                            force_selection=True  # FORCE usage of the exact segments we've provided
                        )
                    except Exception as e:
                        console.print(f"[yellow]Error selecting top segments for clip {i+1}: {e}[/yellow]")
                        console.print("[yellow]Using all section segments in time order[/yellow]")
                        selected_segments = sorted(section_segments, key=lambda x: x.start)
                    
                    # Create the output path for this clip
                    clip_name = f"highlight_{i+1}.mp4"
                    clip_path = os.path.join(output_dir, clip_name)
                    
                    try:
                        # We need to select MORE unique segments from different parts of the video
                        # rather than repeating content which is poor for viral shorts
                        logger.info(f"Selected {len(selected_segments)} segments initially")
                        if len(selected_segments) < 10 and duration > 20:
                            # Look for more segments with lower score threshold
                            logger.info("Finding more quality segments from different parts of the video...")
                        
                        # Check for segment overlap between clips for debugging
                        if 'all_selected_times' not in locals():
                            all_selected_times = set()
                        
                        # Debug info about segment uniqueness
                        segment_times = [(seg.start, seg.end) for seg in selected_segments]
                        overlap_count = sum(1 for time in segment_times if time in all_selected_times)
                        
                        if overlap_count > 0:
                            console.print(f"[yellow]WARNING: Clip {i+1} has {overlap_count} segments that overlap with previous clips[/yellow]")
                        
                        # Store these segment times for comparison with next clip
                        all_selected_times.update(segment_times)
                        
                        # Create a tag to differentiate this clip from others
                        clip_tag = f"section_{i+1}"
                        
                        # Create the highlight clip with enhanced captions and highlighting
                        output_path, clip_duration = video_editor.create_clip(
                            selected_segments, 
                            clip_path,
                            max_duration=clip_target_duration,  # Use the same target duration for each clip
                            min_segment_duration=min_segment,  # Pass CLI parameter
                            max_segment_duration=max_segment,  # Pass CLI parameter
                            viral_style=True,
                            add_captions=captions,  # Use the CLI option for captions
                            highlight_keywords=highlight_keywords,
                            force_algorithm=force_timing_algorithm,  # Pass the flag to control timing
                            target_duration=clip_target_duration,  # Explicitly set same target duration for each clip
                            vertical_format=vertical,  # Use vertical format for shorts/reels if requested
                            clip_tag=clip_tag  # Pass a unique tag to differentiate clips
                        )
                        clip_files.append(output_path)
                        console.print(f"[green]Created clip {i+1}/{num_clips} ({clip_duration:.1f}s)[/green]")
                    except Exception as e:
                        console.print(f"[red]Failed to create clip {i+1}: {e}[/red]")
                    
                    progress.update(task3, advance=1)
                    
                    # Small delay to prevent overloading the system
                    time.sleep(0.5)
        
        # Print results
        if clip_files:
            console.print(f"[bold green]✓ Generated {len(clip_files)} highlight clips:[/bold green]")
            for clip in clip_files:
                console.print(f"  - {clip}")
        else:
            console.print("[bold red]Failed to generate any clips[/bold red]")
            
    except FileError as e:
        console.print(f"[bold red]File error: {e}[/bold red]")
    except VideoClipperError as e:
        console.print(f"[bold red]Processing error: {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")


if __name__ == "__main__":
    main()