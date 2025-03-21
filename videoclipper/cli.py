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
def process(
    video_input, output_dir, duration, transcribe, whisper_model, min_segment, max_segment, num_clips,
    captions, highlight_words, force_timing_algorithm
):
    """Process a video or YouTube link to create highlight clips.
    
    VIDEO_INPUT can be either a local video file path or a YouTube URL.
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
                        min_segment_length=3.0  # Longer segments for better captions
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
            
            if scene_cache_key and has_cached_segments(scene_cache_key):
                try:
                    cached_scene_segments = load_segments(scene_cache_key)
                    if cached_scene_segments:
                        console.print(f"[green]✓ Loaded {len(cached_scene_segments)} cached scene segments[/green]")
                        scene_segments = cached_scene_segments
                except Exception as e:
                    console.print(f"[yellow]Failed to load cached scene segments: {e}[/yellow]")
            
            # If no cached scene segments, detect scenes
            if not scene_segments:
                try:
                    # Use scene detection to find interesting segments
                    scene_segments = scene_detector.detect_scenes()
                    console.print(f"[green]Found {len(scene_segments)} scene changes[/green]")
                    
                    # Cache the scene segments if we have a video ID
                    if scene_cache_key and scene_segments:
                        if save_segments(scene_segments, scene_cache_key):
                            console.print(f"[green]✓ Cached scene segments for future use[/green]")
                            
                except Exception as e:
                    console.print(f"[yellow]Scene detection failed: {e}, using fallback method[/yellow]")
                    # If scene detection fails, manually create segments spanning the video
                    # This is a fallback to ensure we have some segments to work with
                    video_duration = video_editor._load_video().duration
                    step = video_duration / (num_clips * 2)  # Create 2x as many segments as needed clips
                    for i in range(num_clips * 2):
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
                min_segment_duration=3.0,  # Always use 3 seconds as requested
                max_segment_duration=duration,  # Use the requested clip duration
            )
            
            # Process all segments once to filter, merge, etc.
            try:
                processed_segments = selector.process_segments(segments)
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
                    selected_segments = selector.select_top_segments(
                        processed_segments, 
                        max_duration=duration,
                        min_spacing=min_spacing
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
                        max_duration=duration,  # Use the requested duration
                        viral_style=True,
                        add_captions=True,  # Force captions on for better quality
                        highlight_keywords=highlight_keywords,
                        force_algorithm=force_timing_algorithm,  # Pass the flag to control timing
                        target_duration=duration  # Explicitly set target duration
                    )
                    clip_files.append(output_path)
                    console.print(f"[green]Created clip 1/1 ({clip_duration:.1f}s)[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to create clip: {e}[/red]")
                
                progress.update(task3, advance=1)
            else:
                # For multiple clips, divide the video into time zones
                video_duration = video_editor._duration or 0
                zone_size = video_duration / max(1, num_clips)  # Prevent division by zero
                
                for i in range(num_clips):
                    # Define the time zone for this clip
                    zone_start = i * zone_size
                    zone_end = min(video_duration, (i + 1) * zone_size)
                    
                    # Filter segments that fall primarily within this zone
                    zone_segments = [
                        seg for seg in processed_segments 
                        if seg.start >= zone_start and seg.start < zone_end
                    ]
                    
                    # Add some segments from adjacent zones to ensure enough content
                    if len(zone_segments) < 5:
                        # Add segments that are close to this zone
                        adjacent_segments = [
                            seg for seg in processed_segments
                            if abs(seg.start - zone_start) < zone_size * 0.3
                            or abs(seg.start - zone_end) < zone_size * 0.3
                        ]
                        zone_segments.extend(adjacent_segments)
                        
                        # Remove duplicates
                        zone_segments = list({seg.start: seg for seg in zone_segments}.values())
                    
                    # Select top segments for this clip
                    clip_duration = duration  # Use the requested duration
                    try:
                        # Use min_spacing from config (reduced for multi-clip mode)
                        min_spacing = get_config("min_spacing_between_segments", 10.0) / 2  # Reduce spacing for multi-clip
                        selected_segments = selector.select_top_segments(
                            zone_segments, 
                            max_duration=clip_duration,
                            min_spacing=min_spacing
                        )
                    except Exception as e:
                        console.print(f"[yellow]Error selecting top segments for clip {i+1}: {e}[/yellow]")
                        console.print("[yellow]Using all zone segments[/yellow]")
                        selected_segments = sorted(zone_segments, key=lambda x: x.start)
                    
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
                        
                        # Create the highlight clip with enhanced captions and highlighting
                        output_path, clip_duration = video_editor.create_clip(
                            selected_segments, 
                            clip_path,
                            max_duration=duration,  # Use the requested duration
                            viral_style=True,
                            add_captions=True,  # Force captions on for better quality
                            highlight_keywords=highlight_keywords,
                            force_algorithm=force_timing_algorithm,  # Pass the flag to control timing
                            target_duration=duration  # Explicitly set target duration
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