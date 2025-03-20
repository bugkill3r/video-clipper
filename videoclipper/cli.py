"""
Command-line interface for VideoClipper.
"""

import os
import time
import click
from rich.console import Console
from rich.progress import Progress

from videoclipper.clipper.video_editor import VideoEditor
from videoclipper.analyzer.scene_detector import SceneDetector
from videoclipper.exceptions import FileError, VideoClipperError
from videoclipper.models.segment import Segment
from videoclipper.utils.file_utils import get_file_extension, ensure_directory
from videoclipper.utils.youtube import download_youtube_video, is_youtube_url, get_video_id


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
    default=30,
    help="Target duration of the highlight video in seconds.",
)
@click.option(
    "--transcribe/--no-transcribe",
    default=False,
    help="Enable transcription for content analysis.",
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
    default=5,
    help="Minimum segment duration in seconds.",
)
@click.option(
    "--max-segment",
    type=int,
    default=45,
    help="Maximum segment duration in seconds.",
)
@click.option(
    "--num-clips",
    "-n",
    type=int,
    default=3,
    help="Number of highlight clips to generate.",
)
def process(
    video_input, output_dir, duration, transcribe, whisper_model, min_segment, max_segment, num_clips
):
    """Process a video or YouTube link to create highlight clips.
    
    VIDEO_INPUT can be either a local video file path or a YouTube URL.
    """
    console = Console()

    try:
        # Determine if input is a YouTube URL or local file
        if is_youtube_url(video_input):
            # Handle YouTube URL
            console.print(f"[bold blue]Downloading YouTube video: {video_input}[/bold blue]")
            
            # Create a download directory based on video ID
            video_id = get_video_id(video_input)
            download_dir = os.path.join("downloads", video_id)
            ensure_directory(download_dir)
            
            # Download the video
            with Progress() as progress:
                task = progress.add_task("[cyan]Downloading video...", total=1)
                
                # This happens synchronously
                video_info = download_youtube_video(video_input, download_dir)
                progress.update(task, advance=1)
            
            video_path = video_info["path"]
            console.print(f"[green]✓ Downloaded to: {video_path}[/green]")
            
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
        
        # Implement the actual video processing logic
        segments = []
        with Progress() as progress:
            # Step 1: Load video
            task1 = progress.add_task("[cyan]Loading video...", total=100)
            video_editor = VideoEditor(video_path)
            progress.update(task1, advance=100)
            
            # Step 2: Analyze video for scene changes
            task2 = progress.add_task("[cyan]Analyzing content...", total=100)
            scene_detector = SceneDetector(video_path)
            scene_segments = []
            try:
                # Use scene detection to find interesting segments
                scene_segments = scene_detector.detect_scenes()
                console.print(f"[green]Found {len(scene_segments)} scene changes[/green]")
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
                    segments.append(
                        Segment(
                            start=segment.start,
                            end=segment.end,
                            score=segment.score,
                        )
                    )
                    
            progress.update(task2, advance=100)
            
            # Step 3: Generate highlight clips
            task3 = progress.add_task("[cyan]Generating highlights...", total=num_clips)
            
            # Process segments using the improved segment selector
            selector = SegmentSelector(
                video_duration=video_editor._duration or 0,
                min_segment_duration=min_segment,
                max_segment_duration=max_segment,
            )
            
            # Process all segments once to filter, merge, etc.
            processed_segments = selector.process_segments(segments)
            
            clip_files = []
            if num_clips == 1:
                # For a single clip, just use all processed segments with improved selection
                selected_segments = selector.select_top_segments(
                    processed_segments, 
                    max_duration=duration,
                    min_spacing=10.0  # Minimum 10s spacing to avoid repetitive content
                )
                
                # Create the output path for this clip
                clip_name = "highlight_1.mp4"
                clip_path = os.path.join(output_dir, clip_name)
                
                try:
                    # Create the highlight clip with viral style
                    output_path, clip_duration = video_editor.create_clip(
                        selected_segments, 
                        clip_path,
                        max_duration=max_segment,
                        viral_style=True
                    )
                    clip_files.append(output_path)
                    console.print(f"[green]Created clip 1/1 ({clip_duration:.1f}s)[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to create clip: {e}[/red]")
                
                progress.update(task3, advance=1)
            else:
                # For multiple clips, divide the video into time zones
                video_duration = video_editor._duration or 0
                zone_size = video_duration / num_clips
                
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
                    clip_duration = min(duration, max_segment)
                    selected_segments = selector.select_top_segments(
                        zone_segments, 
                        max_duration=clip_duration,
                        min_spacing=5.0  # Minimum 5s spacing
                    )
                    
                    # Create the output path for this clip
                    clip_name = f"highlight_{i+1}.mp4"
                    clip_path = os.path.join(output_dir, clip_name)
                    
                    try:
                        # Create the highlight clip with viral style
                        output_path, clip_duration = video_editor.create_clip(
                            selected_segments, 
                            clip_path,
                            max_duration=max_segment,
                            viral_style=True
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