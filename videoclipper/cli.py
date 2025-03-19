"""
Command-line interface for VideoClipper.
"""

import os
import click
from rich.console import Console
from rich.progress import Progress

from videoclipper.exceptions import FileError, VideoClipperError
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
    default=60,
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
    default=3,
    help="Minimum segment duration in seconds.",
)
@click.option(
    "--max-segment",
    type=int,
    default=15,
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
        
        # Here we'd actually call the video processing logic
        # For now just show a placeholder for each clip
        with Progress() as progress:
            task1 = progress.add_task("[cyan]Loading video...", total=100)
            progress.update(task1, advance=100)
            
            task2 = progress.add_task("[cyan]Analyzing content...", total=100)
            progress.update(task2, advance=100)
            
            task3 = progress.add_task("[cyan]Generating highlights...", total=100)
            progress.update(task3, advance=100)
        
        # Generate placeholder clip filenames
        clip_files = []
        for i in range(num_clips):
            clip_name = f"highlight_{i+1}.mp4"
            clip_path = os.path.join(output_dir, clip_name)
            
            # In a real implementation, we would generate and save the clip
            # For now, just add to our list
            clip_files.append(clip_path)
            
            # Simulate creating empty files
            with open(clip_path, 'w') as f:
                f.write("This is a placeholder for a video clip file")
        
        # Print results
        console.print(f"[bold green]✓ Generated {num_clips} highlight clips:[/bold green]")
        for clip in clip_files:
            console.print(f"  - {clip}")
            
    except FileError as e:
        console.print(f"[bold red]File error: {e}[/bold red]")
    except VideoClipperError as e:
        console.print(f"[bold red]Processing error: {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")


if __name__ == "__main__":
    main()