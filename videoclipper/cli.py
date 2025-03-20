"""
Command-line interface for VideoClipper.
"""

import os
import time
import click
from rich.console import Console
from rich.progress import Progress

from videoclipper.clipper.video_editor import VideoEditor
from videoclipper.clipper.segment_selector import SegmentSelector
from videoclipper.analyzer.scene_detector import SceneDetector
from videoclipper.exceptions import FileError, VideoClipperError
from videoclipper.models.segment import Segment, SegmentType
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
    default=True,  # Changed to True for better captions
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
@click.option(
    "--captions/--no-captions",
    default=True,
    help="Add colorful captions to the clips (requires transcription).",
)
@click.option(
    "--highlight-words",
    type=str,
    help="Comma-separated list of words to highlight in captions.",
)
def process(
    video_input, output_dir, duration, transcribe, whisper_model, min_segment, max_segment, num_clips,
    captions, highlight_words
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
            
            # Step 2: Analyze video for scene changes and transcribe if enabled
            task2 = progress.add_task("[cyan]Analyzing content...", total=100)
            scene_detector = SceneDetector(video_path)
            scene_segments = []
            speech_segments = []
            
            # Process transcription if enabled (needed for captions)
            if transcribe or captions:
                try:
                    from videoclipper.transcriber.whisper_transcriber import WhisperTranscriber
                    import os
                    
                    # Check if the video file exists
                    if not os.path.exists(video_path):
                        console.print(f"[yellow]Video file not found: {video_path}[/yellow]")
                        raise FileNotFoundError(f"Video file not found: {video_path}")
                    
                    # Check if ffmpeg is installed for audio extraction
                    import subprocess
                    try:
                        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                    except (subprocess.SubprocessError, FileNotFoundError):
                        console.print("[yellow]ffmpeg not found, which is required for transcription[/yellow]")
                        raise RuntimeError("ffmpeg not found, which is required for transcription")
                    
                    console.print(f"[cyan]Transcribing audio with Whisper ({whisper_model} model)...[/cyan]")
                    
                    # Extract audio to a temporary file
                    temp_audio = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
                    extract_cmd = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", temp_audio, "-y"]
                    
                    console.print(f"[cyan]Extracting audio to {temp_audio}...[/cyan]")
                    subprocess.run(extract_cmd, capture_output=True, check=True)
                    
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
                    import traceback
                    console.print(f"[yellow]Transcription failed: {e}[/yellow]")
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
                    console.print("[yellow]Continuing without captions[/yellow]")
            
            # Detect scene changes
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
            
            # Process segments using the improved segment selector
            selector = SegmentSelector(
                video_duration=video_duration,
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
                
                # Process highlight words if provided
                highlight_keywords = None
                if highlight_words:
                    highlight_keywords = [word.strip() for word in highlight_words.split(',')]
                
                try:
                    # Create the highlight clip with viral style and captions
                    output_path, clip_duration = video_editor.create_clip(
                        selected_segments, 
                        clip_path,
                        max_duration=max_segment,
                        viral_style=True,
                        add_captions=captions,
                        highlight_keywords=highlight_keywords
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
                        # Create the highlight clip with viral style and captions
                        output_path, clip_duration = video_editor.create_clip(
                            selected_segments, 
                            clip_path,
                            max_duration=max_segment,
                            viral_style=True,
                            add_captions=captions,
                            highlight_keywords=highlight_keywords
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