"""YouTube download utilities for VideoClipper."""

import os
import re
import subprocess
import datetime
from typing import Dict, List, Optional, Tuple

from videoclipper.exceptions import FileError
from videoclipper.models.segment import Segment, SegmentType


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a valid YouTube URL.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is a valid YouTube URL
    """
    youtube_regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([^\s&]+)"
    return bool(re.match(youtube_regex, url))


def get_video_id(url: str) -> str:
    """Extract video ID from a YouTube URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        YouTube video ID
        
    Raises:
        ValueError: If URL is not a valid YouTube URL
    """
    if "youtube.com/watch?v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    return video_id


def get_safe_filename(video_id: str, title: Optional[str] = None) -> str:
    """Generate a safe filename from a video title and ID.
    
    Args:
        video_id: YouTube video ID
        title: Optional video title
        
    Returns:
        Safe filename
    """
    if title:
        # Remove special characters and limit length
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().lower()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        safe_title = safe_title[:50]  # Limit length
        return f"{safe_title}-{video_id}"
    else:
        return video_id


def parse_srt_time(time_str: str) -> float:
    """Convert SRT timestamp format (HH:MM:SS,mmm) to seconds.
    
    Args:
        time_str: SRT format timestamp (e.g., "00:01:23,456")
        
    Returns:
        Time in seconds
    """
    hours, minutes, seconds = time_str.replace(',', '.').split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_srt_file(srt_path: str) -> List[Segment]:
    """Parse an SRT subtitle file and convert to segments.
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        List of segments with text and timing
    """
    if not os.path.exists(srt_path):
        return []
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try different encodings if UTF-8 fails
        try:
            with open(srt_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading subtitles file: {e}")
            return []
    
    # Extract all subtitle entries with precise timing
    subtitle_entries = []
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\d+\n|$)'
    
    for match in re.finditer(pattern, content):
        idx, start_time, end_time, text = match.groups()
        # Clean text (remove extra newlines and whitespace)
        text = re.sub(r'\n+', ' ', text).strip()
        
        # Skip empty text
        if not text:
            continue
        
        # Convert times to seconds
        start_sec = parse_srt_time(start_time)
        end_sec = parse_srt_time(end_time)
        
        # Skip entries with very short duration
        if end_sec - start_sec < 0.1:
            continue
            
        # Store entry with exact timing
        subtitle_entries.append({
            'index': int(idx) if idx.isdigit() else 0,
            'start': start_sec,
            'end': end_sec,
            'text': text,
            'duration': end_sec - start_sec
        })
    
    # Create segments from subtitle entries
    segments = []
    
    # Process each subtitle entry as an individual segment with exact timing
    for entry in subtitle_entries:
        # Create a metadata dictionary with precise timing information
        metadata = {
            "subtitle_timing": True,
            "exact_start": entry['start'],
            "exact_end": entry['end'],
            "subtitle_index": entry['index'],
            "duration": entry['duration'],
            "original_text": entry['text'],  # Store original text for reference
            "is_subtitle_entry": True  # Flag to identify subtitle-based segments
        }
        
        # Create a segment with precise timing information
        segment = Segment(
            start=entry['start'],
            end=entry['end'],
            score=0.7,  # Decent default score for subtitle segments
            segment_type=SegmentType.SPEECH,
            text=entry['text'],
            metadata=metadata  # Include subtitle timing data
        )
        segments.append(segment)
    
    return segments


def download_youtube_video(url: str, output_dir: str, download_captions: bool = True) -> Dict[str, any]:
    """Download a YouTube video and optionally its captions using yt-dlp.
    
    Args:
        url: YouTube URL
        output_dir: Directory to save the video
        download_captions: Whether to download captions if available
        
    Returns:
        Dictionary with video info including subtitle path if available
        
    Raises:
        FileError: If download fails
    """
    if not is_youtube_url(url):
        raise ValueError(f"Not a valid YouTube URL: {url}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    video_id = get_video_id(url)
    
    try:
        # First get video info
        info_cmd = [
            "yt-dlp", 
            "--print", "title",
            "--print", "duration", 
            "--skip-download",
            url
        ]
        
        process = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
        lines = process.stdout.strip().split('\n')
        if len(lines) >= 2:
            title, duration = lines[0], lines[1]
        else:
            title, duration = f"video-{video_id}", "0"
        
        # Generate a safe filename
        safe_filename = get_safe_filename(video_id, title)
        output_template = os.path.join(output_dir, f"{safe_filename}.%(ext)s")
        
        # Build download command
        download_cmd = [
            "yt-dlp",
            "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", output_template
        ]
        
        # Add subtitle/caption downloading if requested
        subtitles_path = None
        if download_captions:
            # Add automatic subtitle download options
            download_cmd.extend([
                "--write-auto-sub",          # Download auto-generated subtitles
                "--sub-lang", "en",          # Prefer English subtitles
                "--convert-subs", "srt",     # Convert to SRT format (easier to parse)
                "--write-sub"                # Write available subtitles
            ])
        
        # Add the URL at the end
        download_cmd.append(url)
        
        # Execute the download command
        subprocess.run(download_cmd, check=True)
        
        # Find the downloaded video file
        video_files = [f for f in os.listdir(output_dir) if f.startswith(safe_filename) and f.endswith((".mp4", ".mkv", ".webm"))]
        if not video_files:
            raise FileError(f"Download completed but couldn't find video file in {output_dir}")
        
        video_path = os.path.join(output_dir, video_files[0])
        
        # Find subtitle file if it exists
        if download_captions:
            subtitle_files = [f for f in os.listdir(output_dir) 
                             if f.startswith(safe_filename) and f.endswith((".srt", ".vtt"))]
            if subtitle_files:
                subtitles_path = os.path.join(output_dir, subtitle_files[0])
                print(f"Found subtitles: {subtitles_path}")
        
        return {
            "id": video_id,
            "title": title,
            "path": video_path,
            "duration": duration,
            "subtitles_path": subtitles_path
        }
    
    except subprocess.CalledProcessError as e:
        raise FileError(f"Failed to download YouTube video: {e.stderr}")
    except Exception as e:
        raise FileError(f"Error downloading YouTube video: {str(e)}")