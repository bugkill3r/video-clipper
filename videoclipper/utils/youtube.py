"""YouTube download utilities for VideoClipper."""

import os
import re
import subprocess
from typing import Dict, Optional

from videoclipper.exceptions import FileError


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


def download_youtube_video(url: str, output_dir: str) -> Dict[str, str]:
    """Download a YouTube video using yt-dlp.
    
    Args:
        url: YouTube URL
        output_dir: Directory to save the video
        
    Returns:
        Dictionary with video info
        
    Raises:
        FileError: If download fails
    """
    if not is_youtube_url(url):
        raise ValueError(f"Not a valid YouTube URL: {url}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    video_id = get_video_id(url)
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
    
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
        
        # Now download the video
        download_cmd = [
            "yt-dlp",
            "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", output_template,
            url
        ]
        
        subprocess.run(download_cmd, check=True)
        
        # Find the downloaded file
        video_files = [f for f in os.listdir(output_dir) if f.startswith(safe_filename)]
        if not video_files:
            raise FileError(f"Download completed but couldn't find video file in {output_dir}")
        
        video_path = os.path.join(output_dir, video_files[0])
        
        return {
            "id": video_id,
            "title": title,
            "path": video_path,
            "duration": duration
        }
    
    except subprocess.CalledProcessError as e:
        raise FileError(f"Failed to download YouTube video: {e.stderr}")
    except Exception as e:
        raise FileError(f"Error downloading YouTube video: {str(e)}")