# VideoClipper Usage Guide

## Command Line Interface

VideoClipper provides a robust command-line interface for processing videos and generating highlights.

### Basic Usage

```bash
videoclipper process VIDEO_INPUT [OPTIONS]
```

Where `VIDEO_INPUT` can be either:
- A path to a local video file
- A YouTube URL

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir`, `-o` | Output directory for highlight clips | `output/` |
| `--duration`, `-d` | Target duration of highlight clips in seconds | `30` |
| `--transcribe/--no-transcribe` | Enable/disable transcription | `True` |
| `--whisper-model` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) | `base` |
| `--min-segment` | Minimum segment duration in seconds | `5` |
| `--max-segment` | Maximum segment duration in seconds | `45` |
| `--num-clips`, `-n` | Number of highlight clips to generate | `3` |
| `--captions/--no-captions` | Enable/disable captions | `False` |
| `--highlight-words` | Comma-separated list of words to highlight in captions | Auto-detected |

### Examples

#### Process a Local Video

Basic processing:
```bash
videoclipper process path/to/video.mp4
```

With full options:
```bash
videoclipper process path/to/video.mp4 \
  --output-dir highlights/ \
  --duration 60 \
  --transcribe \
  --whisper-model base \
  --min-segment 3 \
  --max-segment 15 \
  --num-clips 5 \
  --captions \
  --highlight-words "important,keywords,action,excitement"
```

#### Process a YouTube Video

Basic processing:
```bash
videoclipper process https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

With options:
```bash
videoclipper process https://www.youtube.com/watch?v=dQw4w9WgXcQ \
  --output-dir youtube_highlights/ \
  --duration 45 \
  --num-clips 2 \
  --captions \
  --highlight-words "never,gonna,give,you,up"
```

## Advanced Features

### Caption Highlighting

VideoClipper can add professional captions with word highlighting to your clips:

1. Enable captions with the `--captions` flag
2. Specify words to highlight with `--highlight-words` option (comma-separated)
3. If no highlight words are specified, important words are automatically detected

The caption system:
- Splits text into readable 4-6 word chunks
- Shows one chunk at a time, synchronized with speech
- Highlights keywords in alternating yellow/green
- Places captions on a semi-transparent dark background
- Positions captions at the bottom of the screen for optimal readability

Example with highlighting specific words:
```bash
videoclipper process video.mp4 --captions --highlight-words "highlight,these,words"
```

### YouTube Integration

VideoClipper can directly download and process YouTube videos:

1. Simply provide a YouTube URL instead of a local file path
2. The tool will automatically download the video in the best available quality
3. If available, YouTube captions will be used (enhancing the transcription accuracy)
4. Downloaded videos are saved in the `downloads/` directory

Example:
```bash
videoclipper process https://www.youtube.com/watch?v=dQw4w9WgXcQ --captions
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Make sure FFmpeg is installed and in your PATH
2. **YouTube download fails**: Check your internet connection and verify the URL is valid
3. **No captions displayed**: Ensure you've enabled captions with the `--captions` flag
4. **Transcription is slow**: Try using a smaller whisper model (e.g., `--whisper-model tiny`)

### Dependencies

- Python 3.8+
- FFmpeg
- MoviePy 2.0.0+
- yt-dlp (for YouTube downloads)
- OpenAI Whisper (for transcription)