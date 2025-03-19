# VideoClipper

An AI-powered video highlight generator that automatically extracts the most interesting segments from videos.

## Features

- Automated video highlight generation
- Scene detection and analysis
- Audio energy analysis for exciting moment detection
- Speech transcription with OpenAI's Whisper model
- Intelligent clip selection and assembly
- YouTube video support (download and process)

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (required for video processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/videoclipper.git
cd videoclipper
```

2. Install dependencies:
```bash
pip install -e .
```

Or for development:
```bash
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

#### Processing Local Videos

Basic usage:

```bash
videoclipper process path/to/video.mp4 --output-dir highlights/
```

With advanced options:

```bash
videoclipper process path/to/video.mp4 \
  --output-dir highlights/ \
  --duration 60 \
  --transcribe \
  --whisper-model base \
  --min-segment 3 \
  --max-segment 15 \
  --num-clips 5
```

#### Processing YouTube Videos

You can directly process YouTube videos by providing a URL:

```bash
videoclipper process https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

With options:

```bash
videoclipper process https://www.youtube.com/watch?v=dQw4w9WgXcQ \
  --output-dir custom_highlights/ \
  --num-clips 3 \
  --duration 45
```

### Python API

```python
from videoclipper import VideoProcessor

# Initialize the processor
processor = VideoProcessor("path/to/video.mp4")

# Run analysis
processor.analyze(transcribe=True, detect_scenes=True)

# Generate highlight clip
processor.create_highlight("output.mp4", max_duration=60)
```

## Development

### Setting up a development environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running tests

```bash
pytest
```

Or with coverage:

```bash
pytest --cov=videoclipper
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
