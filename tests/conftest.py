"""
Test configuration and fixtures.
"""

import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_video_config():
    """Sample video processing configuration."""
    return {
        "language": "auto",
        "whisper_model": "tiny",
        "beam_size": 5,
        "transcribe_only": False,
        "streaming": False,
        "kf_method": "scene",
        "max_fps": 1.0,
        "min_scene_diff": 0.45,
        "report_style": "book"
    }


@pytest.fixture
def sample_transcript_segments():
    """Sample transcript segments for testing."""
    return [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Hello, this is a test video."
        },
        {
            "start": 5.0,
            "end": 10.0,
            "text": "It demonstrates the transcription functionality."
        },
        {
            "start": 10.0,
            "end": 15.0,
            "text": "This is the end of the test."
        }
    ]