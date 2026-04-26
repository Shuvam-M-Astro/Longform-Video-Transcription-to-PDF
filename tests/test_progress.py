"""
Tests for progress tracking functionality.
"""

import io
import time

import pytest
from src.video_doc.progress import (
    PipelineProgress,
    _format_duration,
    make_console_progress_printer,
)
from src.video_doc.frames import list_keyframe_files


class TestProgress:
    """Test progress tracking functionality."""

    def test_progress_initialization(self):
        """Test progress initialization."""
        progress = PipelineProgress(total_weight=100.0)
        assert progress.total_weight == 100.0
        assert progress.completed_weight == 0.0
        assert progress.current is None
        assert progress.overall_percent() == 0.0
        assert progress.label() == "Idle"

    def test_step_progression(self):
        """Test step-by-step progress."""
        progress = PipelineProgress(total_weight=100.0)

        # Start first step (weight 50)
        progress.start_step("Step 1", 50.0)
        assert progress.label() == "Step 1"
        assert progress.overall_percent() == 0.0

        # Update progress within step
        progress.update(50.0)  # 50% of step 1
        assert progress.overall_percent() == 25.0  # 50% of 50 weight = 25 overall

        # Complete first step
        progress.end_step()
        assert progress.completed_weight == 50.0
        assert progress.overall_percent() == 50.0

        # Start second step (weight 50)
        progress.start_step("Step 2", 50.0)
        assert progress.label() == "Step 2"

        # Complete second step
        progress.update(100.0)
        progress.end_step()
        assert progress.completed_weight == 100.0
        assert progress.overall_percent() == 100.0

    def test_multiple_steps(self):
        """Test multiple steps with different weights."""
        progress = PipelineProgress(total_weight=200.0)

        # Step 1: 20% of total
        progress.start_step("Download", 40.0)
        progress.update(100.0)
        progress.end_step()
        assert progress.overall_percent() == 20.0

        # Step 2: 30% of total
        progress.start_step("Process", 60.0)
        progress.update(100.0)
        progress.end_step()
        assert progress.overall_percent() == 50.0

        # Step 3: 50% of total
        progress.start_step("Generate", 100.0)
        progress.update(100.0)
        progress.end_step()
        assert progress.overall_percent() == 100.0

    def test_progress_callback(self):
        """Test progress callback functionality."""
        callback_calls = []

        def mock_callback(percent, label):
            callback_calls.append((percent, label))

        progress = PipelineProgress(total_weight=100.0, on_change=mock_callback)

        # Start step
        progress.start_step("Test Step", 50.0)
        assert len(callback_calls) == 1
        assert callback_calls[-1] == (0.0, "Test Step")

        # Update progress
        progress.update(75.0)
        assert len(callback_calls) == 2
        assert callback_calls[-1] == (37.5, "Test Step")

        # End step
        progress.end_step()
        assert len(callback_calls) == 3
        assert callback_calls[-1] == (50.0, "Idle")

    def test_console_progress_printer(self):
        """Test console progress printer creation."""
        printer = make_console_progress_printer()
        assert callable(printer)

        # Test that it doesn't raise errors
        try:
            printer(50.0, "Test Label")
            printer(100.0, "Complete")
        except Exception as e:
            pytest.fail(f"Console printer raised an exception: {e}")

    def test_list_keyframe_files_uses_requested_format(self, tmp_path):
        """Resume logic should reuse the configured frame format, not just JPG."""
        frames_dir = tmp_path / "keyframes"
        frames_dir.mkdir()
        png_frame = frames_dir / "frame_000001.png"
        jpg_frame = frames_dir / "frame_000001.jpg"
        webp_frame = frames_dir / "frame_000001.webp"
        png_frame.write_bytes(b"png")
        jpg_frame.write_bytes(b"jpg")
        webp_frame.write_bytes(b"webp")

        assert list_keyframe_files(frames_dir, "png") == [png_frame]
        assert list_keyframe_files(frames_dir, "jpeg") == [jpg_frame]
        assert list_keyframe_files(frames_dir) == [jpg_frame, png_frame, webp_frame]


class TestProgressTiming:
    """Elapsed/ETA helpers and TTY-aware printer."""

    def test_elapsed_is_non_negative_and_monotonic(self):
        progress = PipelineProgress()
        first = progress.elapsed_seconds()
        assert first >= 0.0
        time.sleep(0.01)
        assert progress.elapsed_seconds() >= first

    def test_eta_is_none_until_progress_recorded(self):
        progress = PipelineProgress()
        assert progress.eta_seconds() is None
        progress.start_step("Step", 50.0)
        # No progress within step yet -> still None.
        assert progress.eta_seconds() is None

    def test_eta_returns_positive_value_after_progress(self):
        progress = PipelineProgress()
        progress.start_step("Step", 100.0)
        time.sleep(0.02)
        progress.update(50.0)
        eta = progress.eta_seconds()
        assert eta is not None
        assert eta > 0.0

    def test_eta_is_none_when_complete(self):
        progress = PipelineProgress()
        progress.start_step("Step", 100.0)
        progress.update(100.0)
        progress.end_step()
        assert progress.eta_seconds() is None

    def test_format_duration_handles_none_and_hours(self):
        assert _format_duration(None) == "--:--"
        assert _format_duration(0) == "00:00"
        assert _format_duration(65) == "01:05"
        assert _format_duration(3725) == "1:02:05"

    def test_non_tty_printer_writes_one_line_per_call(self):
        buf = io.StringIO()
        # StringIO has no isatty -> treated as non-TTY.
        printer = make_console_progress_printer(stream=buf)
        printer(10.0, "Step A")
        printer(20.0, "Step B")
        lines = [ln for ln in buf.getvalue().splitlines() if ln]
        assert len(lines) == 2
        assert "10.00%" in lines[0]
        assert "Step B" in lines[1]

    def test_printer_includes_elapsed_when_progress_provided(self):
        buf = io.StringIO()
        progress = PipelineProgress()
        printer = make_console_progress_printer(stream=buf, progress=progress)
        printer(0.0, "Idle")
        out = buf.getvalue()
        assert "elapsed" in out
        assert "ETA" in out