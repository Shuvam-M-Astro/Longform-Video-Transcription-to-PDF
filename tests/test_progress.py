"""
Tests for progress tracking functionality.
"""

import pytest
from src.video_doc.progress import PipelineProgress, make_console_progress_printer


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