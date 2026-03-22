"""
Tests for data validation functionality.
"""

import pytest
from pydantic import ValidationError

from src.video_doc.data_validation import VideoConfig, ValidationResult, ValidationStatus


class TestDataValidation:
    """Test data validation functionality."""

    def test_valid_video_config(self):
        """Test valid video configuration."""
        config = VideoConfig(
            language="en",
            whisper_model="medium",
            beam_size=5,
            transcribe_only=False,
            kf_method="scene",
            max_fps=1.0,
            min_scene_diff=0.45,
            report_style="book"
        )
        assert config.language == "en"
        assert config.whisper_model == "medium"
        assert config.beam_size == 5

    def test_invalid_language(self):
        """Test invalid language code."""
        with pytest.raises(ValidationError):
            VideoConfig(language="invalid")

    def test_invalid_whisper_model(self):
        """Test invalid whisper model."""
        with pytest.raises(ValidationError):
            VideoConfig(whisper_model="invalid_model")

    def test_invalid_beam_size(self):
        """Test invalid beam size."""
        with pytest.raises(ValidationError):
            VideoConfig(beam_size=0)  # Too low

        with pytest.raises(ValidationError):
            VideoConfig(beam_size=20)  # Too high

    def test_invalid_kf_method(self):
        """Test invalid keyframe method."""
        with pytest.raises(ValidationError):
            VideoConfig(kf_method="invalid")

    def test_invalid_max_fps(self):
        """Test invalid max FPS."""
        with pytest.raises(ValidationError):
            VideoConfig(max_fps=-1.0)  # Negative

        with pytest.raises(ValidationError):
            VideoConfig(max_fps=20.0)  # Too high

    def test_invalid_scene_diff(self):
        """Test invalid scene difference."""
        with pytest.raises(ValidationError):
            VideoConfig(min_scene_diff=-0.1)  # Negative

        with pytest.raises(ValidationError):
            VideoConfig(min_scene_diff=2.0)  # Too high

    def test_invalid_report_style(self):
        """Test invalid report style."""
        with pytest.raises(ValidationError):
            VideoConfig(report_style="invalid")

    def test_validation_result_creation(self):
        """Test creating validation results."""
        result = ValidationResult(
            check_name="config_validation",
            status=ValidationStatus.PASSED,
            severity="info",
            message="Configuration is valid"
        )
        assert result.check_name == "config_validation"
        assert result.status == ValidationStatus.PASSED
        assert result.severity == "info"
        assert result.message == "Configuration is valid"
        assert result.timestamp is not None

    def test_validation_result_with_metadata(self):
        """Test validation result with metadata."""
        result = ValidationResult(
            check_name="file_check",
            status=ValidationStatus.FAILED,
            severity="error",
            message="File not found",
            expected_value="/expected/path",
            actual_value=None,
            metadata={"error_code": "ENOENT"}
        )
        assert result.expected_value == "/expected/path"
        assert result.actual_value is None
        assert result.metadata["error_code"] == "ENOENT"