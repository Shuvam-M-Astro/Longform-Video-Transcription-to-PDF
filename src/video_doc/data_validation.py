"""
Data validation and quality checks framework for video processing pipeline.
"""

import os
import json
import hashlib
import mimetypes
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image
import pydantic
from pydantic import BaseModel, Field, validator
import structlog

logger = structlog.get_logger(__name__)


class ValidationSeverity(str, Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """Validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    threshold: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


# Pydantic models for data validation
class VideoConfig(BaseModel):
    """Video processing configuration validation."""
    url: Optional[str] = None
    video_path: Optional[str] = None
    language: str = Field(default="auto", regex=r"^[a-z]{2}$|^auto$")
    whisper_model: str = Field(default="medium", regex=r"^(tiny|base|small|medium|large|large-v2|large-v3)$")
    beam_size: int = Field(default=5, ge=1, le=10)
    transcribe_only: bool = False
    streaming: bool = False
    kf_method: str = Field(default="scene", regex=r"^(scene|iframe|interval)$")
    max_fps: float = Field(default=1.0, gt=0, le=10.0)
    min_scene_diff: float = Field(default=0.45, ge=0.0, le=1.0)
    report_style: str = Field(default="book", regex=r"^(minimal|book)$")
    
    @validator('url')
    def validate_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @validator('video_path')
    def validate_video_path(cls, v):
        if v and not Path(v).exists():
            raise ValueError('Video file does not exist')
        return v


class AudioQualityMetrics(BaseModel):
    """Audio quality metrics."""
    duration: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    bit_depth: int = Field(gt=0)
    file_size: int = Field(gt=0)
    format: str
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        if v not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError('Unsupported sample rate')
        return v


class VideoQualityMetrics(BaseModel):
    """Video quality metrics."""
    duration: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    bitrate: int = Field(gt=0)
    file_size: int = Field(gt=0)
    format: str
    codec: str
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v < 32 or v > 7680:  # 8K resolution limit
            raise ValueError('Invalid video dimensions')
        return v


class TranscriptQualityMetrics(BaseModel):
    """Transcript quality metrics."""
    total_segments: int = Field(ge=0)
    total_duration: float = Field(gt=0)
    average_segment_length: float = Field(gt=0)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    language_detected: Optional[str] = None
    word_count: int = Field(ge=0)
    character_count: int = Field(ge=0)


# Base validation classes
class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.severity = severity
        self.logger = structlog.get_logger(__name__)
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data and return result."""
        raise NotImplementedError
    
    def _create_result(
        self,
        status: ValidationStatus,
        message: str,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Create validation result."""
        return ValidationResult(
            check_name=self.name,
            status=status,
            severity=self.severity,
            message=message,
            expected_value=expected_value,
            actual_value=actual_value,
            threshold=threshold,
            metadata=metadata
        )


class FileValidator(BaseValidator):
    """Validator for file-related checks."""
    
    def __init__(self, name: str = "file_validator"):
        super().__init__(name)
    
    def validate(self, file_path: Union[str, Path], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate file properties."""
        results = []
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            results.append(self._create_result(
                ValidationStatus.FAILED,
                f"File does not exist: {file_path}",
                metadata={"file_path": str(file_path)}
            ))
            return results
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            results.append(self._create_result(
                ValidationStatus.FAILED,
                "File is empty",
                actual_value=file_size,
                metadata={"file_path": str(file_path)}
            ))
        
        # Check file permissions
        if not os.access(file_path, os.R_OK):
            results.append(self._create_result(
                ValidationStatus.FAILED,
                "File is not readable",
                metadata={"file_path": str(file_path)}
            ))
        
        # Check file extension
        allowed_extensions = context.get('allowed_extensions', []) if context else []
        if allowed_extensions and file_path.suffix.lower() not in allowed_extensions:
            results.append(self._create_result(
                ValidationStatus.FAILED,
                f"File extension not allowed: {file_path.suffix}",
                expected_value=allowed_extensions,
                actual_value=file_path.suffix,
                metadata={"file_path": str(file_path)}
            ))
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        expected_mime_types = context.get('expected_mime_types', []) if context else []
        if expected_mime_types and mime_type not in expected_mime_types:
            results.append(self._create_result(
                ValidationStatus.WARNING,
                f"MIME type mismatch: {mime_type}",
                expected_value=expected_mime_types,
                actual_value=mime_type,
                metadata={"file_path": str(file_path)}
            ))
        
        return results


class AudioValidator(BaseValidator):
    """Validator for audio file quality."""
    
    def __init__(self, name: str = "audio_validator"):
        super().__init__(name)
    
    def validate(self, audio_path: Union[str, Path], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate audio file quality."""
        results = []
        audio_path = Path(audio_path)
        
        try:
            # Use OpenCV to get audio properties
            cap = cv2.VideoCapture(str(audio_path))
            
            if not cap.isOpened():
                results.append(self._create_result(
                    ValidationStatus.FAILED,
                    "Cannot open audio file",
                    metadata={"file_path": str(audio_path)}
                ))
                return results
            
            # Get audio properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            
            # Validate duration
            min_duration = context.get('min_duration', 1.0) if context else 1.0
            max_duration = context.get('max_duration', 3600.0) if context else 3600.0
            
            if duration < min_duration:
                results.append(self._create_result(
                    ValidationStatus.FAILED,
                    f"Audio too short: {duration:.2f}s",
                    expected_value=f">= {min_duration}s",
                    actual_value=f"{duration:.2f}s",
                    metadata={"file_path": str(audio_path)}
                ))
            elif duration > max_duration:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Audio very long: {duration:.2f}s",
                    expected_value=f"<= {max_duration}s",
                    actual_value=f"{duration:.2f}s",
                    metadata={"file_path": str(audio_path)}
                ))
            
            # Check for silence
            silence_threshold = context.get('silence_threshold', 0.01) if context else 0.01
            if self._detect_silence(audio_path, silence_threshold):
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    "Audio appears to be mostly silent",
                    threshold=silence_threshold,
                    metadata={"file_path": str(audio_path)}
                ))
            
            cap.release()
            
        except Exception as e:
            results.append(self._create_result(
                ValidationStatus.FAILED,
                f"Error validating audio: {str(e)}",
                metadata={"file_path": str(audio_path)}
            ))
        
        return results
    
    def _detect_silence(self, audio_path: Path, threshold: float) -> bool:
        """Detect if audio is mostly silent."""
        try:
            # Simple silence detection using OpenCV
            cap = cv2.VideoCapture(str(audio_path))
            if not cap.isOpened():
                return True
            
            # Sample frames for analysis
            sample_frames = 100
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, frame_count // sample_frames)
            
            silent_frames = 0
            total_frames = 0
            
            for i in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # Convert to grayscale and calculate mean
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_intensity = np.mean(gray)
                    if mean_intensity < threshold * 255:
                        silent_frames += 1
                    total_frames += 1
            
            cap.release()
            return total_frames > 0 and (silent_frames / total_frames) > 0.8
            
        except Exception:
            return False


class VideoValidator(BaseValidator):
    """Validator for video file quality."""
    
    def __init__(self, name: str = "video_validator"):
        super().__init__(name)
    
    def validate(self, video_path: Union[str, Path], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate video file quality."""
        results = []
        video_path = Path(video_path)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                results.append(self._create_result(
                    ValidationStatus.FAILED,
                    "Cannot open video file",
                    metadata={"file_path": str(video_path)}
                ))
                return results
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Validate dimensions
            min_width = context.get('min_width', 320) if context else 320
            min_height = context.get('min_height', 240) if context else 240
            max_width = context.get('max_width', 7680) if context else 7680
            max_height = context.get('max_height', 4320) if context else 4320
            
            if width < min_width or height < min_height:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Video resolution too low: {width}x{height}",
                    expected_value=f">= {min_width}x{min_height}",
                    actual_value=f"{width}x{height}",
                    metadata={"file_path": str(video_path)}
                ))
            elif width > max_width or height > max_height:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Video resolution very high: {width}x{height}",
                    expected_value=f"<= {max_width}x{max_height}",
                    actual_value=f"{width}x{height}",
                    metadata={"file_path": str(video_path)}
                ))
            
            # Validate FPS
            min_fps = context.get('min_fps', 1.0) if context else 1.0
            max_fps = context.get('max_fps', 120.0) if context else 120.0
            
            if fps < min_fps:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Video FPS too low: {fps:.2f}",
                    expected_value=f">= {min_fps}",
                    actual_value=f"{fps:.2f}",
                    metadata={"file_path": str(video_path)}
                ))
            elif fps > max_fps:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Video FPS very high: {fps:.2f}",
                    expected_value=f"<= {max_fps}",
                    actual_value=f"{fps:.2f}",
                    metadata={"file_path": str(video_path)}
                ))
            
            # Validate duration
            min_duration = context.get('min_duration', 1.0) if context else 1.0
            max_duration = context.get('max_duration', 7200.0) if context else 7200.0
            
            if duration < min_duration:
                results.append(self._create_result(
                    ValidationStatus.FAILED,
                    f"Video too short: {duration:.2f}s",
                    expected_value=f">= {min_duration}s",
                    actual_value=f"{duration:.2f}s",
                    metadata={"file_path": str(video_path)}
                ))
            elif duration > max_duration:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Video very long: {duration:.2f}s",
                    expected_value=f"<= {max_duration}s",
                    actual_value=f"{duration:.2f}s",
                    metadata={"file_path": str(video_path)}
                ))
            
            cap.release()
            
        except Exception as e:
            results.append(self._create_result(
                ValidationStatus.FAILED,
                f"Error validating video: {str(e)}",
                metadata={"file_path": str(video_path)}
            ))
        
        return results


class TranscriptValidator(BaseValidator):
    """Validator for transcript quality."""
    
    def __init__(self, name: str = "transcript_validator"):
        super().__init__(name)
    
    def validate(self, transcript_data: Union[Dict, List], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate transcript quality."""
        results = []
        
        try:
            if isinstance(transcript_data, list):
                segments = transcript_data
            elif isinstance(transcript_data, dict) and 'segments' in transcript_data:
                segments = transcript_data['segments']
            else:
                results.append(self._create_result(
                    ValidationStatus.FAILED,
                    "Invalid transcript data format",
                    metadata={"data_type": type(transcript_data).__name__}
                ))
                return results
            
            if not segments:
                results.append(self._create_result(
                    ValidationStatus.FAILED,
                    "Transcript is empty",
                    metadata={"segment_count": 0}
                ))
                return results
            
            # Validate segment count
            min_segments = context.get('min_segments', 1) if context else 1
            max_segments = context.get('max_segments', 10000) if context else 10000
            
            if len(segments) < min_segments:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Too few segments: {len(segments)}",
                    expected_value=f">= {min_segments}",
                    actual_value=len(segments),
                    metadata={"segment_count": len(segments)}
                ))
            elif len(segments) > max_segments:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Very many segments: {len(segments)}",
                    expected_value=f"<= {max_segments}",
                    actual_value=len(segments),
                    metadata={"segment_count": len(segments)}
                ))
            
            # Validate individual segments
            total_text_length = 0
            empty_segments = 0
            invalid_timestamps = 0
            
            for i, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    results.append(self._create_result(
                        ValidationStatus.FAILED,
                        f"Invalid segment format at index {i}",
                        metadata={"segment_index": i}
                    ))
                    continue
                
                # Check required fields
                required_fields = ['start', 'end', 'text']
                missing_fields = [field for field in required_fields if field not in segment]
                if missing_fields:
                    results.append(self._create_result(
                        ValidationStatus.FAILED,
                        f"Missing fields in segment {i}: {missing_fields}",
                        expected_value=required_fields,
                        actual_value=list(segment.keys()),
                        metadata={"segment_index": i}
                    ))
                    continue
                
                # Validate timestamps
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                
                if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
                    invalid_timestamps += 1
                elif start_time < 0 or end_time < 0 or start_time >= end_time:
                    invalid_timestamps += 1
                
                # Validate text
                text = segment.get('text', '')
                if not text or not text.strip():
                    empty_segments += 1
                else:
                    total_text_length += len(text.strip())
            
            # Check for empty segments
            empty_threshold = context.get('empty_segment_threshold', 0.1) if context else 0.1
            if empty_segments > 0:
                empty_ratio = empty_segments / len(segments)
                if empty_ratio > empty_threshold:
                    results.append(self._create_result(
                        ValidationStatus.WARNING,
                        f"High ratio of empty segments: {empty_ratio:.2%}",
                        expected_value=f"<= {empty_threshold:.2%}",
                        actual_value=f"{empty_ratio:.2%}",
                        threshold=empty_threshold,
                        metadata={"empty_segments": empty_segments, "total_segments": len(segments)}
                    ))
            
            # Check for invalid timestamps
            if invalid_timestamps > 0:
                results.append(self._create_result(
                    ValidationStatus.FAILED,
                    f"Invalid timestamps in {invalid_timestamps} segments",
                    actual_value=invalid_timestamps,
                    metadata={"invalid_timestamps": invalid_timestamps, "total_segments": len(segments)}
                ))
            
            # Check total text length
            min_text_length = context.get('min_text_length', 10) if context else 10
            if total_text_length < min_text_length:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Very short transcript: {total_text_length} characters",
                    expected_value=f">= {min_text_length}",
                    actual_value=total_text_length,
                    metadata={"text_length": total_text_length}
                ))
            
        except Exception as e:
            results.append(self._create_result(
                ValidationStatus.FAILED,
                f"Error validating transcript: {str(e)}",
                metadata={"error": str(e)}
            ))
        
        return results


class ImageValidator(BaseValidator):
    """Validator for image quality."""
    
    def __init__(self, name: str = "image_validator"):
        super().__init__(name)
    
    def validate(self, image_path: Union[str, Path], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate image quality."""
        results = []
        image_path = Path(image_path)
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Get image properties
            width, height = image.size
            format_name = image.format
            mode = image.mode
            
            # Validate dimensions
            min_width = context.get('min_width', 32) if context else 32
            min_height = context.get('min_height', 32) if context else 32
            max_width = context.get('max_width', 8192) if context else 8192
            max_height = context.get('max_height', 8192) if context else 8192
            
            if width < min_width or height < min_height:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Image too small: {width}x{height}",
                    expected_value=f">= {min_width}x{min_height}",
                    actual_value=f"{width}x{height}",
                    metadata={"file_path": str(image_path)}
                ))
            elif width > max_width or height > max_height:
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    f"Image very large: {width}x{height}",
                    expected_value=f"<= {max_width}x{max_height}",
                    actual_value=f"{width}x{height}",
                    metadata={"file_path": str(image_path)}
                ))
            
            # Check for corruption
            try:
                image.verify()
            except Exception:
                results.append(self._create_result(
                    ValidationStatus.FAILED,
                    "Image file is corrupted",
                    metadata={"file_path": str(image_path)}
                ))
            
            # Check for blurriness (using OpenCV if available)
            blur_threshold = context.get('blur_threshold', 100.0) if context else 100.0
            if self._is_blurry(image_path, blur_threshold):
                results.append(self._create_result(
                    ValidationStatus.WARNING,
                    "Image appears to be blurry",
                    threshold=blur_threshold,
                    metadata={"file_path": str(image_path)}
                ))
            
            image.close()
            
        except Exception as e:
            results.append(self._create_result(
                ValidationStatus.FAILED,
                f"Error validating image: {str(e)}",
                metadata={"file_path": str(image_path)}
            ))
        
        return results
    
    def _is_blurry(self, image_path: Path, threshold: float) -> bool:
        """Check if image is blurry using Laplacian variance."""
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return True
            
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            return laplacian_var < threshold
            
        except Exception:
            return False


# Data quality manager
class DataQualityManager:
    """Manages data quality validation across the pipeline."""
    
    def __init__(self):
        self.validators: Dict[str, BaseValidator] = {}
        self.logger = structlog.get_logger(__name__)
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default validators."""
        self.validators['file'] = FileValidator()
        self.validators['audio'] = AudioValidator()
        self.validators['video'] = VideoValidator()
        self.validators['transcript'] = TranscriptValidator()
        self.validators['image'] = ImageValidator()
    
    def register_validator(self, name: str, validator: BaseValidator):
        """Register a custom validator."""
        self.validators[name] = validator
    
    def validate_file(self, file_path: Union[str, Path], file_type: str, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate a file based on its type."""
        results = []
        
        # Always run file validator first
        if 'file' in self.validators:
            file_results = self.validators['file'].validate(file_path, context)
            results.extend(file_results)
        
        # Run type-specific validator
        if file_type in self.validators:
            type_results = self.validators[file_type].validate(file_path, context)
            results.extend(type_results)
        
        return results
    
    def validate_transcript(self, transcript_data: Union[Dict, List], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate transcript data."""
        if 'transcript' in self.validators:
            return self.validators['transcript'].validate(transcript_data, context)
        return []
    
    def validate_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate processing configuration."""
        results = []
        
        try:
            # Validate using Pydantic model
            validated_config = VideoConfig(**config)
            results.append(ValidationResult(
                check_name="config_validation",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message="Configuration is valid",
                metadata={"config": validated_config.dict()}
            ))
        except pydantic.ValidationError as e:
            results.append(ValidationResult(
                check_name="config_validation",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.ERROR,
                message=f"Configuration validation failed: {str(e)}",
                metadata={"errors": e.errors()}
            ))
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results."""
        if not results:
            return {"status": "no_checks", "total": 0}
        
        status_counts = {}
        severity_counts = {}
        
        for result in results:
            status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1
            severity_counts[result.severity.value] = severity_counts.get(result.severity.value, 0) + 1
        
        # Determine overall status
        if status_counts.get('failed', 0) > 0:
            overall_status = 'failed'
        elif status_counts.get('warning', 0) > 0:
            overall_status = 'warning'
        else:
            overall_status = 'passed'
        
        return {
            "status": overall_status,
            "total": len(results),
            "status_counts": status_counts,
            "severity_counts": severity_counts,
            "results": [asdict(result) for result in results]
        }


# Global data quality manager
data_quality_manager = DataQualityManager()


# Utility functions
def validate_processing_job(job_config: Dict[str, Any], input_path: Optional[str] = None) -> Dict[str, Any]:
    """Validate a complete processing job."""
    results = []
    
    # Validate configuration
    config_results = data_quality_manager.validate_config(job_config)
    results.extend(config_results)
    
    # Validate input file if provided
    if input_path:
        file_type = job_config.get('job_type', 'unknown')
        file_results = data_quality_manager.validate_file(input_path, file_type)
        results.extend(file_results)
    
    # Get summary
    summary = data_quality_manager.get_validation_summary(results)
    
    return summary


def log_validation_results(results: List[ValidationResult], context: Optional[Dict[str, Any]] = None):
    """Log validation results."""
    logger = structlog.get_logger(__name__)
    
    for result in results:
        log_level = {
            ValidationSeverity.INFO: logger.info,
            ValidationSeverity.WARNING: logger.warning,
            ValidationSeverity.ERROR: logger.error,
            ValidationSeverity.CRITICAL: logger.critical
        }.get(result.severity, logger.info)
        
        log_level(
            "Validation result",
            check_name=result.check_name,
            status=result.status.value,
            severity=result.severity.value,
            message=result.message,
            expected_value=result.expected_value,
            actual_value=result.actual_value,
            threshold=result.threshold,
            metadata=result.metadata,
            context=context
        )
