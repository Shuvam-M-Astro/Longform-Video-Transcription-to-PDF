"""
Enhanced main processing pipeline with data engineering improvements.
"""

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import enhanced modules
from src.video_doc.database import (
    get_db_session, ProcessingJob, ProcessingStep, QualityCheck,
    ProcessingStatus, JobManager, check_database_health
)
from src.video_doc.error_handling import (
    ProcessingError, ErrorSeverity, error_handler, retry_on_failure,
    with_error_handling, with_circuit_breaker, error_context
)
from src.video_doc.monitoring import (
    get_logger, metrics, PerformanceMonitor, correlation_id_context,
    log_job_start, log_job_completion, log_step_start, log_step_completion,
    log_error, log_quality_check, audit_logger
)
from src.video_doc.data_validation import (
    data_quality_manager, validate_processing_job, log_validation_results
)

# Import original processing modules
from src.video_doc.download import download_video
from src.video_doc.audio import extract_audio_wav
from src.video_doc.transcribe import transcribe_audio
from src.video_doc.frames import extract_keyframes, build_contact_sheet
from src.video_doc.classify import classify_frames
from src.video_doc.pdf import build_pdf_report
from src.video_doc.stream import (
    resolve_stream_urls, stream_extract_audio, stream_extract_keyframes,
    fallback_download_audio_via_ytdlp, fallback_download_small_video,
)
from src.video_doc.progress import PipelineProgress, make_console_progress_printer

logger = get_logger(__name__)


class EnhancedVideoProcessor:
    """Enhanced video processor with data engineering features."""
    
    def __init__(self, job_id: Optional[str] = None):
        self.job_id = job_id or str(uuid.uuid4())
        self.db_session = get_db_session()
        self.job_manager = JobManager(self.db_session)
        self.start_time = time.time()
        
        # Initialize job in database
        self.job = None
        self.processing_steps: Dict[str, ProcessingStep] = {}
        
        # Set up correlation ID for logging
        correlation_id_context.set_correlation_id(self.job_id)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        try:
            if self.db_session:
                self.db_session.close()
        except Exception as e:
            logger.error("Error closing database session", error=str(e))
    
    def create_job(self, job_type: str, identifier: str, config: Dict[str, Any]) -> ProcessingJob:
        """Create a new processing job."""
        try:
            # Validate configuration
            validation_results = data_quality_manager.validate_config(config)
            log_validation_results(validation_results, {"job_id": self.job_id})
            
            # Create job in database
            self.job = self.job_manager.create_job(job_type, identifier, config)
            self.job_id = str(self.job.id)
            
            # Update correlation ID
            correlation_id_context.set_correlation_id(self.job_id)
            
            # Log job creation
            log_job_start(self.job_id, job_type, config)
            audit_logger.log_operation(
                operation="create_job",
                resource_type="processing_job",
                resource_id=self.job_id,
                new_values={"job_type": job_type, "identifier": identifier, "config": config}
            )
            
            return self.job
            
        except Exception as e:
            log_error(e, {"operation": "create_job", "job_type": job_type})
            raise ProcessingError(
                f"Failed to create job: {str(e)}",
                "JOB_CREATION_FAILED",
                ErrorSeverity.HIGH,
                context={"job_type": job_type, "identifier": identifier}
            )
    
    def add_processing_step(self, step_name: str, step_order: int) -> ProcessingStep:
        """Add a processing step."""
        try:
            step = self.job_manager.add_processing_step(
                self.job_id, step_name, step_order, ProcessingStatus.PENDING
            )
            self.processing_steps[step_name] = step
            
            log_step_start(step_name, self.job_id)
            return step
            
        except Exception as e:
            log_error(e, {"operation": "add_step", "step_name": step_name})
            raise ProcessingError(
                f"Failed to add processing step: {str(e)}",
                "STEP_CREATION_FAILED",
                ErrorSeverity.MEDIUM,
                context={"step_name": step_name}
            )
    
    def update_step_status(
        self,
        step_name: str,
        status: ProcessingStatus,
        progress: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Update processing step status."""
        try:
            if step_name in self.processing_steps:
                step_id = str(self.processing_steps[step_name].id)
                success = self.job_manager.update_step_status(
                    step_id, status, progress, metrics, error_message
                )
                
                if success:
                    duration = time.time() - self.start_time
                    log_step_completion(step_name, self.job_id, duration, status == ProcessingStatus.COMPLETED)
                else:
                    logger.warning("Failed to update step status", step_name=step_name)
            
        except Exception as e:
            log_error(e, {"operation": "update_step", "step_name": step_name})
    
    def update_job_status(
        self,
        status: ProcessingStatus,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Update job status."""
        try:
            success = self.job_manager.update_job_status(
                self.job_id, status, progress, current_step, error_message
            )
            
            if success:
                duration = time.time() - self.start_time
                log_job_completion(
                    self.job_id,
                    self.job.job_type if self.job else "unknown",
                    duration,
                    status == ProcessingStatus.COMPLETED
                )
            else:
                logger.warning("Failed to update job status", job_id=self.job_id)
                
        except Exception as e:
            log_error(e, {"operation": "update_job", "job_id": self.job_id})
    
    def add_quality_check(
        self,
        check_name: str,
        check_type: str,
        status: str,
        expected_value: Optional[str] = None,
        actual_value: Optional[str] = None,
        threshold: Optional[float] = None,
        message: Optional[str] = None
    ):
        """Add quality check result."""
        try:
            self.job_manager.add_quality_check(
                self.job_id, check_name, check_type, status,
                expected_value, actual_value, threshold, message
            )
            
            log_quality_check(check_name, check_type, status, {
                "expected_value": expected_value,
                "actual_value": actual_value,
                "threshold": threshold,
                "message": message
            })
            
        except Exception as e:
            log_error(e, {"operation": "add_quality_check", "check_name": check_name})
    
    def process_video(self, args) -> Dict[str, Any]:
        """Process video with enhanced error handling and monitoring."""
        
        with PerformanceMonitor("video_processing", {"job_id": self.job_id}):
            try:
                # Update job status to processing
                self.update_job_status(ProcessingStatus.PROCESSING, 0.0, "Initializing")
                
                # Validate input
                self._validate_input(args)
                
                # Process video based on mode
                if args.streaming:
                    result = self._process_streaming(args)
                else:
                    result = self._process_download(args)
                
                # Update job status to completed
                self.update_job_status(ProcessingStatus.COMPLETED, 100.0, "Completed")
                
                return result
                
            except Exception as e:
                # Update job status to failed
                self.update_job_status(
                    ProcessingStatus.FAILED,
                    error_message=str(e),
                    current_step="Error"
                )
                
                # Log error
                log_error(e, {"job_id": self.job_id, "operation": "process_video"})
                
                raise ProcessingError(
                    f"Video processing failed: {str(e)}",
                    "PROCESSING_FAILED",
                    ErrorSeverity.HIGH,
                    context={"job_id": self.job_id},
                    original_exception=e
                )
    
    def _validate_input(self, args):
        """Validate input parameters."""
        with error_context("input_validation"):
            # Validate configuration
            config = {
                "url": getattr(args, 'url', None),
                "video_path": getattr(args, 'video', None),
                "language": args.language,
                "whisper_model": args.whisper_model,
                "beam_size": args.beam_size,
                "transcribe_only": args.transcribe_only,
                "streaming": args.streaming,
                "kf_method": args.kf_method,
                "max_fps": args.max_fps,
                "min_scene_diff": args.min_scene_diff,
                "report_style": args.report_style
            }
            
            validation_results = data_quality_manager.validate_config(config)
            
            # Check for critical validation failures
            critical_failures = [
                r for r in validation_results 
                if r.status == "failed" and r.severity == ValidationSeverity.CRITICAL
            ]
            
            if critical_failures:
                raise ProcessingError(
                    f"Critical validation failures: {[f.message for f in critical_failures]}",
                    "VALIDATION_FAILED",
                    ErrorSeverity.CRITICAL,
                    context={"failures": [f.message for f in critical_failures]}
                )
            
            # Log validation results
            log_validation_results(validation_results, {"job_id": self.job_id})
            
            # Add quality checks for validation results
            for result in validation_results:
                self.add_quality_check(
                    f"config_{result.check_name}",
                    "validation",
                    result.status.value,
                    result.expected_value,
                    result.actual_value,
                    result.threshold,
                    result.message
                )
    
    @retry_on_failure(max_attempts=3, base_delay=2.0)
    @with_circuit_breaker("download_service")
    def _process_download(self, args):
        """Process video in download mode."""
        
        # Download step
        download_step = self.add_processing_step("download", 1)
        self.update_step_status(download_step.step_name, ProcessingStatus.PROCESSING)
        
        try:
            with PerformanceMonitor("download", {"job_id": self.job_id}):
                video_path = Path(args.out) / "video.mp4"
                
                if not args.skip_download or not video_path.exists():
                    download_video(
                        args.url, video_path,
                        cookies_from_browser=args.cookies_from_browser,
                        browser_profile=args.browser_profile,
                        cookies_file=Path(args.cookies_file) if args.cookies_file else None,
                        use_android_client=args.use_android_client,
                        progress_cb=lambda p: self.update_step_status(
                            download_step.step_name, ProcessingStatus.PROCESSING, p
                        )
                    )
                
                # Validate downloaded video
                self._validate_video_file(video_path)
                
                self.update_step_status(download_step.step_name, ProcessingStatus.COMPLETED)
                
        except Exception as e:
            self.update_step_status(
                download_step.step_name, ProcessingStatus.FAILED, error_message=str(e)
            )
            raise
        
        # Continue with audio extraction and other steps
        return self._process_audio_and_transcription(args, video_path)
    
    @retry_on_failure(max_attempts=3, base_delay=1.0)
    @with_circuit_breaker("streaming_service")
    def _process_streaming(self, args):
        """Process video in streaming mode."""
        
        # Stream resolution step
        stream_step = self.add_processing_step("stream_resolution", 1)
        self.update_step_status(stream_step.step_name, ProcessingStatus.PROCESSING)
        
        try:
            with PerformanceMonitor("stream_resolution", {"job_id": self.job_id}):
                resolved = resolve_stream_urls(
                    args.url,
                    cookies_from_browser=args.cookies_from_browser,
                    browser_profile=args.browser_profile,
                    cookies_file=Path(args.cookies_file) if args.cookies_file else None,
                    use_android_client=args.use_android_client,
                )
                
                self.update_step_status(stream_step.step_name, ProcessingStatus.COMPLETED)
                
        except Exception as e:
            self.update_step_status(
                stream_step.step_name, ProcessingStatus.FAILED, error_message=str(e)
            )
            raise
        
        # Process audio and video streams
        return self._process_streaming_audio_video(args, resolved)
    
    def _process_audio_and_transcription(self, args, video_path: Path):
        """Process audio extraction and transcription."""
        
        # Audio extraction step
        audio_step = self.add_processing_step("audio_extraction", 2)
        self.update_step_status(audio_step.step_name, ProcessingStatus.PROCESSING)
        
        try:
            with PerformanceMonitor("audio_extraction", {"job_id": self.job_id}):
                audio_path = Path(args.out) / "audio.wav"
                
                extract_audio_wav(
                    video_path, audio_path,
                    progress_cb=lambda p: self.update_step_status(
                        audio_step.step_name, ProcessingStatus.PROCESSING, p
                    ),
                    start_time=args.trim_start,
                    end_trim=args.trim_end,
                    volume_gain_db=args.volume_gain,
                )
                
                # Validate audio file
                self._validate_audio_file(audio_path)
                
                self.update_step_status(audio_step.step_name, ProcessingStatus.COMPLETED)
                
        except Exception as e:
            self.update_step_status(
                audio_step.step_name, ProcessingStatus.FAILED, error_message=str(e)
            )
            raise
        
        # Transcription step
        transcription_step = self.add_processing_step("transcription", 3)
        self.update_step_status(transcription_step.step_name, ProcessingStatus.PROCESSING)
        
        try:
            with PerformanceMonitor("transcription", {"job_id": self.job_id}):
                transcript_txt = Path(args.out) / "transcript" / "transcript.txt"
                segments_json = Path(args.out) / "transcript" / "segments.json"
                srt_path = transcript_txt.parent / "transcript.srt" if args.export_srt else None
                
                segments = transcribe_audio(
                    audio_path=audio_path,
                    transcript_txt_path=transcript_txt,
                    segments_json_path=segments_json,
                    language=args.language,
                    beam_size=args.beam_size,
                    model_size=args.whisper_model,
                    progress_cb=lambda p: self.update_step_status(
                        transcription_step.step_name, ProcessingStatus.PROCESSING, p
                    ),
                    cpu_threads=(args.whisper_cpu_threads if args.whisper_cpu_threads and args.whisper_cpu_threads > 0 else None),
                    num_workers=(args.whisper_num_workers if args.whisper_num_workers and args.whisper_num_workers > 0 else None),
                    srt_path=srt_path,
                )
                
                # Validate transcript
                self._validate_transcript(segments)
                
                self.update_step_status(transcription_step.step_name, ProcessingStatus.COMPLETED)
                
        except Exception as e:
            self.update_step_status(
                transcription_step.step_name, ProcessingStatus.FAILED, error_message=str(e)
            )
            raise
        
        # Continue with frame extraction and classification if not transcribe-only
        if not args.transcribe_only:
            return self._process_frames_and_classification(args, video_path, segments)
        else:
            return self._build_final_pdf(args, segments, {"code": [], "plots": [], "images": []})
    
    def _process_streaming_audio_video(self, args, resolved: Dict[str, Any]):
        """Process streaming audio and video."""
        
        # Audio streaming step
        audio_step = self.add_processing_step("audio_streaming", 2)
        self.update_step_status(audio_step.step_name, ProcessingStatus.PROCESSING)
        
        try:
            with PerformanceMonitor("audio_streaming", {"job_id": self.job_id}):
                audio_path = Path(args.out) / "audio.wav"
                
                if resolved.get("audio_url"):
                    stream_extract_audio(
                        resolved["audio_url"], audio_path,
                        headers=resolved.get("headers"),
                        progress_cb=lambda p: self.update_step_status(
                            audio_step.step_name, ProcessingStatus.PROCESSING, p
                        )
                    )
                else:
                    # Fallback to yt-dlp
                    fallback_download_audio_via_ytdlp(
                        args.url, audio_path,
                        cookies_from_browser=args.cookies_from_browser,
                        browser_profile=args.browser_profile,
                        cookies_file=Path(args.cookies_file) if args.cookies_file else None,
                        use_android_client=args.use_android_client,
                    )
                
                # Validate audio file
                self._validate_audio_file(audio_path)
                
                self.update_step_status(audio_step.step_name, ProcessingStatus.COMPLETED)
                
        except Exception as e:
            self.update_step_status(
                audio_step.step_name, ProcessingStatus.FAILED, error_message=str(e)
            )
            raise
        
        # Continue with transcription
        return self._process_audio_and_transcription(args, None)
    
    def _process_frames_and_classification(self, args, video_path: Path, segments):
        """Process frame extraction and classification."""
        
        # Frame extraction step
        frames_step = self.add_processing_step("frame_extraction", 4)
        self.update_step_status(frames_step.step_name, ProcessingStatus.PROCESSING)
        
        try:
            with PerformanceMonitor("frame_extraction", {"job_id": self.job_id}):
                frames_dir = Path(args.out) / "frames" / "keyframes"
                
                keyframe_paths = extract_keyframes(
                    video_path=video_path,
                    output_dir=frames_dir,
                    max_fps=args.max_fps,
                    scene_threshold=args.min_scene_diff,
                    method=args.kf_method,
                    interval_sec=args.kf_interval_sec,
                    output_format=args.frame_format,
                    jpeg_quality=args.frame_quality,
                    max_width=args.frame_max_width,
                    max_frames=(args.frame_max_frames if args.frame_max_frames > 0 else None),
                    skip_dark=args.skip_dark,
                    dark_pixel_value=args.dark_value,
                    dark_ratio_threshold=args.dark_ratio,
                    dedupe=args.dedupe,
                    dedupe_similarity=args.dedupe_sim,
                    progress_cb=lambda p: self.update_step_status(
                        frames_step.step_name, ProcessingStatus.PROCESSING, p
                    )
                )
                
                # Validate extracted frames
                self._validate_frames(keyframe_paths)
                
                self.update_step_status(frames_step.step_name, ProcessingStatus.COMPLETED)
                
        except Exception as e:
            self.update_step_status(
                frames_step.step_name, ProcessingStatus.FAILED, error_message=str(e)
            )
            raise
        
        # Classification step
        classification_step = self.add_processing_step("classification", 5)
        self.update_step_status(classification_step.step_name, ProcessingStatus.PROCESSING)
        
        try:
            with PerformanceMonitor("classification", {"job_id": self.job_id}):
                classified_dir = Path(args.out) / "classified"
                snippets_dir = Path(args.out) / "snippets" / "code"
                
                classification_result = classify_frames(
                    frame_paths=keyframe_paths,
                    classified_root=classified_dir,
                    snippets_dir=snippets_dir,
                    progress_cb=lambda p: self.update_step_status(
                        classification_step.step_name, ProcessingStatus.PROCESSING, p
                    ),
                    ocr_languages=[lang.strip() for lang in str(args.ocr_langs).split(',') if lang.strip()],
                    skip_blurry=args.skip_blurry,
                    blurry_threshold=args.blurry_threshold,
                    max_per_category=(args.max_per_category if args.max_per_category > 0 else None),
                )
                
                # Validate classification results
                self._validate_classification(classification_result)
                
                self.update_step_status(classification_step.step_name, ProcessingStatus.COMPLETED)
                
        except Exception as e:
            self.update_step_status(
                classification_step.step_name, ProcessingStatus.FAILED, error_message=str(e)
            )
            raise
        
        # Build final PDF
        return self._build_final_pdf(args, segments, classification_result)
    
    def _build_final_pdf(self, args, segments, classification_result):
        """Build final PDF report."""
        
        # PDF generation step
        pdf_step = self.add_processing_step("pdf_generation", 6)
        self.update_step_status(pdf_step.step_name, ProcessingStatus.PROCESSING)
        
        try:
            with PerformanceMonitor("pdf_generation", {"job_id": self.job_id}):
                report_pdf_path = Path(args.out) / "report.pdf"
                transcript_txt = Path(args.out) / "transcript" / "transcript.txt"
                segments_json = Path(args.out) / "transcript" / "segments.json"
                
                # Detect contact sheet
                contact_sheet_path = None
                try:
                    possible_cs = Path(args.out) / "frames" / "contact_sheet.jpg"
                    contact_sheet_path = possible_cs if possible_cs.exists() else None
                except Exception:
                    contact_sheet_path = None
                
                build_pdf_report(
                    output_pdf_path=report_pdf_path,
                    transcript_txt_path=transcript_txt,
                    segments_json_path=segments_json,
                    classification_result=classification_result,
                    output_dir=Path(args.out),
                    progress_cb=lambda p: self.update_step_status(
                        pdf_step.step_name, ProcessingStatus.PROCESSING, p
                    ),
                    report_style=args.report_style,
                    video_title=getattr(args, 'video_title', 'Video Report'),
                    contact_sheet_path=contact_sheet_path,
                )
                
                # Validate PDF
                self._validate_pdf(report_pdf_path)
                
                self.update_step_status(pdf_step.step_name, ProcessingStatus.COMPLETED)
                
        except Exception as e:
            self.update_step_status(
                pdf_step.step_name, ProcessingStatus.FAILED, error_message=str(e)
            )
            raise
        
        # Return final results
        return {
            "job_id": self.job_id,
            "status": "completed",
            "output_files": {
                "pdf": str(report_pdf_path),
                "transcript": str(transcript_txt),
                "segments": str(segments_json),
                "audio": str(Path(args.out) / "audio.wav")
            },
            "metrics": {
                "total_segments": len(segments),
                "classified_frames": sum(len(frames) for frames in classification_result.values()),
                "processing_time": time.time() - self.start_time
            }
        }
    
    def _validate_video_file(self, video_path: Path):
        """Validate video file quality."""
        validation_results = data_quality_manager.validate_file(
            video_path, "video", {
                "min_duration": 1.0,
                "max_duration": 7200.0,
                "min_width": 320,
                "min_height": 240
            }
        )
        
        for result in validation_results:
            self.add_quality_check(
                f"video_{result.check_name}",
                "quality",
                result.status.value,
                result.expected_value,
                result.actual_value,
                result.threshold,
                result.message
            )
    
    def _validate_audio_file(self, audio_path: Path):
        """Validate audio file quality."""
        validation_results = data_quality_manager.validate_file(
            audio_path, "audio", {
                "min_duration": 1.0,
                "max_duration": 3600.0,
                "silence_threshold": 0.01
            }
        )
        
        for result in validation_results:
            self.add_quality_check(
                f"audio_{result.check_name}",
                "quality",
                result.status.value,
                result.expected_value,
                result.actual_value,
                result.threshold,
                result.message
            )
    
    def _validate_transcript(self, segments):
        """Validate transcript quality."""
        validation_results = data_quality_manager.validate_transcript(
            segments, {
                "min_segments": 1,
                "max_segments": 10000,
                "min_text_length": 10,
                "empty_segment_threshold": 0.1
            }
        )
        
        for result in validation_results:
            self.add_quality_check(
                f"transcript_{result.check_name}",
                "quality",
                result.status.value,
                result.expected_value,
                result.actual_value,
                result.threshold,
                result.message
            )
    
    def _validate_frames(self, frame_paths: List[Path]):
        """Validate extracted frames."""
        for i, frame_path in enumerate(frame_paths[:10]):  # Sample first 10 frames
            validation_results = data_quality_manager.validate_file(
                frame_path, "image", {
                    "min_width": 32,
                    "min_height": 32,
                    "blur_threshold": 100.0
                }
            )
            
            for result in validation_results:
                self.add_quality_check(
                    f"frame_{i}_{result.check_name}",
                    "quality",
                    result.status.value,
                    result.expected_value,
                    result.actual_value,
                    result.threshold,
                    result.message
                )
    
    def _validate_classification(self, classification_result):
        """Validate classification results."""
        total_frames = sum(len(frames) for frames in classification_result.values())
        
        self.add_quality_check(
            "classification_total_frames",
            "quality",
            "passed" if total_frames > 0 else "warning",
            "> 0",
            str(total_frames),
            message=f"Total classified frames: {total_frames}"
        )
        
        for category, frames in classification_result.items():
            self.add_quality_check(
                f"classification_{category}",
                "quality",
                "passed",
                None,
                str(len(frames)),
                message=f"Frames in {category}: {len(frames)}"
            )
    
    def _validate_pdf(self, pdf_path: Path):
        """Validate generated PDF."""
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            self.add_quality_check(
                "pdf_generation",
                "quality",
                "passed",
                "> 0 bytes",
                f"{pdf_path.stat().st_size} bytes",
                message="PDF generated successfully"
            )
        else:
            self.add_quality_check(
                "pdf_generation",
                "quality",
                "failed",
                "> 0 bytes",
                "0 bytes",
                message="PDF generation failed"
            )


def ensure_dirs(output_dir: Path) -> None:
    """Ensure output directories exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transcript").mkdir(parents=True, exist_ok=True)
    (output_dir / "frames" / "keyframes").mkdir(parents=True, exist_ok=True)
    (output_dir / "classified" / "code").mkdir(parents=True, exist_ok=True)
    (output_dir / "classified" / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "classified" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "snippets" / "code").mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Video Documentation Builder")
    
    # Input options
    parser.add_argument("--url", type=str, help="Video URL")
    parser.add_argument("--video", type=str, help="Path to local video file")
    parser.add_argument("--out", type=str, default="./outputs/run", help="Output directory")
    
    # Processing options
    parser.add_argument("--language", type=str, default="auto", help="Language code or 'auto'")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--whisper-model", type=str, default="medium", help="faster-whisper model size")
    parser.add_argument("--transcribe-only", action="store_true", help="Skip frames/classification")
    parser.add_argument("--streaming", action="store_true", help="Process via streaming")
    
    # Keyframe options
    parser.add_argument("--kf-method", type=str, choices=["scene", "iframe", "interval"], default="scene")
    parser.add_argument("--max-fps", type=float, default=1.0, help="Max FPS for keyframe detection")
    parser.add_argument("--min-scene-diff", type=float, default=0.45, help="Scene change threshold")
    parser.add_argument("--kf-interval-sec", type=float, default=5.0, help="Interval seconds")
    
    # Output options
    parser.add_argument("--report-style", type=str, choices=["minimal", "book"], default="book")
    parser.add_argument("--export-srt", action="store_true", help="Export SRT file")
    
    # Download options
    parser.add_argument("--cookies-from-browser", type=str, help="Browser for cookies")
    parser.add_argument("--browser-profile", type=str, help="Browser profile")
    parser.add_argument("--cookies-file", type=str, help="Cookies file path")
    parser.add_argument("--use-android-client", action="store_true", help="Use Android client")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if exists")
    
    # Additional options
    parser.add_argument("--resume", action="store_true", help="Resume processing")
    parser.add_argument("--whisper-cpu-threads", type=int, default=0, help="CPU threads")
    parser.add_argument("--whisper-num-workers", type=int, default=1, help="Number of workers")
    
    # Frame processing options
    parser.add_argument("--frame-format", type=str, choices=["jpg", "png", "webp"], default="jpg")
    parser.add_argument("--frame-quality", type=int, default=90, help="Frame quality")
    parser.add_argument("--frame-max-width", type=int, default=1280, help="Max frame width")
    parser.add_argument("--frame-max-frames", type=int, default=0, help="Max frames")
    parser.add_argument("--skip-dark", action="store_true", help="Skip dark frames")
    parser.add_argument("--dark-value", type=int, default=16, help="Dark pixel threshold")
    parser.add_argument("--dark-ratio", type=float, default=0.98, help="Dark pixel ratio")
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate frames")
    parser.add_argument("--dedupe-sim", type=float, default=0.995, help="Deduplication similarity")
    
    # Audio processing options
    parser.add_argument("--trim-start", type=float, default=0.0, help="Trim start seconds")
    parser.add_argument("--trim-end", type=float, default=0.0, help="Trim end seconds")
    parser.add_argument("--volume-gain", type=float, default=0.0, help="Volume gain in dB")
    
    # Classification options
    parser.add_argument("--ocr-langs", type=str, default="en", help="OCR languages")
    parser.add_argument("--skip-blurry", action="store_true", help="Skip blurry frames")
    parser.add_argument("--blurry-threshold", type=float, default=60.0, help="Blur threshold")
    parser.add_argument("--max-per-category", type=int, default=0, help="Max per category")
    
    args = parser.parse_args()
    
    if not args.url and not args.video:
        parser.error("You must provide either --url or --video")
    
    return args


def main() -> None:
    """Main entry point with enhanced processing."""
    
    # Check database health
    db_health = check_database_health()
    if db_health["status"] != "healthy":
        logger.error("Database health check failed", health=db_health)
        sys.exit(1)
    
    args = parse_args()
    output_dir = Path(args.out)
    
    # Clean output directory unless resuming
    if output_dir.exists() and not getattr(args, "resume", False):
        logger.info("Cleaning output directory", path=str(output_dir))
        shutil.rmtree(output_dir, ignore_errors=True)
    
    ensure_dirs(output_dir)
    
    # Process video with enhanced pipeline
    with EnhancedVideoProcessor() as processor:
        try:
            # Create job
            job_type = "url" if args.url else "file"
            identifier = args.url or args.video
            
            config = {
                "language": args.language,
                "whisper_model": args.whisper_model,
                "beam_size": args.beam_size,
                "transcribe_only": args.transcribe_only,
                "streaming": args.streaming,
                "kf_method": args.kf_method,
                "max_fps": args.max_fps,
                "min_scene_diff": args.min_scene_diff,
                "report_style": args.report_style
            }
            
            job = processor.create_job(job_type, identifier, config)
            
            # Process video
            result = processor.process_video(args)
            
            # Log final results
            logger.info("Video processing completed", **result)
            
            # Print results in JSON format
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            logger.error("Video processing failed", error=str(e), job_id=processor.job_id)
            sys.exit(1)


if __name__ == "__main__":
    main()
