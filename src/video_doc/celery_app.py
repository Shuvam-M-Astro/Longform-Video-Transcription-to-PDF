"""
Distributed task queue system using Celery for horizontal scaling.
"""

import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, worker_ready
from celery.exceptions import Retry, MaxRetriesExceededError
import redis
from kombu import Queue

# Import our enhanced modules
from src.video_doc.database import (
    get_db_session, ProcessingJob, ProcessingStep, QualityCheck,
    ProcessingStatus, JobManager
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

logger = get_logger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
CELERY_DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://video_doc:password@localhost:5432/video_doc_db')

# Create Celery app
app = Celery('video_processor')

# Celery configuration
app.conf.update(
    broker_url=CELERY_BROKER_URL,
    result_backend=CELERY_RESULT_BACKEND,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing and queues
    task_routes={
        'video_processor.tasks.process_video': {'queue': 'video_processing'},
        'video_processor.tasks.process_audio': {'queue': 'audio_processing'},
        'video_processor.tasks.process_transcription': {'queue': 'transcription'},
        'video_processor.tasks.process_frames': {'queue': 'frame_processing'},
        'video_processor.tasks.process_classification': {'queue': 'classification'},
        'video_processor.tasks.generate_pdf': {'queue': 'pdf_generation'},
    },
    
    # Queue configuration
    task_default_queue='default',
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('video_processing', routing_key='video_processing'),
        Queue('audio_processing', routing_key='audio_processing'),
        Queue('transcription', routing_key='transcription'),
        Queue('frame_processing', routing_key='frame_processing'),
        Queue('classification', routing_key='classification'),
        Queue('pdf_generation', routing_key='pdf_generation'),
    ),
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True,
    
    # Retry configuration
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Result backend configuration
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Security
    worker_hijack_root_logger=False,
    worker_log_color=False,
)


class DatabaseTask(Task):
    """Base task class with database session management."""
    
    def __init__(self):
        self.db_session = None
        self.job_manager = None
    
    def before_start(self, task_id, args, kwargs):
        """Initialize database session before task starts."""
        try:
            self.db_session = get_db_session()
            self.job_manager = JobManager(self.db_session)
            logger.info(f"Database session initialized for task {task_id}")
        except Exception as e:
            logger.error(f"Failed to initialize database session for task {task_id}: {e}")
            raise
    
    def on_success(self, retval, task_id, args, kwargs):
        """Clean up database session on success."""
        if self.db_session:
            self.db_session.close()
            logger.info(f"Database session closed for task {task_id}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Clean up database session on failure."""
        if self.db_session:
            self.db_session.close()
            logger.error(f"Database session closed after failure for task {task_id}")
        try:
            metrics.increment_task_failure(getattr(self, 'name', 'unknown'), type(exc).__name__)
        except Exception:
            pass
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(f"Task {task_id} retrying due to: {exc}")
        try:
            metrics.increment_task_retry(getattr(self, 'name', 'unknown'))
        except Exception:
            pass


# Utility to enqueue tasks with enqueued_at header
def _apply_async_with_headers(task_sig, args=None, kwargs=None, queue: Optional[str] = None):
    args = args or []
    kwargs = kwargs or {}
    headers = {"enqueued_at": time.time()}
    return task_sig.apply_async(args=args, kwargs=kwargs, headers=headers, queue=queue)


# Task definitions
@app.task(bind=True, base=DatabaseTask, name='video_processor.tasks.process_video')
def process_video_task(self, job_id: str, job_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main video processing task that orchestrates the entire pipeline.
    """
    correlation_id = str(uuid.uuid4())
    
    with correlation_id_context(correlation_id):
        with PerformanceMonitor("video_processing_task", {"job_id": job_id, "task_id": self.request.id}):
            try:
                logger.info(f"Starting video processing task", job_id=job_id, task_id=self.request.id)
                
                # Update job status to processing
                self.job_manager.update_job_status(job_id, ProcessingStatus.PROCESSING, 0.0, "Starting")
                
                # Validate configuration
                validation_results = data_quality_manager.validate_config(job_config)
                log_validation_results(validation_results, {"job_id": job_id, "task_id": self.request.id})
                
                # Process based on job type
                if job_config.get('job_type') == 'url':
                    result = _apply_async_with_headers(process_url_video, args=[job_id, job_config], queue='video_processing')
                else:
                    result = _apply_async_with_headers(process_file_video, args=[job_id, job_config], queue='video_processing')
                
                # Wait for completion with timeout
                try:
                    final_result = result.get(timeout=3600)  # 1 hour timeout
                    
                    # Update job status to completed
                    self.job_manager.update_job_status(job_id, ProcessingStatus.COMPLETED, 100.0, "Completed")
                    
                    logger.info(f"Video processing task completed", job_id=job_id, task_id=self.request.id)
                    return final_result
                    
                except Exception as e:
                    logger.error(f"Sub-task failed for job {job_id}: {e}")
                    self.job_manager.update_job_status(job_id, ProcessingStatus.FAILED, error_message=str(e))
                    raise ProcessingError(
                        f"Video processing failed: {str(e)}",
                        "PROCESSING_FAILED",
                        ErrorSeverity.HIGH,
                        context={"job_id": job_id, "task_id": self.request.id}
                    )
                
            except Exception as e:
                logger.error(f"Video processing task failed", job_id=job_id, task_id=self.request.id, error=str(e))
                self.job_manager.update_job_status(job_id, ProcessingStatus.FAILED, error_message=str(e))
                raise


@app.task(bind=True, base=DatabaseTask, name='video_processor.tasks.process_url_video')
@retry_on_failure(max_attempts=3, base_delay=2.0)
@with_circuit_breaker("download_service")
def process_url_video(self, job_id: str, job_config: Dict[str, Any]) -> Dict[str, Any]:
    """Process video from URL."""
    try:
        from src.video_doc.download import download_video
        from src.video_doc.stream import resolve_stream_urls
        
        logger.info(f"Processing URL video", job_id=job_id)
        
        # Add processing step
        step = self.job_manager.add_processing_step(job_id, "download", 1, ProcessingStatus.PROCESSING)
        
        # Download or stream video
        if job_config.get('streaming', False):
            # Streaming mode
            resolved = resolve_stream_urls(job_config['url'])
            # Process streaming audio and video
            audio_result = process_streaming_audio.delay(job_id, resolved)
            video_result = process_streaming_video.delay(job_id, resolved) if not job_config.get('transcribe_only', False) else None
        else:
            # Download mode
            output_dir = Path(job_config['output_dir'])
            video_path = output_dir / "video.mp4"
            
            download_video(
                job_config['url'],
                video_path,
                cookies_from_browser=job_config.get('cookies_from_browser'),
                browser_profile=job_config.get('browser_profile'),
                cookies_file=Path(job_config['cookies_file']) if job_config.get('cookies_file') else None,
                use_android_client=job_config.get('use_android_client', False)
            )
            
            # Process downloaded video
            audio_result = _apply_async_with_headers(process_audio_task, args=[job_id, {"video_path": str(video_path)}], queue='audio_processing')
            video_result = _apply_async_with_headers(process_frames_task, args=[job_id, {"video_path": str(video_path)}], queue='frame_processing') if not job_config.get('transcribe_only', False) else None
        
        # Update step status
        self.job_manager.update_step_status(str(step.id), ProcessingStatus.COMPLETED)
        
        # Wait for audio processing
        audio_result_data = audio_result.get(timeout=1800)  # 30 minutes
        
        # Wait for video processing if applicable
        video_result_data = video_result.get(timeout=1800) if video_result else None
        
        return {
            "job_id": job_id,
            "audio_result": audio_result_data,
            "video_result": video_result_data,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"URL video processing failed", job_id=job_id, error=str(e))
        raise


@app.task(bind=True, base=DatabaseTask, name='video_processor.tasks.process_file_video')
def process_file_video(self, job_id: str, job_config: Dict[str, Any]) -> Dict[str, Any]:
    """Process uploaded video file."""
    try:
        logger.info(f"Processing file video", job_id=job_id)
        
        video_path = job_config['file_path']
        
        # Process audio and video in parallel
        audio_result = _apply_async_with_headers(process_audio_task, args=[job_id, {"video_path": video_path}], queue='audio_processing')
        video_result = _apply_async_with_headers(process_frames_task, args=[job_id, {"video_path": video_path}], queue='frame_processing') if not job_config.get('transcribe_only', False) else None
        
        # Wait for results
        audio_result_data = audio_result.get(timeout=1800)  # 30 minutes
        video_result_data = video_result.get(timeout=1800) if video_result else None
        
        return {
            "job_id": job_id,
            "audio_result": audio_result_data,
            "video_result": video_result_data,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"File video processing failed", job_id=job_id, error=str(e))
        raise


@app.task(bind=True, base=DatabaseTask, name='video_processor.tasks.process_audio')
@retry_on_failure(max_attempts=3, base_delay=1.0)
def process_audio_task(self, job_id: str, audio_config: Dict[str, Any]) -> Dict[str, Any]:
    """Process audio extraction and transcription."""
    try:
        logger.info(f"Processing audio", job_id=job_id)
        
        # Add processing step
        step = self.job_manager.add_processing_step(job_id, "audio_extraction", 2, ProcessingStatus.PROCESSING)
        
        from src.video_doc.audio import extract_audio_wav
        from src.video_doc.transcribe import transcribe_audio
        
        video_path = Path(audio_config['video_path'])
        output_dir = video_path.parent
        audio_path = output_dir / "audio.wav"
        
        # Extract audio
        extract_audio_wav(video_path, audio_path)
        
        # Update step status
        self.job_manager.update_step_status(str(step.id), ProcessingStatus.COMPLETED)
        
        # Process transcription
        transcription_result = _apply_async_with_headers(process_transcription_task, args=[job_id, {"audio_path": str(audio_path)}], queue='transcription')
        transcription_data = transcription_result.get(timeout=1800)  # 30 minutes
        
        return {
            "job_id": job_id,
            "audio_path": str(audio_path),
            "transcription": transcription_data,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Audio processing failed", job_id=job_id, error=str(e))
        raise


@app.task(bind=True, base=DatabaseTask, name='video_processor.tasks.process_transcription')
@retry_on_failure(max_attempts=2, base_delay=1.0)
def process_transcription_task(self, job_id: str, transcription_config: Dict[str, Any]) -> Dict[str, Any]:
    """Process audio transcription."""
    try:
        logger.info(f"Processing transcription", job_id=job_id)
        
        # Add processing step
        step = self.job_manager.add_processing_step(job_id, "transcription", 3, ProcessingStatus.PROCESSING)
        
        from src.video_doc.transcribe import transcribe_audio
        
        audio_path = Path(transcription_config['audio_path'])
        output_dir = audio_path.parent
        transcript_txt = output_dir / "transcript" / "transcript.txt"
        segments_json = output_dir / "transcript" / "segments.json"
        
        # Ensure directories exist
        transcript_txt.parent.mkdir(parents=True, exist_ok=True)
        
        # Transcribe audio
        segments = transcribe_audio(
            audio_path=audio_path,
            transcript_txt_path=transcript_txt,
            segments_json_path=segments_json,
            language=transcription_config.get('language', 'auto'),
            beam_size=transcription_config.get('beam_size', 5),
            model_size=transcription_config.get('whisper_model', 'medium')
        )
        
        # Update step status
        self.job_manager.update_step_status(str(step.id), ProcessingStatus.COMPLETED)
        
        return {
            "job_id": job_id,
            "segments": len(segments),
            "transcript_path": str(transcript_txt),
            "segments_path": str(segments_json),
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Transcription processing failed", job_id=job_id, error=str(e))
        raise


@app.task(bind=True, base=DatabaseTask, name='video_processor.tasks.process_frames')
@retry_on_failure(max_attempts=2, base_delay=1.0)
def process_frames_task(self, job_id: str, frames_config: Dict[str, Any]) -> Dict[str, Any]:
    """Process frame extraction and classification."""
    try:
        logger.info(f"Processing frames", job_id=job_id)
        
        # Add processing step
        step = self.job_manager.add_processing_step(job_id, "frame_extraction", 4, ProcessingStatus.PROCESSING)
        
        from src.video_doc.frames import extract_keyframes
        from src.video_doc.classify import classify_frames
        
        video_path = Path(frames_config['video_path'])
        output_dir = video_path.parent
        frames_dir = output_dir / "frames" / "keyframes"
        classified_dir = output_dir / "classified"
        snippets_dir = output_dir / "snippets" / "code"
        
        # Ensure directories exist
        frames_dir.mkdir(parents=True, exist_ok=True)
        classified_dir.mkdir(parents=True, exist_ok=True)
        snippets_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract keyframes
        keyframe_paths = extract_keyframes(
            video_path=video_path,
            output_dir=frames_dir,
            max_fps=frames_config.get('max_fps', 1.0),
            scene_threshold=frames_config.get('min_scene_diff', 0.45),
            method=frames_config.get('kf_method', 'scene')
        )
        
        # Update step status
        self.job_manager.update_step_status(str(step.id), ProcessingStatus.COMPLETED)
        
        # Process classification
        classification_result = _apply_async_with_headers(process_classification_task, args=[job_id, {
            "keyframe_paths": [str(p) for p in keyframe_paths],
            "classified_dir": str(classified_dir),
            "snippets_dir": str(snippets_dir)
        }], queue='classification')
        classification_data = classification_result.get(timeout=1800)  # 30 minutes
        
        return {
            "job_id": job_id,
            "keyframes": len(keyframe_paths),
            "classification": classification_data,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Frame processing failed", job_id=job_id, error=str(e))
        raise


@app.task(bind=True, base=DatabaseTask, name='video_processor.tasks.process_classification')
@retry_on_failure(max_attempts=2, base_delay=1.0)
def process_classification_task(self, job_id: str, classification_config: Dict[str, Any]) -> Dict[str, Any]:
    """Process frame classification."""
    try:
        logger.info(f"Processing classification", job_id=job_id)
        
        # Add processing step
        step = self.job_manager.add_processing_step(job_id, "classification", 5, ProcessingStatus.PROCESSING)
        
        from src.video_doc.classify import classify_frames
        
        keyframe_paths = [Path(p) for p in classification_config['keyframe_paths']]
        classified_dir = Path(classification_config['classified_dir'])
        snippets_dir = Path(classification_config['snippets_dir'])
        
        # Classify frames
        classification_result = classify_frames(
            frame_paths=keyframe_paths,
            classified_root=classified_dir,
            snippets_dir=snippets_dir,
            ocr_languages=['en'],
            skip_blurry=True,
            blurry_threshold=60.0
        )
        
        # Update step status
        self.job_manager.update_step_status(str(step.id), ProcessingStatus.COMPLETED)
        
        return {
            "job_id": job_id,
            "classified_frames": sum(len(frames) for frames in classification_result.values()),
            "categories": {k: len(v) for k, v in classification_result.items()},
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Classification processing failed", job_id=job_id, error=str(e))
        raise


@app.task(bind=True, base=DatabaseTask, name='video_processor.tasks.generate_pdf')
@retry_on_failure(max_attempts=2, base_delay=1.0)
def generate_pdf_task(self, job_id: str, pdf_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final PDF report."""
    try:
        logger.info(f"Generating PDF", job_id=job_id)
        
        # Add processing step
        step = self.job_manager.add_processing_step(job_id, "pdf_generation", 6, ProcessingStatus.PROCESSING)
        
        from src.video_doc.pdf import build_pdf_report
        
        output_dir = Path(pdf_config['output_dir'])
        report_pdf_path = output_dir / "report.pdf"
        transcript_txt = output_dir / "transcript" / "transcript.txt"
        segments_json = output_dir / "transcript" / "segments.json"
        
        # Build PDF report
        build_pdf_report(
            output_pdf_path=report_pdf_path,
            transcript_txt_path=transcript_txt,
            segments_json_path=segments_json,
            classification_result=pdf_config.get('classification_result', {"code": [], "plots": [], "images": []}),
            output_dir=output_dir,
            report_style=pdf_config.get('report_style', 'book'),
            video_title=pdf_config.get('video_title', 'Video Report')
        )
        
        # Update step status
        self.job_manager.update_step_status(str(step.id), ProcessingStatus.COMPLETED)
        
        return {
            "job_id": job_id,
            "pdf_path": str(report_pdf_path),
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"PDF generation failed", job_id=job_id, error=str(e))
        raise


# Celery signal handlers
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal."""
    logger.info(f"Celery worker {sender} is ready")


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task prerun signal."""
    logger.info(f"Task {task_id} starting", task_name=task.name)
    try:
        # Queue wait time
        enq = None
        if hasattr(task, 'request'):
            headers = getattr(task.request, 'headers', None) or {}
            if isinstance(headers, dict):
                enq = headers.get('enqueued_at')
        if enq is not None:
            wait_s = max(0.0, time.time() - float(enq))
            queue_name = 'unknown'
            try:
                delivery = getattr(task.request, 'delivery_info', {}) or {}
                queue_name = delivery.get('routing_key') or delivery.get('exchange') or 'default'
            except Exception:
                pass
            metrics.observe_task_queue_wait(task.name if task else 'unknown', queue_name, wait_s)
        # Mark start time for duration
        if hasattr(task, 'request'):
            setattr(task.request, '_metrics_start_time', time.time())
    except Exception:
        pass


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handle task postrun signal."""
    logger.info(f"Task {task_id} completed", task_name=task.name, state=state)
    try:
        start = None
        queue_name = 'unknown'
        if hasattr(task, 'request'):
            start = getattr(task.request, '_metrics_start_time', None)
            try:
                delivery = getattr(task.request, 'delivery_info', {}) or {}
                queue_name = delivery.get('routing_key') or delivery.get('exchange') or 'default'
            except Exception:
                pass
        if start is not None:
            duration = max(0.0, time.time() - float(start))
            status = 'success' if (state or '').upper() == 'SUCCESS' else 'failure'
            metrics.observe_task_duration(task.name if task else 'unknown', queue_name, status, duration)
    except Exception:
        pass


# Utility functions
def get_celery_app() -> Celery:
    """Get Celery app instance."""
    return app


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status."""
    try:
        result = app.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.successful() else None,
            "error": str(result.result) if result.failed() else None
        }
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        return {"task_id": task_id, "status": "UNKNOWN", "error": str(e)}


def cancel_task(task_id: str) -> bool:
    """Cancel a task."""
    try:
        app.control.revoke(task_id, terminate=True)
        logger.info(f"Task {task_id} cancelled")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        return False


def get_queue_stats() -> Dict[str, Any]:
    """Get queue statistics."""
    try:
        inspect = app.control.inspect()
        
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()
        reserved_tasks = inspect.reserved()
        
        return {
            "active_tasks": active_tasks,
            "scheduled_tasks": scheduled_tasks,
            "reserved_tasks": reserved_tasks,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return {"error": str(e)}


if __name__ == '__main__':
    # Start Celery worker
    app.start()
