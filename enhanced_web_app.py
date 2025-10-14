"""
Enhanced web application with data engineering features.
"""

import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid
import re
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_socketio import SocketIO, emit
import yaml

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
    log_error, log_quality_check, audit_logger, health_monitor
)
from src.video_doc.data_validation import (
    data_quality_manager, validate_processing_job, log_validation_results
)

# Import enhanced processor
from enhanced_main import EnhancedVideoProcessor

# Import health check functionality
from src.video_doc.health_checks import get_health_status, get_health_summary, get_service_health

# Import authentication functionality
from src.video_doc.flask_auth import init_auth_system, require_auth, optional_auth, get_current_user_session
from src.video_doc.auth import Permission

logger = get_logger(__name__)

# Configuration
class Config:
    """Application configuration with environment variable support."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'video-doc-secret-key-2024')
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', 'web_outputs')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE', 2 * 1024 * 1024 * 1024))  # 2GB default
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # Database configuration
    DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://video_doc:password@localhost:5432/video_doc_db')
    
    # File validation
    ALLOWED_EXTENSIONS = {
        'video': {'mp4', 'mkv', 'mov', 'webm', 'avi', 'wmv'},
        'audio': {'mp3', 'm4a', 'wav', 'flac', 'aac', 'ogg'}
    }
    
    # Processing limits
    MAX_CONCURRENT_JOBS = int(os.environ.get('MAX_CONCURRENT_JOBS', 3))
    JOB_TIMEOUT = int(os.environ.get('JOB_TIMEOUT', 3600))  # 1 hour

app = Flask(__name__)
app.config.from_object(Config)

# Initialize authentication system
init_auth_system(app)

socketio = SocketIO(app, cors_allowed_origins="*")

# Global storage for processing jobs
processing_jobs: Dict[str, 'EnhancedProcessingJob'] = {}
job_lock = threading.Lock()

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# Validation utilities
class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_file_upload(file) -> None:
    """Validate uploaded file."""
    if not file or file.filename == '':
        raise ValidationError("No file provided")
    
    # Check file extension
    filename = file.filename.lower()
    file_ext = filename.split('.')[-1] if '.' in filename else ''
    
    allowed_exts = Config.ALLOWED_EXTENSIONS['video'] | Config.ALLOWED_EXTENSIONS['audio']
    if file_ext not in allowed_exts:
        raise ValidationError(f"Unsupported file type. Allowed: {', '.join(allowed_exts)}")
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        max_size_gb = app.config['MAX_CONTENT_LENGTH'] / (1024**3)
        raise ValidationError(f"File size exceeds {max_size_gb:.1f}GB limit")

def validate_url(url: str) -> None:
    """Validate video URL."""
    if not url or not url.strip():
        raise ValidationError("No URL provided")
    
    try:
        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            raise ValidationError("Invalid URL format")
        
        # Basic check for common video platforms
        domain = parsed.netloc.lower()
        if not any(platform in domain for platform in ['youtube.com', 'youtu.be', 'vimeo.com', 'twitch.tv']):
            logger.warning(f"Unrecognized video platform: {domain}")
            
    except Exception as e:
        raise ValidationError(f"Invalid URL: {str(e)}")

def validate_processing_options(options: Dict[str, Any]) -> None:
    """Validate processing options."""
    # Language validation
    valid_languages = {'auto', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'}
    if options.get('language') not in valid_languages:
        raise ValidationError(f"Invalid language. Must be one of: {', '.join(valid_languages)}")
    
    # Whisper model validation
    valid_models = {'tiny', 'base', 'small', 'medium', 'large'}
    if options.get('whisper_model') not in valid_models:
        raise ValidationError(f"Invalid whisper model. Must be one of: {', '.join(valid_models)}")
    
    # Beam size validation
    beam_size = options.get('beam_size', 5)
    if not isinstance(beam_size, int) or beam_size < 1 or beam_size > 10:
        raise ValidationError("Beam size must be an integer between 1 and 10")
    
    # Keyframe method validation
    valid_methods = {'scene', 'iframe', 'interval'}
    if options.get('kf_method') not in valid_methods:
        raise ValidationError(f"Invalid keyframe method. Must be one of: {', '.join(valid_methods)}")
    
    # Report style validation
    valid_styles = {'minimal', 'book'}
    if options.get('report_style') not in valid_styles:
        raise ValidationError(f"Invalid report style. Must be one of: {', '.join(valid_styles)}")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200-len(ext)] + ext
    return filename


class EnhancedProcessingJob:
    """Enhanced processing job with database integration."""
    
    def __init__(self, job_id: str, job_type: str, identifier: str, options: Dict[str, Any]):
        self.job_id = job_id
        self.job_type = job_type
        self.identifier = identifier
        self.options = options
        self.status = 'pending'
        self.progress = 0.0
        self.current_step = ''
        self.error_message = ''
        self.start_time = None
        self.end_time = None
        self.output_files = {}
        self.thread = None
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Database session
        self.db_session = get_db_session()
        self.job_manager = JobManager(self.db_session)
        
        # Validate options
        try:
            validate_processing_options(options)
        except ValidationError as e:
            logger.error(f"Invalid options for job {job_id}: {e}")
            raise
        
    def start_processing(self) -> None:
        """Start processing in a separate thread."""
        if self.status != 'pending':
            raise ValueError(f"Cannot start job {self.job_id} with status {self.status}")
        
        # Check concurrent job limit
        with job_lock:
            active_jobs = sum(1 for job in processing_jobs.values() if job.status == 'processing')
            if active_jobs >= Config.MAX_CONCURRENT_JOBS:
                raise ValueError(f"Maximum concurrent jobs ({Config.MAX_CONCURRENT_JOBS}) reached")
        
        self.thread = threading.Thread(target=self._process_video, name=f"Job-{self.job_id}")
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started processing job {self.job_id}")
    
    def cancel(self) -> bool:
        """Cancel the processing job if possible."""
        if self.status in ['completed', 'failed']:
            return False
        
        self.status = 'cancelled'
        self.end_time = time.time()
        self._emit_status_update()
        logger.info(f"Cancelled job {self.job_id}")
        return True
    
    def cleanup(self) -> None:
        """Clean up job resources."""
        try:
            # Clean up output directory if job failed or was cancelled
            if self.status in ['failed', 'cancelled']:
                output_dir = Path(app.config['OUTPUT_FOLDER']) / self.job_id
                if output_dir.exists():
                    import shutil
                    shutil.rmtree(output_dir, ignore_errors=True)
                    logger.info(f"Cleaned up output directory for job {self.job_id}")
            
            # Close database session
            if self.db_session:
                self.db_session.close()
                
        except Exception as e:
            logger.error(f"Error cleaning up job {self.job_id}: {e}")
    
    def is_stale(self) -> bool:
        """Check if job is stale (no activity for too long)."""
        if self.status == 'processing':
            return (datetime.now() - self.last_activity).seconds > Config.JOB_TIMEOUT
        return False
        
    def _process_video(self):
        """Process the video using enhanced pipeline."""
        try:
            self.status = 'processing'
            self.start_time = time.time()
            self._emit_status_update()
            
            # Create enhanced processor
            with EnhancedVideoProcessor(self.job_id) as processor:
                # Create job in database
                job = processor.create_job(self.job_type, self.identifier, self.options)
                
                # Prepare arguments for processing
                args = self._prepare_arguments()
                
                # Process the video
                result = processor.process_video(args)
                
                # Update job status
                self.status = 'completed'
                self.progress = 100.0
                self.end_time = time.time()
                self.output_files = result.get('output_files', {})
                self._emit_status_update()
                
                logger.info(f"Job {self.job_id} completed successfully")
                
        except Exception as e:
            self.status = 'failed'
            self.error_message = str(e)
            self.end_time = time.time()
            self.last_activity = datetime.now()
            self._emit_status_update()
            logger.error(f"Processing error for job {self.job_id}: {e}", exc_info=True)
            
            # Clean up on failure
            try:
                self.cleanup()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup for job {self.job_id}: {cleanup_error}")
    
    def _prepare_arguments(self):
        """Prepare arguments for the enhanced processor."""
        class MockArgs:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        output_dir = Path(app.config['OUTPUT_FOLDER']) / self.job_id
        
        if self.job_type == 'url':
            return MockArgs(
                url=self.identifier,
                video=None,
                out=str(output_dir),
                language=self.options.get('language', 'auto'),
                beam_size=self.options.get('beam_size', 5),
                whisper_model=self.options.get('whisper_model', 'medium'),
                transcribe_only=self.options.get('transcribe_only', False),
                streaming=self.options.get('streaming', False),
                kf_method=self.options.get('kf_method', 'scene'),
                max_fps=self.options.get('max_fps', 1.0),
                min_scene_diff=self.options.get('min_scene_diff', 0.45),
                report_style=self.options.get('report_style', 'book'),
                resume=False,
                # Additional args with defaults
                whisper_cpu_threads=0,
                whisper_num_workers=1,
                kf_interval_sec=5.0,
                frame_format='jpg',
                frame_quality=90,
                frame_max_width=1280,
                frame_max_frames=0,
                skip_dark=False,
                dark_value=16,
                dark_ratio=0.98,
                dedupe=False,
                dedupe_sim=0.995,
                skip_download=False,
                export_srt=False,
                cookies_from_browser=None,
                browser_profile=None,
                cookies_file=None,
                use_android_client=False,
                trim_start=0.0,
                trim_end=0.0,
                volume_gain=0.0,
                ocr_langs='en',
                skip_blurry=False,
                blurry_threshold=60.0,
                max_per_category=0
            )
        else:  # file upload
            return MockArgs(
                url=None,
                video=self.identifier,
                out=str(output_dir),
                language=self.options.get('language', 'auto'),
                beam_size=self.options.get('beam_size', 5),
                whisper_model=self.options.get('whisper_model', 'medium'),
                transcribe_only=self.options.get('transcribe_only', False),
                streaming=False,
                kf_method=self.options.get('kf_method', 'scene'),
                max_fps=self.options.get('max_fps', 1.0),
                min_scene_diff=self.options.get('min_scene_diff', 0.45),
                report_style=self.options.get('report_style', 'book'),
                resume=False,
                # Additional args with defaults
                whisper_cpu_threads=0,
                whisper_num_workers=1,
                kf_interval_sec=5.0,
                frame_format='jpg',
                frame_quality=90,
                frame_max_width=1280,
                frame_max_frames=0,
                skip_dark=False,
                dark_value=16,
                dark_ratio=0.98,
                dedupe=False,
                dedupe_sim=0.995,
                skip_download=False,
                export_srt=False,
                cookies_from_browser=None,
                browser_profile=None,
                cookies_file=None,
                use_android_client=False,
                trim_start=0.0,
                trim_end=0.0,
                volume_gain=0.0,
                ocr_langs='en',
                skip_blurry=False,
                blurry_threshold=60.0,
                max_per_category=0
            )
    
    def _emit_status_update(self):
        """Emit status update via WebSocket."""
        socketio.emit('job_status', {
            'job_id': self.job_id,
            'status': self.status,
            'progress': self.progress,
            'current_step': self.current_step,
            'error_message': self.error_message,
            'duration': self.end_time - self.start_time if self.end_time and self.start_time else None
        })
        
    def _emit_progress_update(self):
        """Emit progress update via WebSocket."""
        socketio.emit('job_progress', {
            'job_id': self.job_id,
            'progress': self.progress,
            'current_step': self.current_step
        })
        
    def _emit_step_update(self, step: str):
        """Emit step update via WebSocket."""
        self.current_step = step
        socketio.emit('job_step', {
            'job_id': self.job_id,
            'current_step': step
        })


# Health check endpoints
@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        # Check database health
        db_health = check_database_health()
        
        # Check overall system health
        overall_health = health_monitor.get_overall_health()
        
        # Get error statistics
        error_stats = error_handler.get_error_stats()
        
        health_status = {
            "status": "healthy" if db_health["status"] == "healthy" and overall_health == "healthy" else "unhealthy",
            "database": db_health,
            "system": overall_health,
            "errors": error_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503


@app.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint."""
    try:
        from flask import Response
        return Response(metrics.get_metrics(), mimetype='text/plain')
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        return jsonify({"error": str(e)}), 500


# Existing routes with enhanced error handling
@app.route('/')
@optional_auth
def index():
    """Main page with upload form."""
    user_session = get_current_user_session()
    if not user_session:
        return render_template('login.html')
    return render_template('index.html', user=user_session)


@app.route('/upload', methods=['POST'])
@with_error_handling("file_upload", ErrorSeverity.MEDIUM)
def upload_file():
    """Handle file upload with enhanced validation."""
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        validate_file_upload(file)
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Sanitize filename and save uploaded file
        original_filename = sanitize_filename(file.filename)
        filename = f"{job_id}_{original_filename}"
        file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        
        try:
            file.save(file_path)
            logger.info(f"Saved uploaded file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Get and validate processing options from form
        try:
            options = {
                'language': request.form.get('language', 'auto'),
                'whisper_model': request.form.get('whisper_model', 'medium'),
                'beam_size': int(request.form.get('beam_size', 5)),
                'transcribe_only': request.form.get('transcribe_only') == 'on',
                'streaming': False,  # Not applicable for file uploads
                'kf_method': request.form.get('kf_method', 'scene'),
                'max_fps': float(request.form.get('max_fps', 1.0)),
                'min_scene_diff': float(request.form.get('min_scene_diff', 0.45)),
                'report_style': request.form.get('report_style', 'book')
            }
            validate_processing_options(options)
        except (ValueError, ValidationError) as e:
            # Clean up uploaded file if validation fails
            try:
                file_path.unlink()
            except:
                pass
            return jsonify({'error': str(e)}), 400
        
        # Create processing job
        try:
            job = EnhancedProcessingJob(job_id, 'file', str(file_path), options)
            
            with job_lock:
                processing_jobs[job_id] = job
            
            # Start processing
            job.start_processing()
            
            logger.info(f"Started file processing job {job_id} for file {original_filename}")
            
            # Log audit event
            audit_logger.log_operation(
                operation="file_upload",
                resource_type="processing_job",
                resource_id=job_id,
                new_values={"filename": original_filename, "options": options}
            )
            
            return jsonify({
                'job_id': job_id,
                'status': 'started',
                'message': 'File uploaded and processing started'
            })
            
        except ValueError as e:
            # Clean up uploaded file if job creation fails
            try:
                file_path.unlink()
            except:
                pass
            return jsonify({'error': str(e)}), 400
            
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/process_url', methods=['POST'])
@with_error_handling("url_processing", ErrorSeverity.MEDIUM)
def process_url():
    """Handle URL processing with enhanced validation."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        url = data.get('url')
        validate_url(url)
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Get and validate processing options
        try:
            options = {
                'language': data.get('language', 'auto'),
                'whisper_model': data.get('whisper_model', 'medium'),
                'beam_size': int(data.get('beam_size', 5)),
                'transcribe_only': data.get('transcribe_only', False),
                'streaming': data.get('streaming', False),
                'kf_method': data.get('kf_method', 'scene'),
                'max_fps': float(data.get('max_fps', 1.0)),
                'min_scene_diff': float(data.get('min_scene_diff', 0.45)),
                'report_style': data.get('report_style', 'book')
            }
            validate_processing_options(options)
        except (ValueError, ValidationError) as e:
            return jsonify({'error': str(e)}), 400
        
        # Create processing job
        try:
            job = EnhancedProcessingJob(job_id, 'url', url, options)
            
            with job_lock:
                processing_jobs[job_id] = job
            
            # Start processing
            job.start_processing()
            
            logger.info(f"Started URL processing job {job_id} for URL {url}")
            
            # Log audit event
            audit_logger.log_operation(
                operation="url_processing",
                resource_type="processing_job",
                resource_id=job_id,
                new_values={"url": url, "options": options}
            )
            
            return jsonify({
                'job_id': job_id,
                'status': 'started',
                'message': 'URL processing started'
            })
            
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
            
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in process_url: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


# Additional enhanced endpoints
@app.route('/jobs')
def list_jobs():
    """List all processing jobs with database integration."""
    try:
        with job_lock:
            jobs = []
            for job_id, job in processing_jobs.items():
                jobs.append({
                    'job_id': job_id,
                    'job_type': job.job_type,
                    'identifier': job.identifier,
                    'status': job.status,
                    'progress': job.progress,
                    'current_step': job.current_step,
                    'start_time': job.start_time,
                    'end_time': job.end_time,
                    'duration': job.end_time - job.start_time if job.end_time and job.start_time else None,
                    'created_at': job.created_at.isoformat(),
                    'last_activity': job.last_activity.isoformat()
                })
        
        return jsonify({'jobs': jobs})
    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        return jsonify({'error': 'Failed to list jobs'}), 500


@app.route('/job/<job_id>/quality')
def get_job_quality(job_id):
    """Get quality check results for a job."""
    try:
        db = get_db_session()
        try:
            quality_checks = db.query(QualityCheck).filter(
                QualityCheck.job_id == job_id
            ).all()
            
            results = []
            for check in quality_checks:
                results.append({
                    'check_name': check.check_name,
                    'check_type': check.check_type,
                    'status': check.status,
                    'expected_value': check.expected_value,
                    'actual_value': check.actual_value,
                    'threshold': check.threshold,
                    'message': check.message,
                    'executed_at': check.executed_at.isoformat()
                })
            
            return jsonify({'quality_checks': results})
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting quality checks for job {job_id}: {e}")
        return jsonify({'error': 'Failed to get quality checks'}), 500


# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    try:
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'message': 'Connected to enhanced video processing server'})
    except Exception as e:
        logger.error(f"Error handling client connection: {e}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    try:
        logger.info(f"Client disconnected: {request.sid}")
    except Exception as e:
        logger.error(f"Error handling client disconnection: {e}")


@socketio.on_error_default
def default_error_handler(e):
    """Default error handler for WebSocket events."""
    logger.error(f"WebSocket error: {e}", exc_info=True)
    emit('error', {'message': 'An error occurred'})


# Error handlers for Flask routes
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(400)
def bad_request(e):
    """Handle bad request error."""
    return jsonify({'error': 'Bad request'}), 400


@app.errorhandler(404)
def not_found(e):
    """Handle not found error."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    logger.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500


# Health Check Endpoints
@app.route('/health')
def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_data = get_health_status()
        
        # Set appropriate HTTP status code
        status_code = 200
        if health_data['status'] == 'unhealthy':
            status_code = 503
        elif health_data['status'] == 'degraded':
            status_code = 200  # Still operational but with warnings
        
        return jsonify(health_data), status_code
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': 'Health check failed',
            'error_message': str(e)
        }), 503


@app.route('/health/summary')
def health_summary():
    """Simplified health summary endpoint."""
    try:
        summary = get_health_summary()
        
        # Set appropriate HTTP status code
        status_code = 200
        if summary['status'] == 'unhealthy':
            status_code = 503
        elif summary['status'] == 'degraded':
            status_code = 200
        
        return jsonify(summary), status_code
    except Exception as e:
        logger.error(f"Health summary failed: {e}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': 'Health summary failed',
            'error_message': str(e)
        }), 503


@app.route('/health/service/<service_name>')
def service_health(service_name):
    """Get health status for a specific service."""
    try:
        service_data = get_service_health(service_name)
        
        if service_data is None:
            return jsonify({'error': 'Service not found'}), 404
        
        # Set appropriate HTTP status code
        status_code = 200
        if service_data['status'] == 'unhealthy':
            status_code = 503
        elif service_data['status'] == 'degraded':
            status_code = 200
        
        return jsonify(service_data), status_code
    except Exception as e:
        logger.error(f"Service health check failed for {service_name}: {e}", exc_info=True)
        return jsonify({
            'error': 'Service health check failed',
            'error_message': str(e)
        }), 500


@app.route('/health/live')
def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    try:
        # Simple check - if the app is running, it's alive
        return jsonify({
            'status': 'alive',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - app.config.get('start_time', time.time())
        }), 200
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}", exc_info=True)
        return jsonify({
            'status': 'dead',
            'error': str(e)
        }), 503


@app.route('/health/ready')
def readiness_probe():
    """Kubernetes readiness probe endpoint."""
    try:
        # Check if the app is ready to serve requests
        summary = get_health_summary()
        
        # App is ready if core services are healthy
        core_services = ['database', 'file_system']
        ready = True
        
        for service_name in core_services:
            service_data = get_service_health(service_name)
            if service_data and service_data['status'] == 'unhealthy':
                ready = False
                break
        
        if ready:
            return jsonify({
                'status': 'ready',
                'timestamp': datetime.now().isoformat(),
                'summary': summary
            }), 200
        else:
            return jsonify({
                'status': 'not_ready',
                'timestamp': datetime.now().isoformat(),
                'summary': summary
            }), 503
            
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}", exc_info=True)
        return jsonify({
            'status': 'not_ready',
            'error': str(e)
        }), 503


@app.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint."""
    try:
        from src.video_doc.monitoring import metrics
        from prometheus_client.exposition import CONTENT_TYPE_LATEST
        
        metrics_data = metrics.get_metrics()
        return metrics_data, 200, {'Content-Type': CONTENT_TYPE_LATEST}
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}", exc_info=True)
        return jsonify({'error': 'Metrics unavailable'}), 500


@app.route('/health-dashboard')
def health_dashboard():
    """Health monitoring dashboard."""
    return render_template('health_dashboard.html')


@app.route('/user-management')
@require_auth()
def user_management():
    """User management dashboard."""
    return render_template('user_management.html')


if __name__ == '__main__':
    logger.info("Starting Enhanced Video Documentation Builder Web Interface...")
    logger.info(f"Configuration: DEBUG={Config.DEBUG}, HOST={Config.HOST}, PORT={Config.PORT}")
    logger.info(f"Database URL: {Config.DATABASE_URL}")
    logger.info(f"Max file size: {Config.MAX_CONTENT_LENGTH / (1024**3):.1f}GB")
    logger.info(f"Max concurrent jobs: {Config.MAX_CONCURRENT_JOBS}")
    logger.info(f"Job timeout: {Config.JOB_TIMEOUT}s")
    
    # Check database health before starting
    db_health = check_database_health()
    if db_health["status"] != "healthy":
        logger.error("Database is not healthy, cannot start server", health=db_health)
        sys.exit(1)
    
    print("Starting Enhanced Video Documentation Builder Web Interface...")
    print(f"Open your browser and go to: http://{Config.HOST}:{Config.PORT}")
    print(f"Login: http://{Config.HOST}:{Config.PORT}/")
    print(f"User Management: http://{Config.HOST}:{Config.PORT}/user-management")
    print(f"Health Dashboard: http://{Config.HOST}:{Config.PORT}/health-dashboard")
    print(f"Health Check API: http://{Config.HOST}:{Config.PORT}/health")
    print(f"Health Summary: http://{Config.HOST}:{Config.PORT}/health/summary")
    print(f"Metrics: http://{Config.HOST}:{Config.PORT}/metrics")
    print("\nDefault Admin Credentials:")
    print("Username: admin")
    print("Password: admin123")
    
    try:
        socketio.run(app, debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise
