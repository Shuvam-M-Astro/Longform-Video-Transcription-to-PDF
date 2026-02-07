#!/usr/bin/env python3
"""
Web-based GUI for Video Documentation Builder

A Flask web application that provides a user-friendly interface for the video
transcription tool, allowing users to upload videos, configure options, and
download results through a web browser.

Usage:
    python web_app.py
    Then open http://localhost:5000 in your browser
"""

import os
import json
import logging
import threading
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import asdict
import uuid
import re
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_socketio import SocketIO, emit
import yaml

# Import health check functionality
from src.video_doc.health_checks import get_health_status, get_health_summary, get_service_health

# Import authentication functionality
from src.video_doc.flask_auth import init_auth_system, require_auth, optional_auth, get_current_user_session
from src.video_doc.auth import Permission
from src.video_doc.security_enhancements import security_manager, password_policy
from src.video_doc.user_management import user_manager, session_manager
from src.video_doc.enhanced_api_docs import create_api_docs_blueprint

# Import database functionality
from src.video_doc.database import get_db_session, ProcessingJob as DBProcessingJob, ProcessingStatus, JobManager, ProcessingPreset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    
    # File validation
    ALLOWED_EXTENSIONS = {
        'video': {'mp4', 'mkv', 'mov', 'webm', 'avi', 'wmv'},
        'audio': {'mp3', 'm4a', 'wav', 'flac', 'aac', 'ogg'}
    }
    
    # Processing limits
    MAX_CONCURRENT_JOBS = int(os.environ.get('MAX_CONCURRENT_JOBS', 3))
    JOB_TIMEOUT = int(os.environ.get('JOB_TIMEOUT', 3600))  # 1 hour
    
    # Cleanup configuration
    CLEANUP_INTERVAL = int(os.environ.get('CLEANUP_INTERVAL', 3600))  # Run cleanup every hour
    JOB_RETENTION_DAYS = int(os.environ.get('JOB_RETENTION_DAYS', 7))  # Keep completed jobs for 7 days
    FAILED_JOB_RETENTION_DAYS = int(os.environ.get('FAILED_JOB_RETENTION_DAYS', 1))  # Keep failed jobs for 1 day
    UPLOAD_RETENTION_DAYS = int(os.environ.get('UPLOAD_RETENTION_DAYS', 1))  # Keep uploaded files for 1 day

app = Flask(__name__)
app.config.from_object(Config)

# Initialize authentication system
init_auth_system(app)

# Register API documentation blueprint
api_docs_bp = create_api_docs_blueprint()
app.register_blueprint(api_docs_bp)

socketio = SocketIO(app, cors_allowed_origins="*")

# Global storage for processing jobs
processing_jobs: Dict[str, 'ProcessingJob'] = {}
job_lock = threading.Lock()

# Global storage for batch jobs
batch_jobs: Dict[str, 'BatchJob'] = {}
batch_lock = threading.Lock()

# Cleanup thread
cleanup_thread: Optional[threading.Thread] = None
cleanup_running = False

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
    valid_models = {'tiny', 'base', 'small', 'medium', 'large', 'large-v3'}
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


class ProcessingJob:
    """Represents a video processing job with real-time updates."""
    
    def __init__(self, job_id: str, job_type: str, identifier: str, options: Dict[str, Any]):
        self.job_id = job_id
        self.job_type = job_type  # 'url' or 'file'
        self.identifier = identifier
        self.options = options
        self.status = 'pending'  # pending, processing, completed, failed
        self.progress = 0.0
        self.current_step = ''
        self.error_message = ''
        self.start_time = None
        self.end_time = None
        self.output_files = {}
        self.thread = None
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.db_job_id = None  # Database job ID if persisted
        
        # Validate options
        try:
            validate_processing_options(options)
        except ValidationError as e:
            logger.error(f"Invalid options for job {job_id}: {e}")
            raise
        
        # Persist to database
        self._persist_to_db()
        
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
        except Exception as e:
            logger.error(f"Error cleaning up job {self.job_id}: {e}")
    
    def is_stale(self) -> bool:
        """Check if job is stale (no activity for too long)."""
        if self.status == 'processing':
            return (datetime.now() - self.last_activity).seconds > Config.JOB_TIMEOUT
        return False
        
    def _process_video(self):
        """Process the video with progress callbacks."""
        try:
            self.status = 'processing'
            self.start_time = time.time()
            self._update_db_status()
            self._emit_status_update()
            
            # Prepare arguments for main processing
            args = self._prepare_arguments()
            
            # Process the video using the actual main function
            self._emit_step_update("Starting video processing...")
            
            # Import the main processing modules
            from main import (
                ensure_dirs, 
                download_video, extract_audio_wav, transcribe_audio,
                extract_keyframes, classify_frames, build_pdf_report,
                resolve_stream_urls, stream_extract_audio, stream_extract_keyframes,
                fallback_download_audio_via_ytdlp, fallback_download_small_video
            )
            from src.video_doc.progress import PipelineProgress
            from yt_dlp import YoutubeDL
            import json
            import shutil
            
            # Create a custom progress tracker
            class WebProgressTracker:
                def __init__(self, job_instance):
                    self.job = job_instance
                    self.current_step = ""
                    self.step_progress = 0.0
                    self.total_progress = 0.0
                    self.step_weights = {
                        'download': 20, 'audio': 10, 'frames': 20, 
                        'transcribe': 40, 'classify': 5, 'pdf': 5
                    }
                    self.completed_weight = 0.0
                    self.step_start_time = None
                    
                def start_step(self, step_name: str, weight: float):
                    try:
                        self.current_step = step_name
                        self.step_progress = 0.0
                        self.step_start_time = time.time()
                        self.job.current_step = step_name
                        self.job.last_activity = datetime.now()
                        self.job._emit_step_update(step_name)
                        logger.info(f"Job {self.job.job_id}: Started step '{step_name}'")
                    except Exception as e:
                        logger.error(f"Error in start_step: {e}")
                    
                def update(self, progress: float):
                    try:
                        # Validate progress value
                        progress = max(0.0, min(100.0, float(progress)))
                        
                        self.step_progress = progress
                        # Calculate total progress based on step weights
                        step_key = self.current_step.lower().split()[0]
                        step_weight = self.step_weights.get(step_key, 10)
                        current_weight_progress = (progress / 100.0) * step_weight
                        self.total_progress = min(95.0, self.completed_weight + current_weight_progress)
                        
                        self.job.progress = self.total_progress
                        self.job.last_activity = datetime.now()
                        self.job._emit_progress_update()
                    except Exception as e:
                        logger.error(f"Error in progress update: {e}")
                    
                def end_step(self):
                    try:
                        step_key = self.current_step.lower().split()[0]
                        step_weight = self.step_weights.get(step_key, 10)
                        self.completed_weight += step_weight
                        self.total_progress = min(95.0, self.completed_weight)
                        
                        self.job.progress = self.total_progress
                        self.job.last_activity = datetime.now()
                        self.job._emit_progress_update()
                        
                        if self.step_start_time:
                            duration = time.time() - self.step_start_time
                            logger.info(f"Job {self.job.job_id}: Completed step '{self.current_step}' in {duration:.1f}s")
                    except Exception as e:
                        logger.error(f"Error in end_step: {e}")
            
            # Create progress tracker
            progress_tracker = WebProgressTracker(self)
            
            # Execute the main processing logic
            output_dir = Path(args.out)
            
            # Clean output directory before running, unless resume is requested
            if output_dir.exists() and not getattr(args, "resume", False):
                self._emit_step_update("Cleaning output directory...")
                shutil.rmtree(output_dir, ignore_errors=True)
            
            ensure_dirs(output_dir)
            
            # Extract video title
            def _extract_video_title():
                if getattr(args, "video", None):
                    try:
                        stem = Path(args.video).stem
                        title = stem.replace("-", " ").replace("_", " ").strip()
                        return title.title() if title else "Video Report"
                    except Exception:
                        return "Video Report"
                
                def try_opts(use_cookies: bool, use_android: bool):
                    ydl_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
                    if use_cookies and args.cookies_from_browser:
                        tup = (args.cookies_from_browser,) if not args.browser_profile else (args.cookies_from_browser, args.browser_profile)
                        ydl_opts["cookiesfrombrowser"] = tup
                    if use_cookies and args.cookies_file:
                        ydl_opts["cookiefile"] = str(Path(args.cookies_file))
                    if use_android:
                        ydl_opts["extractor_args"] = {"youtube": {"player_client": ["android"]}}
                    with YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(args.url, download=False)
                        return str(info.get("title") or "")
                
                # Try different combinations
                for use_cookies, use_android in [(True, args.use_android_client), (False, args.use_android_client), (False, True)]:
                    try:
                        title = try_opts(use_cookies, use_android)
                        if title:
                            return title
                    except Exception:
                        pass
                
                # Fallback
                try:
                    from urllib.parse import urlparse, parse_qs
                    parsed = urlparse(args.url)
                    if parsed.netloc.lower().endswith("youtube.com"):
                        vid = parse_qs(parsed.query).get("v", [""])[0]
                        if vid:
                            return f"YouTube Video {vid}"
                    if parsed.path:
                        slug = Path(parsed.path).name.replace("-", " ").replace("_", " ")
                        slug = slug.strip() or "Video Report"
                        return slug.title()
                except Exception:
                    pass
                return "Video Report"
            
            video_title = _extract_video_title()
            
            # Determine source video path
            if getattr(args, "video", None):
                src_video_path = Path(args.video)
                if not src_video_path.exists():
                    raise FileNotFoundError(f"Local video not found: {src_video_path}")
                video_path = src_video_path
            else:
                video_path = output_dir / "video.mp4"
            
            audio_path = output_dir / "audio.wav"
            transcript_dir = output_dir / "transcript"
            frames_dir = output_dir / "frames" / "keyframes"
            classified_dir = output_dir / "classified"
            snippets_dir = output_dir / "snippets" / "code"
            
            keyframe_paths = []
            
            # Process based on mode
            if args.streaming:
                self._emit_step_update("Resolving stream URLs...")
                resolved = resolve_stream_urls(
                    args.url,
                    cookies_from_browser=args.cookies_from_browser,
                    browser_profile=args.browser_profile,
                    cookies_file=Path(args.cookies_file) if args.cookies_file else None,
                    use_android_client=args.use_android_client,
                )
                
                # Audio extraction
                try:
                    if not resolved.get("audio_url"):
                        raise RuntimeError("No audio_url")
                    self._emit_step_update("Extracting audio from stream...")
                    progress_tracker.start_step("Audio (stream)", 10)
                    stream_extract_audio(resolved["audio_url"], audio_path, headers=resolved.get("headers"), progress_cb=lambda p: progress_tracker.update(p))
                    progress_tracker.end_step()
                except Exception:
                    self._emit_step_update("Stream audio failed; falling back to yt-dlp audio-only...")
                    progress_tracker.start_step("Audio fallback download", 10)
                    audio_path = fallback_download_audio_via_ytdlp(
                        args.url, audio_path,
                        cookies_from_browser=args.cookies_from_browser,
                        browser_profile=args.browser_profile,
                        cookies_file=Path(args.cookies_file) if args.cookies_file else None,
                        use_android_client=args.use_android_client,
                    )
                    progress_tracker.end_step()
                
                # Keyframes (skip if transcribe-only)
                if not args.transcribe_only:
                    try:
                        if not resolved.get("video_url"):
                            raise RuntimeError("No video_url")
                        self._emit_step_update("Extracting keyframes from stream...")
                        progress_tracker.start_step("Keyframes (stream)", 20)
                        keyframe_paths = stream_extract_keyframes(
                            resolved["video_url"], frames_dir,
                            max_fps=args.max_fps, scene_threshold=args.min_scene_diff,
                            headers=resolved.get("headers"),
                            output_format=args.frame_format, jpeg_quality=args.frame_quality,
                            max_width=args.frame_max_width,
                            max_frames=(args.frame_max_frames if args.frame_max_frames > 0 else None),
                            progress_cb=lambda p: progress_tracker.update(p),
                        )
                        progress_tracker.end_step()
                    except Exception:
                        self._emit_step_update("Stream keyframes failed; downloading tiny MP4 for keyframes...")
                        small_video = output_dir / "video_small.mp4"
                        progress_tracker.start_step("Download tiny video", 10)
                        fallback_download_small_video(
                            args.url, small_video,
                            cookies_from_browser=args.cookies_from_browser,
                            browser_profile=args.browser_profile,
                            cookies_file=Path(args.cookies_file) if args.cookies_file else None,
                            use_android_client=args.use_android_client,
                        )
                        progress_tracker.end_step()
                        self._emit_step_update("Extracting keyframes from tiny MP4...")
                        progress_tracker.start_step("Keyframes (tiny)", 20)
                        keyframe_paths = extract_keyframes(
                            video_path=small_video, output_dir=frames_dir,
                            max_fps=args.max_fps, scene_threshold=args.min_scene_diff,
                            method=args.kf_method, interval_sec=args.kf_interval_sec,
                            progress_cb=lambda p: progress_tracker.update(p),
                        )
                        progress_tracker.end_step()
            else:
                # Non-streaming mode
                if getattr(args, "video", None):
                    self._emit_step_update(f"Using local video file: {video_path}")
                else:
                    if not args.skip_download and video_path.exists() and getattr(args, "resume", False):
                        self._emit_step_update(f"Reusing existing video: {video_path}")
                    elif not args.skip_download or not video_path.exists():
                        self._emit_step_update("Downloading full video (mp4)...")
                        progress_tracker.start_step("Download video", 20)
                        download_video(
                            args.url, video_path,
                            cookies_from_browser=args.cookies_from_browser,
                            browser_profile=args.browser_profile,
                            cookies_file=Path(args.cookies_file) if args.cookies_file else None,
                            use_android_client=args.use_android_client,
                            progress_cb=lambda p: progress_tracker.update(p),
                        )
                        progress_tracker.end_step()
                
                # Audio extraction
                if audio_path.exists() and getattr(args, "resume", False):
                    self._emit_step_update(f"Reusing existing audio: {audio_path}")
                else:
                    self._emit_step_update("Extracting audio from file...")
                    progress_tracker.start_step("Extract audio", 10)
                    extract_audio_wav(
                        video_path, audio_path,
                        progress_cb=lambda p: progress_tracker.update(p),
                        start_time=args.trim_start, end_trim=args.trim_end,
                        volume_gain_db=args.volume_gain,
                    )
                    progress_tracker.end_step()
                
                # Keyframes
                if not args.transcribe_only:
                    self._emit_step_update("Extracting keyframes from file...")
                    existing_frames = sorted(frames_dir.glob("frame_*.jpg"))
                    if existing_frames and getattr(args, "resume", False):
                        keyframe_paths = existing_frames
                        self._emit_step_update(f"Reusing existing keyframes: {len(keyframe_paths)}")
                    else:
                        progress_tracker.start_step("Keyframes", 20)
                        keyframe_paths = extract_keyframes(
                            video_path=video_path, output_dir=frames_dir,
                            max_fps=args.max_fps, scene_threshold=args.min_scene_diff,
                            method=args.kf_method, interval_sec=args.kf_interval_sec,
                            output_format=args.frame_format, jpeg_quality=args.frame_quality,
                            max_width=args.frame_max_width,
                            max_frames=(args.frame_max_frames if args.frame_max_frames > 0 else None),
                            skip_dark=args.skip_dark, dark_pixel_value=args.dark_value,
                            dark_ratio_threshold=args.dark_ratio, dedupe=args.dedupe,
                            dedupe_similarity=args.dedupe_sim,
                            progress_cb=lambda p: progress_tracker.update(p),
                        )
                        progress_tracker.end_step()
            
            # Transcription
            transcript_txt = transcript_dir / "transcript.txt"
            segments_json = transcript_dir / "segments.json"
            srt_path = transcript_dir / "transcript.srt" if args.export_srt else None
            
            if getattr(args, "resume", False) and transcript_txt.exists() and segments_json.exists() and (not args.export_srt or (srt_path and srt_path.exists())):
                self._emit_step_update("Reusing existing transcript artifacts")
                try:
                    with open(segments_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    segments = [
                        type("TS", (), {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": str(s.get("text", ""))})
                        for s in data if isinstance(s, dict)
                    ]
                except Exception:
                    segments = []
            else:
                self._emit_step_update(f"Transcribing audio with model={args.whisper_model}, beam_size={args.beam_size}...")
                progress_tracker.start_step("Transcribe", 40)
                segments = transcribe_audio(
                    audio_path=audio_path, transcript_txt_path=transcript_txt,
                    segments_json_path=segments_json, language=args.language,
                    beam_size=args.beam_size, model_size=args.whisper_model,
                    progress_cb=lambda p: progress_tracker.update(p),
                    cpu_threads=(args.whisper_cpu_threads if args.whisper_cpu_threads and args.whisper_cpu_threads > 0 else None),
                    num_workers=(args.whisper_num_workers if args.whisper_num_workers and args.whisper_num_workers > 0 else None),
                    srt_path=srt_path,
                )
                progress_tracker.end_step()
            
            # Classification
            if args.transcribe_only:
                classification_result = {"code": [], "plots": [], "images": []}
            else:
                self._emit_step_update("Classifying frames...")
                manifest_path = classified_dir / "classification_result.json"
                if getattr(args, "resume", False) and manifest_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            classification_result = json.load(f)
                        for k in ("code", "plots", "images"):
                            classification_result[k] = [Path(p) for p in classification_result.get(k, [])]
                        self._emit_step_update("Reusing existing classification manifest")
                    except Exception:
                        classification_result = {"code": [], "plots": [], "images": []}
                else:
                    progress_tracker.start_step("Classify frames", 5)
                    classification_result = classify_frames(
                        frame_paths=keyframe_paths, classified_root=classified_dir,
                        snippets_dir=snippets_dir, progress_cb=lambda p: progress_tracker.update(p),
                        ocr_languages=[lang.strip() for lang in str(args.ocr_langs).split(',') if lang.strip()],
                        skip_blurry=args.skip_blurry, blurry_threshold=args.blurry_threshold,
                        max_per_category=(args.max_per_category if args.max_per_category > 0 else None),
                    )
                    progress_tracker.end_step()
                    # Write manifest
                    try:
                        with open(manifest_path, "w", encoding="utf-8") as f:
                            json.dump({k: [str(p) for p in v] for k, v in classification_result.items()}, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
            
            # PDF Generation
            self._emit_step_update("Building PDF report...")
            progress_tracker.start_step("Build PDF", 5)
            report_pdf_path = output_dir / "report.pdf"
            
            # Detect contact sheet path
            contact_sheet_path = None
            try:
                possible_cs = (frames_dir.parent / "contact_sheet.jpg")
                contact_sheet_path = possible_cs if possible_cs.exists() else None
            except Exception:
                contact_sheet_path = None
            
            if getattr(args, "resume", False) and report_pdf_path.exists():
                self._emit_step_update(f"Reusing existing report: {report_pdf_path}")
                progress_tracker.update(100.0)
                progress_tracker.end_step()
            else:
                build_pdf_report(
                    output_pdf_path=report_pdf_path, transcript_txt_path=transcript_txt,
                    segments_json_path=segments_json, classification_result=classification_result,
                    output_dir=output_dir, progress_cb=lambda p: progress_tracker.update(p),
                    report_style=args.report_style, video_title=video_title,
                    contact_sheet_path=contact_sheet_path,
                )
                progress_tracker.end_step()
            
            # Mark as completed
            self.status = 'completed'
            self.progress = 100.0
            self.end_time = time.time()
            self._update_db_status()
            self._emit_status_update()
            
            # Automatically index transcript for search (async, non-blocking)
            try:
                from src.video_doc.search import get_search_service
                import threading
                
                def index_in_background():
                    try:
                        search_service = get_search_service()
                        success = search_service.index_transcript(
                            job_id=self.job_id,
                            segments_json_path=segments_json
                        )
                        if success:
                            logger.info(f"Successfully indexed transcript for job {self.job_id}")
                        else:
                            logger.warning(f"Failed to index transcript for job {self.job_id}")
                    except Exception as e:
                        logger.error(f"Error indexing transcript for job {self.job_id}: {e}")
                
                # Start indexing in background thread
                index_thread = threading.Thread(target=index_in_background, daemon=True)
                index_thread.start()
                logger.info(f"Started background indexing for job {self.job_id}")
            except Exception as e:
                logger.warning(f"Could not start indexing for job {self.job_id}: {e}")
            
        except Exception as e:
            self.status = 'failed'
            self.error_message = str(e)
            self.end_time = time.time()
            self.last_activity = datetime.now()
            self._update_db_status()
            self._emit_status_update()
            logger.error(f"Processing error for job {self.job_id}: {e}", exc_info=True)
            
            # Clean up on failure
            try:
                self.cleanup()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup for job {self.job_id}: {cleanup_error}")
            
    def _prepare_arguments(self):
        """Prepare arguments for the main processing function."""
        # Create a mock args object
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
    
    def _persist_to_db(self):
        """Persist job to database."""
        try:
            db = get_db_session()
            try:
                job_manager = JobManager(db)
                db_job = job_manager.create_job(
                    job_type=self.job_type,
                    identifier=self.identifier,
                    config=self.options
                )
                self.db_job_id = str(db_job.id)
                logger.info(f"Persisted job {self.job_id} to database as {self.db_job_id}")
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to persist job {self.job_id} to database: {e}")
            # Don't fail the job creation if DB persistence fails
    
    def _update_db_status(self):
        """Update job status in database."""
        if not self.db_job_id:
            return
        
        try:
            db = get_db_session()
            try:
                job_manager = JobManager(db)
                status_map = {
                    'pending': ProcessingStatus.PENDING,
                    'processing': ProcessingStatus.PROCESSING,
                    'completed': ProcessingStatus.COMPLETED,
                    'failed': ProcessingStatus.FAILED,
                    'cancelled': ProcessingStatus.CANCELLED
                }
                
                job_manager.update_job_status(
                    job_id=self.db_job_id,
                    status=status_map.get(self.status, ProcessingStatus.PENDING),
                    progress=self.progress,
                    current_step=self.current_step,
                    error_message=self.error_message if self.error_message else None
                )
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to update job {self.job_id} in database: {e}")


class BatchJob:
    """Represents a batch of video processing jobs."""
    
    def __init__(self, batch_id: str, options: Dict[str, Any]):
        self.batch_id = batch_id
        self.options = options
        self.status = 'pending'  # pending, processing, completed, failed, cancelled
        self.job_ids: List[str] = []
        self.completed_count = 0
        self.failed_count = 0
        self.total_count = 0
        self.created_at = datetime.now()
        self.start_time = None
        self.end_time = None
        self.error_message = ''
        
    def add_job(self, job_id: str):
        """Add a job to this batch."""
        if job_id not in self.job_ids:
            self.job_ids.append(job_id)
            self.total_count = len(self.job_ids)
    
    def update_status(self):
        """Update batch status based on individual job statuses."""
        with job_lock:
            completed = 0
            failed = 0
            processing = 0
            pending = 0
            
            for job_id in self.job_ids:
                if job_id in processing_jobs:
                    job = processing_jobs[job_id]
                    if job.status == 'completed':
                        completed += 1
                    elif job.status == 'failed':
                        failed += 1
                    elif job.status == 'processing':
                        processing += 1
                    else:
                        pending += 1
            
            self.completed_count = completed
            self.failed_count = failed
            
            if self.status == 'cancelled':
                return
            
            if completed + failed == self.total_count:
                self.status = 'completed' if failed == 0 else 'failed'
                self.end_time = time.time()
            elif processing > 0 or completed > 0 or failed > 0:
                self.status = 'processing'
                if not self.start_time:
                    self.start_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch job to dictionary."""
        self.update_status()
        return {
            'batch_id': self.batch_id,
            'status': self.status,
            'total_count': self.total_count,
            'completed_count': self.completed_count,
            'failed_count': self.failed_count,
            'created_at': self.created_at.isoformat(),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'job_ids': self.job_ids,
            'options': self.options
        }


@app.route('/')
@optional_auth
def index():
    """Main page with upload form."""
    user_session = get_current_user_session()
    if not user_session:
        return render_template('login.html')
    return render_template('index.html', user=user_session)


@app.route('/search')
@require_auth(Permission.VIEW_JOB)
def search_page():
    """Dedicated search page for transcript search."""
    user_session = get_current_user_session()
    return render_template('search.html', user=user_session)


@app.route('/jobs-dashboard')
@require_auth(Permission.VIEW_JOB)
def jobs_dashboard():
    """Jobs management dashboard."""
    user_session = get_current_user_session()
    return render_template('jobs_dashboard.html', user=user_session)


@app.route('/upload', methods=['POST'])
@require_auth(Permission.UPLOAD_FILES)
def upload_file():
    """Handle file upload."""
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
            job = ProcessingJob(job_id, 'file', str(file_path), options)
            
            with job_lock:
                processing_jobs[job_id] = job
            
            # Start processing
            job.start_processing()
            
            logger.info(f"Started file processing job {job_id} for file {original_filename}")
            
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
@require_auth(Permission.CREATE_JOB)
def process_url():
    """Handle URL processing."""
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
            job = ProcessingJob(job_id, 'url', url, options)
            
            with job_lock:
                processing_jobs[job_id] = job
            
            # Start processing
            job.start_processing()
            
            logger.info(f"Started URL processing job {job_id} for URL {url}")
            
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


@app.route('/job/<job_id>')
def job_status(job_id):
    """Get job status from memory or database."""
    # First check in-memory jobs
    with job_lock:
        if job_id in processing_jobs:
            job = processing_jobs[job_id]
            return jsonify({
                'job_id': job_id,
                'status': job.status,
                'progress': job.progress,
                'current_step': job.current_step,
                'error_message': job.error_message,
                'start_time': job.start_time,
                'end_time': job.end_time,
                'duration': job.end_time - job.start_time if job.end_time and job.start_time else None,
                'in_memory': True
            })
    
    # If not in memory, check database
    try:
        db = get_db_session()
        try:
            job_uuid = uuid.UUID(job_id)
            db_job = db.query(DBProcessingJob).filter(DBProcessingJob.id == job_uuid).first()
            
            if not db_job:
                return jsonify({'error': 'Job not found'}), 404
            
            duration = None
            if db_job.completed_at and db_job.started_at:
                duration = (db_job.completed_at - db_job.started_at).total_seconds()
            elif db_job.started_at:
                duration = (datetime.utcnow() - db_job.started_at).total_seconds()
            
            return jsonify({
                'job_id': job_id,
                'status': db_job.status,
                'progress': db_job.progress or 0.0,
                'current_step': db_job.current_step or '',
                'error_message': db_job.error_message or '',
                'start_time': db_job.started_at.timestamp() if db_job.started_at else None,
                'end_time': db_job.completed_at.timestamp() if db_job.completed_at else None,
                'duration': duration,
                'created_at': db_job.created_at.isoformat() if db_job.created_at else None,
                'in_memory': False
            })
        finally:
            db.close()
    except ValueError:
        return jsonify({'error': 'Invalid job ID format'}), 400
    except Exception as e:
        logger.error(f"Error getting job status from database: {e}")
        return jsonify({'error': 'Job not found'}), 404


@app.route('/download/<job_id>/<file_type>')
def download_file(job_id, file_type):
    """Download generated files."""
    with job_lock:
        if job_id not in processing_jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = processing_jobs[job_id]
        if job.status != 'completed':
            return jsonify({'error': 'Job not completed'}), 400
    
    output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
    
    if file_type == 'pdf':
        file_path = output_dir / 'report.pdf'
    elif file_type == 'transcript':
        file_path = output_dir / 'transcript' / 'transcript.txt'
    elif file_type == 'audio':
        file_path = output_dir / 'audio.wav'
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True)


@app.route('/job/<job_id>/transcript')
@require_auth(Permission.VIEW_JOB)
def view_transcript(job_id):
    """View transcript for a job."""
    try:
        # Check if job exists in memory or on disk
        job_exists = False
        job_completed = False
        
        with job_lock:
            if job_id in processing_jobs:
                job = processing_jobs[job_id]
                job_exists = True
                job_completed = (job.status == 'completed')
        
        # Also check if transcript file exists on disk (for jobs that completed before server restart)
        if not job_completed:
            output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
            segments_json = output_dir / 'transcript' / 'segments.json'
            if segments_json.exists():
                job_exists = True
                job_completed = True
        
        if not job_exists:
            return jsonify({'error': 'Job not found'}), 404
        
        if not job_completed:
            return jsonify({'error': 'Job not completed'}), 400
        
        # Get timestamp from query parameter
        timestamp = request.args.get('t', type=float)
        
        return render_template('transcript_viewer.html', job_id=job_id, timestamp=timestamp)
    except Exception as e:
        logger.error(f"Error viewing transcript for job {job_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to load transcript'}), 500


@app.route('/api/job/<job_id>/transcript/segments')
@require_auth(Permission.VIEW_JOB)
def get_transcript_segments(job_id):
    """Get transcript segments for a job."""
    try:
        # Load segments from JSON file (check disk directly, not just memory)
        output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
        segments_json = output_dir / 'transcript' / 'segments.json'
        
        if not segments_json.exists():
            # Check if job exists in memory to provide better error message
            with job_lock:
                if job_id in processing_jobs:
                    job = processing_jobs[job_id]
                    if job.status != 'completed':
                        return jsonify({'error': 'Job not completed'}), 400
            return jsonify({'error': 'Transcript not found'}), 404
        
        with open(segments_json, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        return jsonify({
            'job_id': job_id,
            'segments': segments,
            'count': len(segments)
        })
    except Exception as e:
        logger.error(f"Error getting transcript segments for job {job_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to load transcript segments'}), 500


@app.route('/jobs')
@require_auth(Permission.VIEW_JOB)
def list_jobs():
    """List all processing jobs with filtering and sorting, including database jobs."""
    try:
        # Get query parameters
        status_filter = request.args.get('status', '').lower()
        type_filter = request.args.get('type', '').lower()
        search_query = request.args.get('search', '').lower()
        sort_by = request.args.get('sort', 'created_desc')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        include_db = request.args.get('include_db', 'true').lower() == 'true'
        
        # Validate pagination
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 200:
            per_page = 50
        
        all_jobs = []
        in_memory_job_ids = set()
        
        # Get individual jobs from memory
        with job_lock:
            for job_id, job in processing_jobs.items():
                in_memory_job_ids.add(job_id)
                all_jobs.append({
                    'id': job_id,
                    'job_id': job_id,
                    'job_type': job.job_type,
                    'type': 'job',
                    'identifier': job.identifier,
                    'status': job.status,
                    'progress': job.progress,
                    'current_step': job.current_step,
                    'error_message': job.error_message,
                    'start_time': job.start_time,
                    'end_time': job.end_time,
                    'duration': job.end_time - job.start_time if job.end_time and job.start_time else None,
                    'created_at': job.created_at.isoformat(),
                    'last_activity': job.last_activity.isoformat(),
                    'in_memory': True
                })
        
        # Get jobs from database (excluding those already in memory)
        if include_db:
            try:
                db = get_db_session()
                try:
                    query = db.query(DBProcessingJob)
                    
                    # Apply filters
                    if status_filter:
                        query = query.filter(DBProcessingJob.status == status_filter)
                    if type_filter and type_filter != 'batch':
                        query = query.filter(DBProcessingJob.job_type == type_filter)
                    if search_query:
                        query = query.filter(
                            or_(
                                DBProcessingJob.identifier.ilike(f'%{search_query}%'),
                                DBProcessingJob.current_step.ilike(f'%{search_query}%')
                            )
                        )
                    
                    # Get total count
                    total_db_jobs = query.count()
                    
                    # Apply sorting
                    if sort_by == 'created_desc':
                        query = query.order_by(DBProcessingJob.created_at.desc())
                    elif sort_by == 'created_asc':
                        query = query.order_by(DBProcessingJob.created_at.asc())
                    elif sort_by == 'status':
                        query = query.order_by(DBProcessingJob.status, DBProcessingJob.created_at.desc())
                    
                    # Get jobs (limit to avoid too many results)
                    db_jobs = query.limit(1000).all()
                    
                    for db_job in db_jobs:
                        db_job_id = str(db_job.id)
                        # Skip if already in memory
                        if db_job_id in in_memory_job_ids:
                            continue
                        
                        # Calculate duration
                        duration = None
                        if db_job.completed_at and db_job.started_at:
                            duration = (db_job.completed_at - db_job.started_at).total_seconds()
                        elif db_job.started_at:
                            duration = (datetime.utcnow() - db_job.started_at).total_seconds()
                        
                        all_jobs.append({
                            'id': db_job_id,
                            'job_id': db_job_id,
                            'job_type': db_job.job_type,
                            'type': 'job',
                            'identifier': db_job.identifier,
                            'status': db_job.status,
                            'progress': db_job.progress or 0.0,
                            'current_step': db_job.current_step or '',
                            'error_message': db_job.error_message or '',
                            'start_time': db_job.started_at.timestamp() if db_job.started_at else None,
                            'end_time': db_job.completed_at.timestamp() if db_job.completed_at else None,
                            'duration': duration,
                            'created_at': db_job.created_at.isoformat() if db_job.created_at else None,
                            'last_activity': db_job.last_activity.isoformat() if db_job.last_activity else None,
                            'in_memory': False
                        })
                finally:
                    db.close()
            except Exception as e:
                logger.warning(f"Error loading jobs from database: {e}")
                # Continue with in-memory jobs only
        
        # Get batch jobs
        with batch_lock:
            for batch_id, batch_job in batch_jobs.items():
                batch_job.update_status()
                all_jobs.append({
                    'id': batch_id,
                    'job_id': batch_id,
                    'job_type': 'batch',
                    'type': 'batch',
                    'identifier': f"Batch ({batch_job.total_count} jobs)",
                    'status': batch_job.status,
                    'progress': (batch_job.completed_count + batch_job.failed_count) / batch_job.total_count * 100 if batch_job.total_count > 0 else 0,
                    'current_step': f"{batch_job.completed_count}/{batch_job.total_count} completed",
                    'error_message': batch_job.error_message,
                    'start_time': batch_job.start_time,
                    'end_time': batch_job.end_time,
                    'duration': batch_job.end_time - batch_job.start_time if batch_job.end_time and batch_job.start_time else None,
                    'created_at': batch_job.created_at.isoformat(),
                    'last_activity': batch_job.created_at.isoformat(),
                    'total_count': batch_job.total_count,
                    'completed_count': batch_job.completed_count,
                    'failed_count': batch_job.failed_count
                })
        
        # Apply filters
        filtered_jobs = all_jobs
        if status_filter:
            filtered_jobs = [j for j in filtered_jobs if j['status'] == status_filter]
        if type_filter:
            if type_filter == 'batch':
                filtered_jobs = [j for j in filtered_jobs if j['type'] == 'batch']
            else:
                filtered_jobs = [j for j in filtered_jobs if j['type'] == 'job' and j['job_type'] == type_filter]
        if search_query:
            filtered_jobs = [j for j in filtered_jobs if 
                           search_query in j['identifier'].lower() or 
                           search_query in j['job_id'].lower() or
                           (j.get('current_step', '') and search_query in j['current_step'].lower())]
        
        # Apply sorting (only if not already sorted by DB query)
        if not include_db or sort_by not in ['created_desc', 'created_asc', 'status']:
            if sort_by == 'created_desc':
                filtered_jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            elif sort_by == 'created_asc':
                filtered_jobs.sort(key=lambda x: x.get('created_at', ''))
            elif sort_by == 'status':
                filtered_jobs.sort(key=lambda x: (x.get('status', ''), x.get('created_at', '')), reverse=True)
            elif sort_by == 'type':
                filtered_jobs.sort(key=lambda x: (x.get('type', ''), x.get('job_type', '')))
        
        # Apply pagination
        total_jobs = len(filtered_jobs)
        total_pages = (total_jobs + per_page - 1) // per_page if total_jobs > 0 else 0
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_jobs = filtered_jobs[start_idx:end_idx]
        
        return jsonify({
            'jobs': paginated_jobs,
            'total': total_jobs,
            'filtered': len(filtered_jobs),
            'all_total': len(all_jobs),
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        })
    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        return jsonify({'error': 'Failed to list jobs'}), 500


@app.route('/job/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a processing job."""
    try:
        with job_lock:
            if job_id not in processing_jobs:
                return jsonify({'error': 'Job not found'}), 404
            
            job = processing_jobs[job_id]
            if job.cancel():
                logger.info(f"Cancelled job {job_id}")
                return jsonify({'message': 'Job cancelled successfully'})
            else:
                return jsonify({'error': 'Job cannot be cancelled'}), 400
                
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to cancel job'}), 500


@app.route('/job/<job_id>/retry', methods=['POST'])
@require_auth(Permission.CREATE_JOB)
def retry_job(job_id):
    """Retry a failed or cancelled job."""
    try:
        with job_lock:
            if job_id not in processing_jobs:
                return jsonify({'error': 'Job not found'}), 404
            
            original_job = processing_jobs[job_id]
            
            # Only allow retry for failed or cancelled jobs
            if original_job.status not in ['failed', 'cancelled']:
                return jsonify({'error': f'Job cannot be retried. Current status: {original_job.status}'}), 400
            
            # Create a new job with the same parameters
            new_job_id = str(uuid.uuid4())
            new_job = ProcessingJob(
                job_id=new_job_id,
                job_type=original_job.job_type,
                identifier=original_job.identifier,
                options=original_job.options
            )
            
            # Add to processing jobs
            processing_jobs[new_job_id] = new_job
            
            # Start processing
            try:
                new_job.start_processing()
                logger.info(f"Retrying job {job_id} as new job {new_job_id}")
                
                return jsonify({
                    'message': 'Job retry started successfully',
                    'original_job_id': job_id,
                    'new_job_id': new_job_id,
                    'status': 'started'
                })
            except ValueError as e:
                # Remove the job if starting failed
                del processing_jobs[new_job_id]
                return jsonify({'error': str(e)}), 400
                
    except Exception as e:
        logger.error(f"Error retrying job {job_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to retry job'}), 500


@app.route('/job/<job_id>/cleanup', methods=['POST'])
def cleanup_job(job_id):
    """Clean up job resources."""
    try:
        with job_lock:
            if job_id not in processing_jobs:
                return jsonify({'error': 'Job not found'}), 404
            
            job = processing_jobs[job_id]
            job.cleanup()
            
            # Remove job from memory if it's completed or failed
            if job.status in ['completed', 'failed', 'cancelled']:
                del processing_jobs[job_id]
                logger.info(f"Removed completed job {job_id} from memory")
            
            return jsonify({'message': 'Job cleaned up successfully'})
            
    except Exception as e:
        logger.error(f"Error cleaning up job {job_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to cleanup job'}), 500


@app.route('/job/<job_id>', methods=['DELETE'])
@require_auth(Permission.DELETE_JOB)
def delete_job(job_id):
    """Delete a job and its associated files."""
    try:
        deleted_from_memory = False
        deleted_from_db = False
        
        # Delete from memory if exists
        with job_lock:
            if job_id in processing_jobs:
                job = processing_jobs[job_id]
                # Cancel if still processing
                if job.status == 'processing':
                    job.cancel()
                # Cleanup resources
                job.cleanup()
                del processing_jobs[job_id]
                deleted_from_memory = True
                logger.info(f"Deleted job {job_id} from memory")
        
        # Delete from database
        try:
            db = get_db_session()
            try:
                db_job = db.query(DBProcessingJob).filter(DBProcessingJob.id == job_id).first()
                if db_job:
                    db.delete(db_job)
                    db.commit()
                    deleted_from_db = True
                    logger.info(f"Deleted job {job_id} from database")
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Error deleting job {job_id} from database: {e}")
        
        # Delete output directory
        output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
        if output_dir.exists():
            try:
                shutil.rmtree(output_dir, ignore_errors=True)
                logger.info(f"Deleted output directory for job {job_id}")
            except Exception as e:
                logger.warning(f"Error deleting output directory for job {job_id}: {e}")
        
        if deleted_from_memory or deleted_from_db:
            return jsonify({'message': 'Job deleted successfully'})
        else:
            return jsonify({'error': 'Job not found'}), 404
            
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to delete job'}), 500


@app.route('/api/jobs/bulk-delete', methods=['POST'])
@require_auth(Permission.DELETE_JOB)
def bulk_delete_jobs():
    """Delete multiple jobs at once."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        job_ids = data.get('job_ids', [])
        if not job_ids or not isinstance(job_ids, list):
            return jsonify({'error': 'job_ids list is required'}), 400
        
        if len(job_ids) > 100:
            return jsonify({'error': 'Maximum 100 jobs can be deleted at once'}), 400
        
        deleted_count = 0
        failed_count = 0
        errors = []
        
        for job_id in job_ids:
            try:
                # Delete from memory if exists
                with job_lock:
                    if job_id in processing_jobs:
                        job = processing_jobs[job_id]
                        if job.status == 'processing':
                            job.cancel()
                        job.cleanup()
                        del processing_jobs[job_id]
                
                # Delete from database
                try:
                    db = get_db_session()
                    try:
                        db_job = db.query(DBProcessingJob).filter(DBProcessingJob.id == job_id).first()
                        if db_job:
                            db.delete(db_job)
                            db.commit()
                    finally:
                        db.close()
                except Exception as e:
                    logger.warning(f"Error deleting job {job_id} from database: {e}")
                
                # Delete output directory
                output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
                if output_dir.exists():
                    try:
                        shutil.rmtree(output_dir, ignore_errors=True)
                    except Exception as e:
                        logger.warning(f"Error deleting output directory for job {job_id}: {e}")
                
                deleted_count += 1
                
            except Exception as e:
                failed_count += 1
                errors.append({'job_id': job_id, 'error': str(e)})
                logger.error(f"Error deleting job {job_id}: {e}")
        
        return jsonify({
            'message': f'Deleted {deleted_count} job(s)',
            'deleted_count': deleted_count,
            'failed_count': failed_count,
            'errors': errors
        })
        
    except Exception as e:
        logger.error(f"Error in bulk delete: {e}", exc_info=True)
        return jsonify({'error': 'Failed to delete jobs'}), 500


# Batch Processing Endpoints
@app.route('/batch/create', methods=['POST'])
@require_auth(Permission.CREATE_JOB)
def create_batch():
    """Create a new batch processing job."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Get items (files or URLs)
        items = data.get('items', [])
        if not items or not isinstance(items, list):
            return jsonify({'error': 'Items list is required'}), 400
        
        if len(items) == 0:
            return jsonify({'error': 'At least one item is required'}), 400
        
        if len(items) > 50:  # Limit batch size
            return jsonify({'error': 'Maximum 50 items per batch'}), 400
        
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
        
        # Create batch job
        batch_id = str(uuid.uuid4())
        batch_job = BatchJob(batch_id, options)
        
        # Process each item
        created_jobs = []
        for item in items:
            item_type = item.get('type')  # 'url' or 'file'
            identifier = item.get('identifier')  # URL or file path
            
            if not item_type or not identifier:
                continue
            
            if item_type == 'url':
                validate_url(identifier)
            elif item_type == 'file':
                # File should already be uploaded
                file_path = Path(app.config['UPLOAD_FOLDER']) / identifier
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                identifier = str(file_path)
            else:
                continue
            
            # Create individual job
            job_id = str(uuid.uuid4())
            try:
                job = ProcessingJob(job_id, item_type, identifier, options)
                with job_lock:
                    processing_jobs[job_id] = job
                batch_job.add_job(job_id)
                created_jobs.append(job_id)
            except Exception as e:
                logger.error(f"Error creating job for item {identifier}: {e}")
                continue
        
        if len(created_jobs) == 0:
            return jsonify({'error': 'No valid items to process'}), 400
        
        # Store batch job
        with batch_lock:
            batch_jobs[batch_id] = batch_job
        
        # Start processing all jobs
        for job_id in created_jobs:
            try:
                with job_lock:
                    if job_id in processing_jobs:
                        processing_jobs[job_id].start_processing()
            except Exception as e:
                logger.error(f"Error starting job {job_id}: {e}")
        
        logger.info(f"Created batch {batch_id} with {len(created_jobs)} jobs")
        
        return jsonify({
            'batch_id': batch_id,
            'status': 'started',
            'total_jobs': len(created_jobs),
            'job_ids': created_jobs,
            'message': f'Batch processing started with {len(created_jobs)} jobs'
        })
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in create_batch: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/batch/upload', methods=['POST'])
@require_auth(Permission.UPLOAD_FILES)
def batch_upload_files():
    """Handle batch file upload."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files provided'}), 400
        
        if len(files) > 50:
            return jsonify({'error': 'Maximum 50 files per batch'}), 400
        
        # Validate and save all files
        uploaded_files = []
        for file in files:
            try:
                validate_file_upload(file)
                job_id = str(uuid.uuid4())
                original_filename = sanitize_filename(file.filename)
                filename = f"{job_id}_{original_filename}"
                file_path = Path(app.config['UPLOAD_FOLDER']) / filename
                file.save(file_path)
                uploaded_files.append({
                    'job_id': job_id,
                    'filename': filename,
                    'original_filename': original_filename,
                    'path': str(file_path)
                })
                logger.info(f"Saved uploaded file: {file_path}")
            except ValidationError as e:
                logger.warning(f"File validation failed: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to save file: {e}")
                continue
        
        if len(uploaded_files) == 0:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        return jsonify({
            'files': uploaded_files,
            'count': len(uploaded_files),
            'message': f'Successfully uploaded {len(uploaded_files)} files'
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in batch_upload_files: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/batch/<batch_id>')
@require_auth(Permission.VIEW_JOB)
def get_batch_status(batch_id):
    """Get batch job status."""
    try:
        with batch_lock:
            if batch_id not in batch_jobs:
                return jsonify({'error': 'Batch not found'}), 404
            
            batch_job = batch_jobs[batch_id]
            return jsonify(batch_job.to_dict())
            
    except Exception as e:
        logger.error(f"Error getting batch status {batch_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get batch status'}), 500


@app.route('/batch/<batch_id>/jobs')
@require_auth(Permission.VIEW_JOB)
def get_batch_jobs(batch_id):
    """Get all jobs in a batch."""
    try:
        with batch_lock:
            if batch_id not in batch_jobs:
                return jsonify({'error': 'Batch not found'}), 404
            
            batch_job = batch_jobs[batch_id]
            jobs = []
            
            with job_lock:
                for job_id in batch_job.job_ids:
                    if job_id in processing_jobs:
                        job = processing_jobs[job_id]
                        jobs.append({
                            'job_id': job_id,
                            'job_type': job.job_type,
                            'identifier': job.identifier,
                            'status': job.status,
                            'progress': job.progress,
                            'current_step': job.current_step,
                            'error_message': job.error_message,
                            'start_time': job.start_time,
                            'end_time': job.end_time
                        })
            
            return jsonify({
                'batch_id': batch_id,
                'jobs': jobs,
                'count': len(jobs)
            })
            
    except Exception as e:
        logger.error(f"Error getting batch jobs {batch_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get batch jobs'}), 500


@app.route('/batch/<batch_id>/cancel', methods=['POST'])
@require_auth(Permission.VIEW_JOB)
def cancel_batch(batch_id):
    """Cancel a batch processing job."""
    try:
        with batch_lock:
            if batch_id not in batch_jobs:
                return jsonify({'error': 'Batch not found'}), 404
            
            batch_job = batch_jobs[batch_id]
            if batch_job.status in ['completed', 'failed', 'cancelled']:
                return jsonify({'error': 'Batch cannot be cancelled'}), 400
            
            batch_job.status = 'cancelled'
            batch_job.end_time = time.time()
            
            # Cancel all individual jobs
            with job_lock:
                for job_id in batch_job.job_ids:
                    if job_id in processing_jobs:
                        processing_jobs[job_id].cancel()
            
            logger.info(f"Cancelled batch {batch_id}")
            return jsonify({'message': 'Batch cancelled successfully'})
            
    except Exception as e:
        logger.error(f"Error cancelling batch {batch_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to cancel batch'}), 500


@app.route('/batches')
@require_auth(Permission.VIEW_JOB)
def list_batches():
    """List all batch jobs."""
    try:
        with batch_lock:
            batches = []
            for batch_id, batch_job in batch_jobs.items():
                batches.append(batch_job.to_dict())
        
        return jsonify({'batches': batches, 'count': len(batches)})
        
    except Exception as e:
        logger.error(f"Error listing batches: {e}", exc_info=True)
        return jsonify({'error': 'Failed to list batches'}), 500


# Job History Endpoints
@app.route('/jobs/history')
@require_auth(Permission.VIEW_JOB)
def job_history_page():
    """Job history dashboard page."""
    user_session = get_current_user_session()
    return render_template('job_history.html', user=user_session)


@app.route('/api/jobs/history')
@require_auth(Permission.VIEW_JOB)
def get_job_history():
    """Get job history from database with pagination and filtering."""
    try:
        db = get_db_session()
        try:
            # Get query parameters
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 20))
            status_filter = request.args.get('status')
            job_type_filter = request.args.get('job_type')
            search_query = request.args.get('search')
            
            # Validate pagination
            if page < 1:
                page = 1
            if per_page < 1 or per_page > 100:
                per_page = 20
            
            # Build query
            query = db.query(DBProcessingJob)
            
            # Apply filters
            if status_filter:
                query = query.filter(DBProcessingJob.status == status_filter)
            if job_type_filter:
                query = query.filter(DBProcessingJob.job_type == job_type_filter)
            if search_query:
                query = query.filter(
                    or_(
                        DBProcessingJob.identifier.ilike(f'%{search_query}%'),
                        DBProcessingJob.current_step.ilike(f'%{search_query}%')
                    )
                )
            
            # Get total count
            total = query.count()
            
            # Apply pagination and ordering
            jobs = query.order_by(DBProcessingJob.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()
            
            # Format results
            jobs_data = []
            for job in jobs:
                jobs_data.append({
                    'id': str(job.id),
                    'job_type': job.job_type,
                    'identifier': job.identifier,
                    'status': job.status,
                    'progress': job.progress,
                    'current_step': job.current_step,
                    'error_message': job.error_message,
                    'created_at': job.created_at.isoformat() if job.created_at else None,
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'duration': (job.completed_at - job.started_at).total_seconds() if job.completed_at and job.started_at else None
                })
            
            total_pages = (total + per_page - 1) // per_page if total > 0 else 0
            
            return jsonify({
                'jobs': jobs_data,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'total_pages': total_pages
                }
            })
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting job history: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get job history'}), 500


@app.route('/api/jobs/stats')
@require_auth(Permission.VIEW_JOB)
def get_job_stats():
    """Get job statistics."""
    try:
        db = get_db_session()
        try:
            job_manager = JobManager(db)
            stats = job_manager.get_job_stats()
            return jsonify(stats)
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error getting job stats: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get job stats'}), 500


@app.route('/api/jobs/export')
@require_auth(Permission.VIEW_JOB)
def export_jobs():
    """Export jobs data as CSV or JSON."""
    try:
        export_format = request.args.get('format', 'json').lower()
        if export_format not in ['json', 'csv']:
            return jsonify({'error': 'Invalid format. Use "json" or "csv"'}), 400
        
        # Get filter parameters (same as list_jobs)
        status_filter = request.args.get('status', '').lower()
        type_filter = request.args.get('type', '').lower()
        search_query = request.args.get('search', '').lower()
        include_db = request.args.get('include_db', 'true').lower() == 'true'
        
        # Get all jobs (no pagination for export)
        all_jobs = []
        in_memory_job_ids = set()
        
        # Get individual jobs from memory
        with job_lock:
            for job_id, job in processing_jobs.items():
                in_memory_job_ids.add(job_id)
                all_jobs.append({
                    'job_id': job_id,
                    'job_type': job.job_type,
                    'type': 'job',
                    'identifier': job.identifier,
                    'status': job.status,
                    'progress': job.progress,
                    'current_step': job.current_step,
                    'error_message': job.error_message,
                    'start_time': job.start_time,
                    'end_time': job.end_time,
                    'duration': job.end_time - job.start_time if job.end_time and job.start_time else None,
                    'created_at': job.created_at.isoformat(),
                    'last_activity': job.last_activity.isoformat()
                })
        
        # Get jobs from database
        if include_db:
            try:
                db = get_db_session()
                try:
                    from sqlalchemy import or_
                    query = db.query(DBProcessingJob)
                    
                    if status_filter:
                        query = query.filter(DBProcessingJob.status == status_filter)
                    if type_filter and type_filter != 'batch':
                        query = query.filter(DBProcessingJob.job_type == type_filter)
                    if search_query:
                        query = query.filter(
                            or_(
                                DBProcessingJob.identifier.ilike(f'%{search_query}%'),
                                DBProcessingJob.current_step.ilike(f'%{search_query}%')
                            )
                        )
                    
                    db_jobs = query.order_by(DBProcessingJob.created_at.desc()).limit(5000).all()
                    
                    for db_job in db_jobs:
                        db_job_id = str(db_job.id)
                        if db_job_id in in_memory_job_ids:
                            continue
                        
                        duration = None
                        if db_job.completed_at and db_job.started_at:
                            duration = (db_job.completed_at - db_job.started_at).total_seconds()
                        elif db_job.started_at:
                            duration = (datetime.utcnow() - db_job.started_at).total_seconds()
                        
                        all_jobs.append({
                            'job_id': db_job_id,
                            'job_type': db_job.job_type,
                            'type': 'job',
                            'identifier': db_job.identifier,
                            'status': db_job.status,
                            'progress': db_job.progress or 0.0,
                            'current_step': db_job.current_step or '',
                            'error_message': db_job.error_message or '',
                            'start_time': db_job.started_at.timestamp() if db_job.started_at else None,
                            'end_time': db_job.completed_at.timestamp() if db_job.completed_at else None,
                            'duration': duration,
                            'created_at': db_job.created_at.isoformat() if db_job.created_at else None,
                            'last_activity': db_job.last_activity.isoformat() if db_job.last_activity else None
                        })
                finally:
                    db.close()
            except Exception as e:
                logger.warning(f"Error loading jobs from database for export: {e}")
        
        # Get batch jobs
        with batch_lock:
            for batch_id, batch_job in batch_jobs.items():
                batch_job.update_status()
                all_jobs.append({
                    'job_id': batch_id,
                    'job_type': 'batch',
                    'type': 'batch',
                    'identifier': f"Batch ({batch_job.total_count} jobs)",
                    'status': batch_job.status,
                    'progress': (batch_job.completed_count + batch_job.failed_count) / batch_job.total_count * 100 if batch_job.total_count > 0 else 0,
                    'current_step': f"{batch_job.completed_count}/{batch_job.total_count} completed",
                    'error_message': batch_job.error_message,
                    'start_time': batch_job.start_time,
                    'end_time': batch_job.end_time,
                    'duration': batch_job.end_time - batch_job.start_time if batch_job.end_time and batch_job.start_time else None,
                    'created_at': batch_job.created_at.isoformat(),
                    'last_activity': batch_job.created_at.isoformat(),
                    'total_count': batch_job.total_count,
                    'completed_count': batch_job.completed_count,
                    'failed_count': batch_job.failed_count
                })
        
        # Apply filters
        filtered_jobs = all_jobs
        if status_filter:
            filtered_jobs = [j for j in filtered_jobs if j['status'] == status_filter]
        if type_filter:
            if type_filter == 'batch':
                filtered_jobs = [j for j in filtered_jobs if j['type'] == 'batch']
            else:
                filtered_jobs = [j for j in filtered_jobs if j['type'] == 'job' and j['job_type'] == type_filter]
        if search_query and not include_db:
            filtered_jobs = [j for j in filtered_jobs if search_query in j['identifier'].lower() or (j.get('current_step', '') and search_query in j['current_step'].lower())]
        
        # Sort by created_at descending
        filtered_jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Generate export
        if export_format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            if filtered_jobs:
                fieldnames = ['job_id', 'job_type', 'type', 'identifier', 'status', 'progress', 
                             'current_step', 'error_message', 'duration', 'created_at', 'last_activity']
                writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for job in filtered_jobs:
                    # Convert timestamps to ISO strings for CSV
                    row = job.copy()
                    if row.get('start_time') and isinstance(row['start_time'], (int, float)):
                        row['start_time'] = datetime.fromtimestamp(row['start_time']).isoformat()
                    if row.get('end_time') and isinstance(row['end_time'], (int, float)):
                        row['end_time'] = datetime.fromtimestamp(row['end_time']).isoformat()
                    writer.writerow(row)
            
            csv_data = output.getvalue()
            output.close()
            
            response = app.response_class(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=jobs_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
            )
            return response
        else:  # JSON
            export_data = {
                'export_info': {
                    'exported_at': datetime.utcnow().isoformat(),
                    'total_jobs': len(filtered_jobs),
                    'filters': {
                        'status': status_filter or None,
                        'type': type_filter or None,
                        'search': search_query or None
                    }
                },
                'jobs': filtered_jobs
            }
            
            response = app.response_class(
                json.dumps(export_data, indent=2, default=str),
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename=jobs_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'}
            )
            return response
            
    except Exception as e:
        logger.error(f"Error exporting jobs: {e}", exc_info=True)
        return jsonify({'error': 'Failed to export jobs'}), 500


@app.route('/api/jobs/analytics')
@require_auth(Permission.VIEW_JOB)
def get_job_analytics():
    """Get comprehensive job analytics and insights."""
    try:
        db = get_db_session()
        try:
            from sqlalchemy import func, extract
            from datetime import datetime, timedelta
            
            # Basic statistics
            total_jobs = db.query(DBProcessingJob).count()
            completed_jobs = db.query(DBProcessingJob).filter(
                DBProcessingJob.status == ProcessingStatus.COMPLETED
            ).count()
            failed_jobs = db.query(DBProcessingJob).filter(
                DBProcessingJob.status == ProcessingStatus.FAILED
            ).count()
            processing_jobs = db.query(DBProcessingJob).filter(
                DBProcessingJob.status == ProcessingStatus.PROCESSING
            ).count()
            
            # Success rate
            success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
            
            # Average processing time for completed jobs
            avg_duration = db.query(
                func.avg(
                    func.extract('epoch', DBProcessingJob.completed_at - DBProcessingJob.started_at)
                )
            ).filter(
                DBProcessingJob.status == ProcessingStatus.COMPLETED,
                DBProcessingJob.completed_at.isnot(None),
                DBProcessingJob.started_at.isnot(None)
            ).scalar() or 0
            
            # Jobs by type
            jobs_by_type = db.query(
                DBProcessingJob.job_type,
                func.count(DBProcessingJob.id).label('count')
            ).group_by(DBProcessingJob.job_type).all()
            
            # Jobs by status
            jobs_by_status = db.query(
                DBProcessingJob.status,
                func.count(DBProcessingJob.id).label('count')
            ).group_by(DBProcessingJob.status).all()
            
            # Jobs over time (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            # Use cast to date for cross-database compatibility
            try:
                from sqlalchemy import cast, Date
                jobs_over_time = db.query(
                    cast(DBProcessingJob.created_at, Date).label('date'),
                    func.count(DBProcessingJob.id).label('count')
                ).filter(
                    DBProcessingJob.created_at >= thirty_days_ago
                ).group_by(
                    cast(DBProcessingJob.created_at, Date)
                ).order_by(
                    cast(DBProcessingJob.created_at, Date)
                ).all()
            except:
                # Fallback: group by date string
                jobs_over_time = []
                jobs = db.query(DBProcessingJob).filter(
                    DBProcessingJob.created_at >= thirty_days_ago
                ).all()
                from collections import defaultdict
                daily_counts = defaultdict(int)
                for job in jobs:
                    date_str = job.created_at.date().isoformat()
                    daily_counts[date_str] += 1
                jobs_over_time = [(date, count) for date, count in sorted(daily_counts.items())]
            
            # Jobs by day of week (0=Monday, 6=Sunday)
            try:
                jobs_by_day = db.query(
                    extract('dow', DBProcessingJob.created_at).label('day_of_week'),
                    func.count(DBProcessingJob.id).label('count')
                ).filter(
                    DBProcessingJob.created_at >= thirty_days_ago
                ).group_by(
                    extract('dow', DBProcessingJob.created_at)
                ).all()
            except Exception as e:
                # Fallback: calculate in Python
                logger.debug(f"Using fallback for jobs_by_day: {e}")
                jobs_by_day = []
                jobs = db.query(DBProcessingJob).filter(
                    DBProcessingJob.created_at >= thirty_days_ago
                ).all()
                from collections import defaultdict
                day_counts = defaultdict(int)
                for job in jobs:
                    day_of_week = job.created_at.weekday()  # 0=Monday, 6=Sunday
                    day_counts[day_of_week] += 1
                jobs_by_day = [(day, count) for day, count in sorted(day_counts.items())]
            
            # Most common errors
            common_errors = db.query(
                DBProcessingJob.error_message,
                func.count(DBProcessingJob.id).label('count')
            ).filter(
                DBProcessingJob.status == ProcessingStatus.FAILED,
                DBProcessingJob.error_message.isnot(None),
                DBProcessingJob.error_message != ''
            ).group_by(
                DBProcessingJob.error_message
            ).order_by(
                func.count(DBProcessingJob.id).desc()
            ).limit(10).all()
            
            # Average processing time by job type
            avg_duration_by_type = db.query(
                DBProcessingJob.job_type,
                func.avg(
                    func.extract('epoch', DBProcessingJob.completed_at - DBProcessingJob.started_at)
                ).label('avg_duration')
            ).filter(
                DBProcessingJob.status == ProcessingStatus.COMPLETED,
                DBProcessingJob.completed_at.isnot(None),
                DBProcessingJob.started_at.isnot(None)
            ).group_by(DBProcessingJob.job_type).all()
            
            # Recent activity (last 24 hours)
            twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
            try:
                # Try PostgreSQL date_trunc function
                from sqlalchemy import text
                recent_activity = db.query(
                    text("date_trunc('hour', processing_jobs.created_at) as hour"),
                    func.count(DBProcessingJob.id).label('count')
                ).filter(
                    DBProcessingJob.created_at >= twenty_four_hours_ago
                ).group_by(
                    text("date_trunc('hour', processing_jobs.created_at)")
                ).order_by(
                    text("date_trunc('hour', processing_jobs.created_at)")
                ).all()
            except Exception as e:
                # Fallback: calculate in Python
                logger.debug(f"Using fallback for recent_activity: {e}")
                recent_activity = []
                jobs = db.query(DBProcessingJob).filter(
                    DBProcessingJob.created_at >= twenty_four_hours_ago
                ).all()
                from collections import defaultdict
                hourly_counts = defaultdict(int)
                for job in jobs:
                    # Round to nearest hour
                    hour = job.created_at.replace(minute=0, second=0, microsecond=0)
                    hourly_counts[hour] += 1
                recent_activity = [(hour, count) for hour, count in sorted(hourly_counts.items())]
            
            return jsonify({
                'summary': {
                    'total_jobs': total_jobs,
                    'completed_jobs': completed_jobs,
                    'failed_jobs': failed_jobs,
                    'processing_jobs': processing_jobs,
                    'success_rate': round(success_rate, 2),
                    'avg_duration_seconds': round(avg_duration, 2) if avg_duration else 0
                },
                'by_type': {job_type: count for job_type, count in jobs_by_type},
                'by_status': {status: count for status, count in jobs_by_status},
                'over_time': [
                    {'date': date.isoformat() if hasattr(date, 'isoformat') else str(date), 'count': count}
                    for date, count in jobs_over_time
                ],
                'by_day_of_week': {int(day): count for day, count in jobs_by_day},
                'common_errors': [
                    {'error': error[:200] if error else 'Unknown error', 'count': count}
                    for error, count in common_errors
                ],
                'avg_duration_by_type': {
                    job_type: round(avg_dur, 2) if avg_dur else 0
                    for job_type, avg_dur in avg_duration_by_type
                },
                'recent_activity': [
                    {'hour': hour.isoformat() if hasattr(hour, 'isoformat') else (str(hour) if isinstance(hour, datetime) else hour), 'count': count}
                    for hour, count in recent_activity
                ]
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error getting job analytics: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get job analytics'}), 500


@app.route('/analytics')
@require_auth(Permission.VIEW_JOB)
def analytics_page():
    """Job analytics dashboard page."""
    user_session = get_current_user_session()
    return render_template('analytics.html', user=user_session)


# Processing Preset Endpoints
@app.route('/api/presets', methods=['GET'])
@require_auth(Permission.VIEW_JOB)
def list_presets():
    """List all available presets."""
    try:
        user_session = get_current_user_session()
        user_id = user_session.user_id if user_session else None
        
        db = get_db_session()
        try:
            # Get user's presets and public presets
            query = db.query(ProcessingPreset).filter(
                (ProcessingPreset.user_id == user_id) | (ProcessingPreset.is_public == True)
            )
            
            presets = query.order_by(ProcessingPreset.is_default.desc(), ProcessingPreset.usage_count.desc()).all()
            
            preset_list = []
            for preset in presets:
                preset_list.append({
                    'id': str(preset.id),
                    'name': preset.name,
                    'description': preset.description,
                    'config': preset.config,
                    'is_default': preset.is_default,
                    'is_public': preset.is_public,
                    'usage_count': preset.usage_count,
                    'created_at': preset.created_at.isoformat() if preset.created_at else None,
                    'last_used_at': preset.last_used_at.isoformat() if preset.last_used_at else None,
                    'is_owner': preset.user_id == user_id
                })
            
            return jsonify({'presets': preset_list})
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error listing presets: {e}", exc_info=True)
        return jsonify({'error': 'Failed to list presets'}), 500


@app.route('/api/presets', methods=['POST'])
@require_auth(Permission.CREATE_JOB)
def create_preset():
    """Create a new processing preset."""
    try:
        user_session = get_current_user_session()
        user_id = user_session.user_id if user_session else None
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        name = data.get('name', '').strip()
        if not name:
            return jsonify({'error': 'Preset name is required'}), 400
        
        config = data.get('config', {})
        if not config:
            return jsonify({'error': 'Preset configuration is required'}), 400
        
        description = data.get('description', '').strip()
        is_default = data.get('is_default', False)
        is_public = data.get('is_public', False)
        
        db = get_db_session()
        try:
            # Check if name already exists for this user
            existing = db.query(ProcessingPreset).filter(
                ProcessingPreset.name == name,
                ProcessingPreset.user_id == user_id
            ).first()
            
            if existing:
                return jsonify({'error': 'A preset with this name already exists'}), 400
            
            # If setting as default, unset other defaults for this user
            if is_default:
                db.query(ProcessingPreset).filter(
                    ProcessingPreset.user_id == user_id,
                    ProcessingPreset.is_default == True
                ).update({'is_default': False})
            
            # Create new preset
            preset = ProcessingPreset(
                user_id=user_id,
                name=name,
                description=description,
                config=config,
                is_default=is_default,
                is_public=is_public
            )
            
            db.add(preset)
            db.commit()
            
            logger.info(f"Created preset '{name}' for user {user_id}")
            
            return jsonify({
                'message': 'Preset created successfully',
                'preset': {
                    'id': str(preset.id),
                    'name': preset.name,
                    'description': preset.description,
                    'config': preset.config,
                    'is_default': preset.is_default,
                    'is_public': preset.is_public
                }
            }), 201
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error creating preset: {e}", exc_info=True)
        return jsonify({'error': 'Failed to create preset'}), 500


@app.route('/api/presets/<preset_id>', methods=['GET'])
@require_auth(Permission.VIEW_JOB)
def get_preset(preset_id):
    """Get a specific preset."""
    try:
        user_session = get_current_user_session()
        user_id = user_session.user_id if user_session else None
        
        db = get_db_session()
        try:
            # Convert string to UUID if needed
            try:
                preset_uuid = uuid.UUID(preset_id) if isinstance(preset_id, str) else preset_id
            except ValueError:
                return jsonify({'error': 'Invalid preset ID format'}), 400
            
            preset = db.query(ProcessingPreset).filter(
                ProcessingPreset.id == preset_uuid,
                ((ProcessingPreset.user_id == user_id) | (ProcessingPreset.is_public == True))
            ).first()
            
            if not preset:
                return jsonify({'error': 'Preset not found'}), 404
            
            return jsonify({
                'id': str(preset.id),
                'name': preset.name,
                'description': preset.description,
                'config': preset.config,
                'is_default': preset.is_default,
                'is_public': preset.is_public,
                'usage_count': preset.usage_count,
                'created_at': preset.created_at.isoformat() if preset.created_at else None,
                'last_used_at': preset.last_used_at.isoformat() if preset.last_used_at else None
            })
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting preset: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get preset'}), 500


@app.route('/api/presets/<preset_id>', methods=['PUT'])
@require_auth(Permission.CREATE_JOB)
def update_preset(preset_id):
    """Update an existing preset."""
    try:
        user_session = get_current_user_session()
        user_id = user_session.user_id if user_session else None
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        db = get_db_session()
        try:
            # Convert string to UUID if needed
            try:
                preset_uuid = uuid.UUID(preset_id) if isinstance(preset_id, str) else preset_id
            except ValueError:
                return jsonify({'error': 'Invalid preset ID format'}), 400
            
            preset = db.query(ProcessingPreset).filter(
                ProcessingPreset.id == preset_uuid,
                ProcessingPreset.user_id == user_id
            ).first()
            
            if not preset:
                return jsonify({'error': 'Preset not found or access denied'}), 404
            
            # Update fields
            if 'name' in data:
                new_name = data['name'].strip()
                if new_name and new_name != preset.name:
                    # Check if name already exists
                    existing = db.query(ProcessingPreset).filter(
                        ProcessingPreset.name == new_name,
                        ProcessingPreset.user_id == user_id,
                        ProcessingPreset.id != preset_id
                    ).first()
                    if existing:
                        return jsonify({'error': 'A preset with this name already exists'}), 400
                    preset.name = new_name
            
            if 'description' in data:
                preset.description = data['description'].strip()
            
            if 'config' in data:
                preset.config = data['config']
            
            if 'is_default' in data:
                is_default = data['is_default']
                if is_default and not preset.is_default:
                    # Unset other defaults
                    db.query(ProcessingPreset).filter(
                        ProcessingPreset.user_id == user_id,
                        ProcessingPreset.is_default == True,
                        ProcessingPreset.id != preset_id
                    ).update({'is_default': False})
                preset.is_default = is_default
            
            if 'is_public' in data:
                preset.is_public = data['is_public']
            
            preset.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Updated preset {preset_id}")
            
            return jsonify({'message': 'Preset updated successfully'})
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error updating preset: {e}", exc_info=True)
        return jsonify({'error': 'Failed to update preset'}), 500


@app.route('/api/presets/<preset_id>', methods=['DELETE'])
@require_auth(Permission.CREATE_JOB)
def delete_preset(preset_id):
    """Delete a preset."""
    try:
        user_session = get_current_user_session()
        user_id = user_session.user_id if user_session else None
        
        db = get_db_session()
        try:
            # Convert string to UUID if needed
            try:
                preset_uuid = uuid.UUID(preset_id) if isinstance(preset_id, str) else preset_id
            except ValueError:
                return jsonify({'error': 'Invalid preset ID format'}), 400
            
            preset = db.query(ProcessingPreset).filter(
                ProcessingPreset.id == preset_uuid,
                ProcessingPreset.user_id == user_id
            ).first()
            
            if not preset:
                return jsonify({'error': 'Preset not found or access denied'}), 404
            
            db.delete(preset)
            db.commit()
            
            logger.info(f"Deleted preset {preset_id}")
            
            return jsonify({'message': 'Preset deleted successfully'})
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error deleting preset: {e}", exc_info=True)
        return jsonify({'error': 'Failed to delete preset'}), 500


@app.route('/api/presets/<preset_id>/use', methods=['POST'])
@require_auth(Permission.CREATE_JOB)
def use_preset(preset_id):
    """Mark a preset as used and increment usage count."""
    try:
        user_session = get_current_user_session()
        user_id = user_session.user_id if user_session else None
        
        db = get_db_session()
        try:
            # Convert string to UUID if needed
            try:
                preset_uuid = uuid.UUID(preset_id) if isinstance(preset_id, str) else preset_id
            except ValueError:
                return jsonify({'error': 'Invalid preset ID format'}), 400
            
            preset = db.query(ProcessingPreset).filter(
                ProcessingPreset.id == preset_uuid,
                ((ProcessingPreset.user_id == user_id) | (ProcessingPreset.is_public == True))
            ).first()
            
            if not preset:
                return jsonify({'error': 'Preset not found'}), 404
            
            preset.usage_count += 1
            preset.last_used_at = datetime.utcnow()
            db.commit()
            
            return jsonify({
                'message': 'Preset usage recorded',
                'config': preset.config
            })
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error recording preset usage: {e}", exc_info=True)
        return jsonify({'error': 'Failed to record preset usage'}), 500


@app.route('/cleanup-stale', methods=['POST'])
def cleanup_stale_jobs():
    """Clean up stale jobs."""
    try:
        cleaned_count = 0
        with job_lock:
            stale_jobs = [job_id for job_id, job in processing_jobs.items() if job.is_stale()]
            
            for job_id in stale_jobs:
                job = processing_jobs[job_id]
                job.status = 'failed'
                job.error_message = 'Job timed out'
                job.cleanup()
                del processing_jobs[job_id]
                cleaned_count += 1
                logger.info(f"Cleaned up stale job {job_id}")
        
        return jsonify({'message': f'Cleaned up {cleaned_count} stale jobs'})
        
    except Exception as e:
        logger.error(f"Error cleaning up stale jobs: {e}", exc_info=True)
        return jsonify({'error': 'Failed to cleanup stale jobs'}), 500


def cleanup_old_jobs_and_files():
    """Automatically clean up old jobs and files based on retention policy."""
    global cleanup_running
    
    if cleanup_running:
        return
    
    cleanup_running = True
    try:
        logger.info("Starting automatic cleanup of old jobs and files...")
        
        now = datetime.now()
        job_retention = timedelta(days=Config.JOB_RETENTION_DAYS)
        failed_job_retention = timedelta(days=Config.FAILED_JOB_RETENTION_DAYS)
        upload_retention = timedelta(days=Config.UPLOAD_RETENTION_DAYS)
        
        cleaned_jobs = 0
        cleaned_files = 0
        freed_space = 0
        
        # Clean up old jobs from memory
        with job_lock:
            jobs_to_remove = []
            for job_id, job in list(processing_jobs.items()):
                if job.status in ['completed', 'failed', 'cancelled']:
                    age = now - job.created_at
                    retention = failed_job_retention if job.status == 'failed' else job_retention
                    
                    if age > retention:
                        jobs_to_remove.append(job_id)
                        job.cleanup()
                        cleaned_jobs += 1
                        logger.info(f"Cleaned up old {job.status} job {job_id} (age: {age.days} days)")
            
            for job_id in jobs_to_remove:
                if job_id in processing_jobs:
                    del processing_jobs[job_id]
        
        # Clean up old output directories
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        if output_dir.exists():
            for job_dir in output_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                
                try:
                    # Check if directory is old enough to delete
                    mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
                    age = now - mtime
                    
                    # Determine retention based on job status
                    job_id = job_dir.name
                    retention = failed_job_retention
                    
                    # Check if job exists in memory to determine status
                    with job_lock:
                        if job_id in processing_jobs:
                            job = processing_jobs[job_id]
                            if job.status == 'completed':
                                retention = job_retention
                            elif job.status == 'failed':
                                retention = failed_job_retention
                        else:
                            # Job not in memory, assume completed if old enough
                            retention = job_retention
                    
                    if age > retention:
                        # Calculate size before deletion
                        try:
                            size = sum(f.stat().st_size for f in job_dir.rglob('*') if f.is_file())
                            freed_space += size
                        except:
                            pass
                        
                        shutil.rmtree(job_dir, ignore_errors=True)
                        cleaned_files += 1
                        logger.info(f"Deleted old output directory: {job_dir} (age: {age.days} days)")
                except Exception as e:
                    logger.warning(f"Error cleaning up output directory {job_dir}: {e}")
        
        # Clean up old uploaded files
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        if upload_dir.exists():
            for file_path in upload_dir.iterdir():
                if not file_path.is_file():
                    continue
                
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    age = now - mtime
                    
                    if age > upload_retention:
                        try:
                            size = file_path.stat().st_size
                            freed_space += size
                        except:
                            pass
                        
                        file_path.unlink()
                        cleaned_files += 1
                        logger.info(f"Deleted old uploaded file: {file_path} (age: {age.days} days)")
                except Exception as e:
                    logger.warning(f"Error cleaning up uploaded file {file_path}: {e}")
        
        freed_space_mb = freed_space / (1024 * 1024)
        logger.info(
            f"Cleanup completed: {cleaned_jobs} jobs, {cleaned_files} files/dirs, "
            f"{freed_space_mb:.2f} MB freed"
        )
        
    except Exception as e:
        logger.error(f"Error in automatic cleanup: {e}", exc_info=True)
    finally:
        cleanup_running = False


def start_cleanup_thread():
    """Start background thread for automatic cleanup."""
    global cleanup_thread
    
    def cleanup_loop():
        while True:
            try:
                time.sleep(Config.CLEANUP_INTERVAL)
                cleanup_old_jobs_and_files()
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}", exc_info=True)
                time.sleep(60)  # Wait a minute before retrying
    
    if cleanup_thread is None or not cleanup_thread.is_alive():
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True, name="CleanupThread")
        cleanup_thread.start()
        logger.info(f"Started automatic cleanup thread (interval: {Config.CLEANUP_INTERVAL}s)")


@app.route('/cleanup-old', methods=['POST'])
@require_auth(Permission.MANAGE_USERS)
def manual_cleanup_old():
    """Manually trigger cleanup of old jobs and files."""
    try:
        cleanup_old_jobs_and_files()
        return jsonify({'message': 'Cleanup completed successfully'})
    except Exception as e:
        logger.error(f"Error in manual cleanup: {e}", exc_info=True)
        return jsonify({'error': 'Cleanup failed'}), 500


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
@require_auth(Permission.VIEW_METRICS)
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
    user_session = get_current_user_session()
    return render_template('user_management.html', user=user_session)

@app.route('/security-dashboard')
@require_auth(Permission.VIEW_METRICS)
def security_dashboard():
    """Security monitoring dashboard."""
    return render_template('security_dashboard.html')

@app.route('/api/security/summary')
@require_auth(Permission.VIEW_METRICS)
def security_summary():
    """Get security summary for dashboard."""
    try:
        summary = security_manager.get_security_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Failed to get security summary: {str(e)}")
        return jsonify({"error": "Failed to get security summary"}), 500

def serialize_profile(profile) -> Dict[str, Any]:
    """Serialize user profile dataclass to JSON-serializable dict."""
    profile_dict = asdict(profile)
    # Convert datetime objects to ISO format strings
    for key, value in profile_dict.items():
        if isinstance(value, datetime):
            profile_dict[key] = value.isoformat()
    return profile_dict


@app.route('/api/user/profile')
@require_auth()
def get_user_profile():
    """Get current user's profile."""
    try:
        user_session = get_current_user_session()
        profile = user_manager.get_user_profile(user_session.user_id)
        if profile:
            return jsonify(serialize_profile(profile))
        else:
            return jsonify({"error": "Profile not found"}), 404
    except Exception as e:
        logger.error(f"Failed to get user profile: {str(e)}")
        return jsonify({"error": "Failed to get profile"}), 500

@app.route('/api/user/preferences', methods=['POST'])
@require_auth()
def update_user_preferences():
    """Update user preferences."""
    try:
        user_session = get_current_user_session()
        preferences = request.get_json()
        
        success = user_manager.update_user_preferences(user_session.user_id, preferences)
        if success:
            return jsonify({"message": "Preferences updated successfully"})
        else:
            return jsonify({"error": "Failed to update preferences"}), 400
    except Exception as e:
        logger.error(f"Failed to update preferences: {str(e)}")
        return jsonify({"error": "Failed to update preferences"}), 500

@app.route('/api/user/change-password', methods=['POST'])
@require_auth()
def change_password():
    """Change user password."""
    try:
        user_session = get_current_user_session()
        data = request.get_json()
        
        result = user_manager.change_password(
            user_session.user_id,
            data.get('old_password'),
            data.get('new_password')
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to change password: {str(e)}")
        return jsonify({"success": False, "error": "Password change failed"}), 500

@app.route('/api/user/sessions')
@require_auth()
def get_user_sessions():
    """Get user's active sessions."""
    try:
        user_session = get_current_user_session()
        sessions = session_manager.get_user_sessions(user_session.user_id)
        return jsonify({"sessions": sessions})
    except Exception as e:
        logger.error(f"Failed to get user sessions: {str(e)}")
        return jsonify({"error": "Failed to get sessions"}), 500

@app.route('/api/user/sessions/<session_id>/revoke', methods=['POST'])
@require_auth()
def revoke_session(session_id):
    """Revoke a specific session."""
    try:
        user_session = get_current_user_session()
        success = session_manager.revoke_session(session_id, user_session.user_id)
        if success:
            return jsonify({"message": "Session revoked successfully"})
        else:
            return jsonify({"error": "Session not found"}), 404
    except Exception as e:
        logger.error(f"Failed to revoke session: {str(e)}")
        return jsonify({"error": "Failed to revoke session"}), 500

@app.route('/api/user/sessions/revoke-all', methods=['POST'])
@require_auth()
def revoke_all_sessions():
    """Revoke all user sessions."""
    try:
        user_session = get_current_user_session()
        count = session_manager.revoke_all_sessions(user_session.user_id, user_session.user_id)
        return jsonify({"message": f"Revoked {count} sessions successfully"})
    except Exception as e:
        logger.error(f"Failed to revoke all sessions: {str(e)}")
        return jsonify({"error": "Failed to revoke sessions"}), 500

@app.route('/api/admin/user-statistics')
@require_auth(Permission.MANAGE_USERS)
def get_user_statistics():
    """Get user statistics for admin dashboard."""
    try:
        stats = user_manager.get_user_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Failed to get user statistics: {str(e)}")
        return jsonify({"error": "Failed to get statistics"}), 500


# Search API Endpoints
@app.route('/api/search', methods=['POST'])
@require_auth(Permission.VIEW_JOB)
def search_transcripts():
    """Perform cross-language search across transcripts with multiple modes."""
    try:
        from src.video_doc.search import get_search_service
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query')
        if not query or not query.strip():
            return jsonify({'error': 'Query is required'}), 400
        
        target_language = data.get('target_language')  # Language to return results in
        job_ids = data.get('job_ids')  # Optional: filter by specific jobs
        limit = int(data.get('limit', 100))  # Increased default to support pagination
        min_score = float(data.get('min_score', 0.5))
        search_mode = data.get('search_mode', 'semantic')  # 'semantic', 'keyword', or 'hybrid'
        page = int(data.get('page', 1))
        per_page = int(data.get('per_page', 20))
        
        # Validate pagination parameters
        if page < 1:
            return jsonify({'error': 'page must be >= 1'}), 400
        if per_page < 1 or per_page > 100:
            return jsonify({'error': 'per_page must be between 1 and 100'}), 400
        
        # Validate search mode
        valid_modes = ['semantic', 'keyword', 'hybrid']
        if search_mode not in valid_modes:
            return jsonify({'error': f'Invalid search_mode. Must be one of: {", ".join(valid_modes)}'}), 400
        
        # Get hybrid search weights (optional, only used in hybrid mode)
        semantic_weight = data.get('semantic_weight')
        keyword_weight = data.get('keyword_weight')
        
        # Validate weights if provided
        if semantic_weight is not None:
            try:
                semantic_weight = float(semantic_weight)
                if not (0.0 <= semantic_weight <= 1.0):
                    return jsonify({'error': 'semantic_weight must be between 0.0 and 1.0'}), 400
            except (ValueError, TypeError):
                return jsonify({'error': 'semantic_weight must be a number'}), 400
        
        if keyword_weight is not None:
            try:
                keyword_weight = float(keyword_weight)
                if not (0.0 <= keyword_weight <= 1.0):
                    return jsonify({'error': 'keyword_weight must be between 0.0 and 1.0'}), 400
            except (ValueError, TypeError):
                return jsonify({'error': 'keyword_weight must be a number'}), 400
        
        # Advanced filtering parameters
        date_from_str = data.get('date_from')
        date_to_str = data.get('date_to')
        job_type = data.get('job_type')
        job_status = data.get('job_status')
        original_language = data.get('original_language')
        
        # Parse date strings to datetime objects
        date_from = None
        date_to = None
        if date_from_str:
            try:
                date_from = datetime.fromisoformat(date_from_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return jsonify({'error': 'Invalid date_from format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)'}), 400
        
        if date_to_str:
            try:
                date_to = datetime.fromisoformat(date_to_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return jsonify({'error': 'Invalid date_to format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)'}), 400
        
        # Validate job_type
        if job_type and job_type not in ['url', 'file']:
            return jsonify({'error': 'job_type must be "url" or "file"'}), 400
        
        # Validate job_status
        if job_status and job_status not in ['pending', 'processing', 'completed', 'failed', 'cancelled']:
            return jsonify({'error': 'Invalid job_status'}), 400
        
        search_service = get_search_service()
        search_result = search_service.search(
            query=query,
            target_language=target_language,
            job_ids=job_ids,
            limit=limit,
            min_score=min_score,
            search_mode=search_mode,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            date_from=date_from,
            date_to=date_to,
            job_type=job_type,
            job_status=job_status,
            original_language=original_language,
            page=page,
            per_page=per_page
        )
        
        response_data = {
            'query': query,
            'target_language': target_language,
            'search_mode': search_mode,
            'results': search_result['results'],
            'count': len(search_result['results']),
            'total': search_result['total'],
            'page': search_result['page'],
            'per_page': search_result['per_page'],
            'total_pages': search_result['total_pages']
        }
        
        # Include weights in response if hybrid mode was used
        if search_mode == 'hybrid':
            response_data['semantic_weight'] = semantic_weight if semantic_weight is not None else 0.6
            response_data['keyword_weight'] = keyword_weight if keyword_weight is not None else 0.4
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({'error': 'Search failed', 'message': str(e)}), 500


@app.route('/api/search/index/<job_id>', methods=['POST'])
@require_auth(Permission.VIEW_JOB)
def index_job(job_id):
    """Manually trigger indexing for a completed job."""
    try:
        from src.video_doc.search import get_search_service
        from pathlib import Path
        
        # Check if job exists and is completed
        with job_lock:
            if job_id not in processing_jobs:
                return jsonify({'error': 'Job not found'}), 404
            
            job = processing_jobs[job_id]
            if job.status != 'completed':
                return jsonify({'error': 'Job must be completed before indexing'}), 400
        
        # Find segments.json file
        output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
        segments_json = output_dir / 'transcript' / 'segments.json'
        
        if not segments_json.exists():
            return jsonify({'error': 'Transcript not found for this job'}), 404
        
        # Index the transcript
        search_service = get_search_service()
        success = search_service.index_transcript(
            job_id=job_id,
            segments_json_path=segments_json
        )
        
        if success:
            return jsonify({
                'message': 'Indexing started successfully',
                'job_id': job_id
            })
        else:
            return jsonify({'error': 'Indexing failed'}), 500
            
    except Exception as e:
        logger.error(f"Indexing error: {e}", exc_info=True)
        return jsonify({'error': 'Indexing failed', 'message': str(e)}), 500


@app.route('/api/search/index-status/<job_id>')
@require_auth(Permission.VIEW_JOB)
def get_index_status(job_id):
    """Get indexing status for a job."""
    try:
        from src.video_doc.search import SearchIndex
        from src.video_doc.database import get_db_session
        import uuid
        
        db = get_db_session()
        try:
            try:
                job_uuid = uuid.UUID(job_id)
            except ValueError:
                return jsonify({'error': 'Invalid job ID'}), 400
            
            search_index = db.query(SearchIndex).filter(
                SearchIndex.job_id == job_uuid
            ).first()
            
            if not search_index:
                return jsonify({
                    'indexed': False,
                    'status': 'not_indexed'
                })
            
            return jsonify({
                'indexed': search_index.status == 'completed',
                'status': search_index.status,
                'total_chunks': search_index.total_chunks,
                'indexed_chunks': search_index.indexed_chunks,
                'indexed_at': search_index.indexed_at.isoformat() if search_index.indexed_at else None,
                'error_message': search_index.error_message
            })
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting index status: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get index status'}), 500


@app.route('/api/search/jobs')
@require_auth(Permission.VIEW_JOB)
def list_indexed_jobs():
    """List all jobs that have been indexed."""
    try:
        from src.video_doc.search import SearchIndex
        from src.video_doc.database import get_db_session
        
        db = get_db_session()
        try:
            indexes = db.query(SearchIndex).filter(
                SearchIndex.status == 'completed'
            ).all()
            
            jobs = [{
                'job_id': str(idx.job_id),
                'total_chunks': idx.total_chunks,
                'indexed_at': idx.indexed_at.isoformat() if idx.indexed_at else None
            } for idx in indexes]
            
            return jsonify({
                'jobs': jobs,
                'count': len(jobs)
            })
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error listing indexed jobs: {e}", exc_info=True)
        return jsonify({'error': 'Failed to list indexed jobs'}), 500


@app.route('/api/search/context/<chunk_id>')
@require_auth(Permission.VIEW_JOB)
def get_chunk_context(chunk_id):
    """Get surrounding chunks for context around a search result."""
    try:
        from src.video_doc.search import get_search_service
        
        # Get optional parameters
        context_before = int(request.args.get('context_before', 2))
        context_after = int(request.args.get('context_after', 2))
        target_language = request.args.get('target_language') or None
        
        # Validate parameters
        if context_before < 0 or context_before > 10:
            return jsonify({'error': 'context_before must be between 0 and 10'}), 400
        if context_after < 0 or context_after > 10:
            return jsonify({'error': 'context_after must be between 0 and 10'}), 400
        
        search_service = get_search_service()
        context = search_service.get_chunk_context(
            chunk_id=chunk_id,
            context_before=context_before,
            context_after=context_after,
            target_language=target_language
        )
        
        if 'error' in context:
            return jsonify(context), 404
        
        return jsonify(context)
        
    except Exception as e:
        logger.error(f"Error getting chunk context: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get chunk context', 'message': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    try:
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'message': 'Connected to video processing server'})
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


if __name__ == '__main__':
    # Initialize start time for uptime tracking
    app.config['start_time'] = time.time()
    
    logger.info("Starting Video Documentation Builder Web Interface...")
    logger.info(f"Configuration: DEBUG={Config.DEBUG}, HOST={Config.HOST}, PORT={Config.PORT}")
    logger.info(f"Max file size: {Config.MAX_CONTENT_LENGTH / (1024**3):.1f}GB")
    logger.info(f"Max concurrent jobs: {Config.MAX_CONCURRENT_JOBS}")
    logger.info(f"Job timeout: {Config.JOB_TIMEOUT}s")
    logger.info(f"Cleanup interval: {Config.CLEANUP_INTERVAL}s")
    logger.info(f"Job retention: {Config.JOB_RETENTION_DAYS} days (completed), {Config.FAILED_JOB_RETENTION_DAYS} days (failed)")
    logger.info(f"Upload retention: {Config.UPLOAD_RETENTION_DAYS} days")
    
    # Start automatic cleanup thread
    start_cleanup_thread()
    
    print("\n" + "="*80)
    print("🚀 VIDEO PROCESSING SYSTEM STARTED SUCCESSFULLY!")
    print("="*80)
    print(f"📊 Web Interface: http://{Config.HOST}:{Config.PORT}")
    print(f"🔐 Login Page: http://{Config.HOST}:{Config.PORT}/login")
    print(f"👤 User Management: http://{Config.HOST}:{Config.PORT}/user-management")
    print(f"🛡️ Security Dashboard: http://{Config.HOST}:{Config.PORT}/security-dashboard")
    print(f"📚 API Documentation: http://{Config.HOST}:{Config.PORT}/api/docs")
    print(f"💚 Health Dashboard: http://{Config.HOST}:{Config.PORT}/health-dashboard")
    print(f"🔍 Health Check: http://{Config.HOST}:{Config.PORT}/health")
    print(f"📈 Metrics: http://{Config.HOST}:{Config.PORT}/metrics")
    print("="*80)
    print("🔑 DEFAULT ADMIN CREDENTIALS:")
    print("   Username: admin")
    print("   Password: admin123")
    print("="*80)
    print("🛡️ SECURITY FEATURES:")
    print("   ✅ Rate limiting and suspicious activity detection")
    print("   ✅ Password policy enforcement")
    print("   ✅ Session management and revocation")
    print("   ✅ API key management")
    print("   ✅ Comprehensive audit logging")
    print("="*80)
    print("📚 API FEATURES:")
    print("   ✅ Interactive Swagger documentation")
    print("   ✅ Permission-based endpoint access")
    print("   ✅ Client code generation")
    print("   ✅ Real-time testing interface")
    print("="*80)
    
    try:
        socketio.run(app, debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise
