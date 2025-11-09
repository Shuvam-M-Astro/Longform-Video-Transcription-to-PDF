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
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
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


@app.route('/')
@optional_auth
def index():
    """Main page with upload form."""
    user_session = get_current_user_session()
    if not user_session:
        return render_template('login.html')
    return render_template('index.html', user=user_session)


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
    """Get job status."""
    with job_lock:
        if job_id not in processing_jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = processing_jobs[job_id]
        return jsonify({
            'job_id': job_id,
            'status': job.status,
            'progress': job.progress,
            'current_step': job.current_step,
            'error_message': job.error_message,
            'start_time': job.start_time,
            'end_time': job.end_time,
            'duration': job.end_time - job.start_time if job.end_time and job.start_time else None
        })


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
    """List all processing jobs."""
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

@app.route('/api/user/profile')
@require_auth()
def get_user_profile():
    """Get current user's profile."""
    try:
        user_session = get_current_user_session()
        profile = user_manager.get_user_profile(user_session.user_id)
        if profile:
            return jsonify(asdict(profile))
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
    """Perform cross-language semantic search across transcripts."""
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
        limit = int(data.get('limit', 10))
        min_score = float(data.get('min_score', 0.5))
        
        search_service = get_search_service()
        results = search_service.search(
            query=query,
            target_language=target_language,
            job_ids=job_ids,
            limit=limit,
            min_score=min_score
        )
        
        return jsonify({
            'query': query,
            'target_language': target_language,
            'results': results,
            'count': len(results)
        })
        
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
    logger.info("Starting Video Documentation Builder Web Interface...")
    logger.info(f"Configuration: DEBUG={Config.DEBUG}, HOST={Config.HOST}, PORT={Config.PORT}")
    logger.info(f"Max file size: {Config.MAX_CONTENT_LENGTH / (1024**3):.1f}GB")
    logger.info(f"Max concurrent jobs: {Config.MAX_CONCURRENT_JOBS}")
    logger.info(f"Job timeout: {Config.JOB_TIMEOUT}s")
    
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
