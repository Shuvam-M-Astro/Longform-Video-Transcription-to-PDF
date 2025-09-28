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
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_socketio import SocketIO, emit
import yaml

app = Flask(__name__)
app.config['SECRET_KEY'] = 'video-doc-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'web_outputs'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size

socketio = SocketIO(app, cors_allowed_origins="*")

# Global storage for processing jobs
processing_jobs: Dict[str, Dict[str, Any]] = {}
job_lock = threading.Lock()

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)


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
        
    def start_processing(self):
        """Start processing in a separate thread."""
        self.thread = threading.Thread(target=self._process_video)
        self.thread.daemon = True
        self.thread.start()
        
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
                    
                def start_step(self, step_name: str, weight: float):
                    self.current_step = step_name
                    self.step_progress = 0.0
                    self.job.current_step = step_name
                    self.job._emit_step_update(step_name)
                    
                def update(self, progress: float):
                    self.step_progress = progress
                    # Calculate total progress based on step weights
                    step_weight = self.step_weights.get(self.current_step.lower().split()[0], 10)
                    current_weight_progress = (progress / 100.0) * step_weight
                    self.total_progress = min(95.0, self.completed_weight + current_weight_progress)
                    self.job.progress = self.total_progress
                    self.job._emit_progress_update()
                    
                def end_step(self):
                    step_weight = self.step_weights.get(self.current_step.lower().split()[0], 10)
                    self.completed_weight += step_weight
                    self.total_progress = min(95.0, self.completed_weight)
                    self.job.progress = self.total_progress
                    self.job._emit_progress_update()
            
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
            
        except Exception as e:
            self.status = 'failed'
            self.error_message = str(e)
            self.end_time = time.time()
            self._emit_status_update()
            print(f"Processing error: {e}")  # Log error for debugging
            
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
def index():
    """Main page with upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = f"{job_id}_{file.filename}"
    file_path = Path(app.config['UPLOAD_FOLDER']) / filename
    file.save(file_path)
    
    # Get processing options from form
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
    
    # Create processing job
    job = ProcessingJob(job_id, 'file', str(file_path), options)
    
    with job_lock:
        processing_jobs[job_id] = job
    
    # Start processing
    job.start_processing()
    
    return jsonify({
        'job_id': job_id,
        'status': 'started',
        'message': 'File uploaded and processing started'
    })


@app.route('/process_url', methods=['POST'])
def process_url():
    """Handle URL processing."""
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Get processing options
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
    
    # Create processing job
    job = ProcessingJob(job_id, 'url', url, options)
    
    with job_lock:
        processing_jobs[job_id] = job
    
    # Start processing
    job.start_processing()
    
    return jsonify({
        'job_id': job_id,
        'status': 'started',
        'message': 'URL processing started'
    })


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


@app.route('/jobs')
def list_jobs():
    """List all processing jobs."""
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
                'duration': job.end_time - job.start_time if job.end_time and job.start_time else None
            })
    
    return jsonify({'jobs': jobs})


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to video processing server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")


if __name__ == '__main__':
    print("Starting Video Documentation Builder Web Interface...")
    print("Open your browser and go to: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
