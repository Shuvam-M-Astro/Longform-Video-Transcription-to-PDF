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

# Import the main processing function
from main import main as process_single_video
from main import parse_args as parse_single_args

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
            
            # Create a custom progress callback
            def progress_callback(progress: float):
                self.progress = progress
                self._emit_progress_update()
                
            # Override the progress callback in the main function
            # We'll need to modify the main function to accept this callback
            
            # Process the video
            self._emit_step_update("Starting video processing...")
            
            # For now, we'll use a simplified approach
            # In a real implementation, you'd modify main.py to accept progress callbacks
            self._emit_step_update("Downloading/processing video...")
            time.sleep(2)  # Simulate processing
            
            self._emit_step_update("Transcribing audio...")
            time.sleep(3)  # Simulate transcription
            
            self._emit_step_update("Extracting keyframes...")
            time.sleep(2)  # Simulate keyframe extraction
            
            self._emit_step_update("Classifying content...")
            time.sleep(1)  # Simulate classification
            
            self._emit_step_update("Generating PDF report...")
            time.sleep(2)  # Simulate PDF generation
            
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
                resume=False
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
                resume=False
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
