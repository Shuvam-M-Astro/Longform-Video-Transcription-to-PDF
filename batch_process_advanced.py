#!/usr/bin/env python3
"""
Advanced Batch Video Processing Script

Process multiple videos with YAML configuration support, better error handling,
and advanced features like resume capability and detailed reporting.

Usage:
    python batch_process_advanced.py --config batch_config.yaml
    python batch_process_advanced.py --input-file videos.txt --output-dir ./output --parallel 3
"""

import argparse
import json
import os
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import threading
from datetime import datetime
import signal
import subprocess

# Import the main processing function
from main import main as process_single_video
from main import parse_args as parse_single_args


@dataclass
class BatchItem:
    """Represents a single video to be processed in batch mode."""
    identifier: str
    input_type: str  # 'url', 'file', or 'custom'
    output_subdir: str
    status: str = 'pending'  # pending, processing, completed, failed, skipped, cancelled
    error_message: str = ''
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    output_files: Dict[str, str] = None
    retry_count: int = 0
    priority: int = 0  # Higher number = higher priority
    
    def __post_init__(self):
        if self.output_files is None:
            self.output_files = {}


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_parallel: int = 1
    retry_failed: bool = True
    max_retries: int = 2
    skip_existing: bool = True
    stop_on_error: bool = False
    log_level: str = 'INFO'
    progress_interval: float = 5.0
    timeout_per_video: Optional[int] = None
    cleanup_temp: bool = True
    create_subdirs: bool = True
    preserve_structure: bool = False
    
    # Processing options
    transcribe_only: bool = False
    language: str = 'auto'
    whisper_model: str = 'medium'
    beam_size: int = 5
    export_srt: bool = False
    report_style: str = 'book'
    
    # Keyframe options
    kf_method: str = 'scene'
    max_fps: float = 1.0
    max_frames: int = 0
    scene_threshold: float = 0.45
    interval_sec: float = 5.0
    
    # Download options
    cookies_from_browser: Optional[str] = None
    browser_profile: Optional[str] = None
    cookies_file: Optional[str] = None
    use_android_client: bool = False


class AdvancedBatchProcessor:
    """Advanced batch processor with YAML config support and enhanced features."""
    
    def __init__(self, config: BatchConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe collections
        self.items: List[BatchItem] = []
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.cancelled_count = 0
        self.lock = threading.Lock()
        
        # Progress tracking
        self.start_time = None
        self.last_progress_time = 0
        self.is_cancelled = False
        
        # Logging
        self.log_file = self.output_dir / "batch_log.txt"
        self.results_file = self.output_dir / "batch_results.json"
        self.state_file = self.output_dir / "batch_state.json"
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.log(f"Received signal {signum}, initiating graceful shutdown...", 'WARNING')
        self.is_cancelled = True
    
    def add_item(self, identifier: str, input_type: str, output_subdir: str = None, priority: int = 0):
        """Add a video item to the batch processing queue."""
        if output_subdir is None:
            if input_type == 'url':
                import hashlib
                safe_id = hashlib.md5(identifier.encode()).hexdigest()[:8]
                output_subdir = f"video_{safe_id}"
            else:
                output_subdir = Path(identifier).stem.replace(' ', '_').replace('-', '_')
        
        item = BatchItem(
            identifier=identifier,
            input_type=input_type,
            output_subdir=output_subdir,
            priority=priority
        )
        self.items.append(item)
        return item
    
    def add_from_file(self, file_path: Path):
        """Add videos from a text file (one per line)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse priority if specified (format: "priority:identifier")
                priority = 0
                if ':' in line and not line.startswith(('http://', 'https://')):
                    try:
                        priority, identifier = line.split(':', 1)
                        priority = int(priority.strip())
                        identifier = identifier.strip()
                    except ValueError:
                        identifier = line
                else:
                    identifier = line
                
                # Determine if it's a URL or file path
                if identifier.startswith(('http://', 'https://')):
                    input_type = 'url'
                elif Path(identifier).exists():
                    input_type = 'file'
                else:
                    self.log(f"Warning: Line {line_num} is neither a valid URL nor existing file: {identifier}")
                    continue
                
                self.add_item(identifier, input_type, priority=priority)
    
    def add_from_directory(self, dir_path: Path, extensions: List[str] = None, recursive: bool = True):
        """Add all video files from a directory."""
        if extensions is None:
            extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v', '.flv', '.wmv']
        
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        pattern = "**/*" if recursive else "*"
        for ext in extensions:
            for video_file in dir_path.glob(f"{pattern}{ext}"):
                # Preserve directory structure if configured
                if self.config.preserve_structure:
                    rel_path = video_file.relative_to(dir_path)
                    output_subdir = str(rel_path.parent) if rel_path.parent != Path('.') else video_file.stem
                else:
                    output_subdir = video_file.stem
                
                self.add_item(str(video_file), 'file', output_subdir)
    
    def add_from_urls(self, urls: List[str]):
        """Add videos from a list of URLs."""
        for url in urls:
            self.add_item(url, 'url')
    
    def load_state(self):
        """Load previous batch state if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # Restore items
                for item_data in state.get('items', []):
                    item = BatchItem(**item_data)
                    self.items.append(item)
                
                # Restore counters
                self.completed_count = state.get('completed_count', 0)
                self.failed_count = state.get('failed_count', 0)
                self.skipped_count = state.get('skipped_count', 0)
                
                self.log(f"Loaded previous state: {len(self.items)} items, {self.completed_count} completed")
                return True
            except Exception as e:
                self.log(f"Failed to load state: {e}", 'WARNING')
        return False
    
    def save_state(self):
        """Save current batch state."""
        state = {
            'items': [asdict(item) for item in self.items],
            'completed_count': self.completed_count,
            'failed_count': self.failed_count,
            'skipped_count': self.skipped_count,
            'timestamp': time.time()
        }
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def log(self, message: str, level: str = 'INFO'):
        """Thread-safe logging."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        with self.lock:
            print(log_entry, flush=True)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
    
    def process_single_item(self, item: BatchItem) -> bool:
        """Process a single video item."""
        if self.is_cancelled:
            with self.lock:
                item.status = 'cancelled'
                self.cancelled_count += 1
            return False
        
        try:
            with self.lock:
                item.status = 'processing'
                item.start_time = time.time()
            
            self.log(f"Processing: {item.identifier}")
            
            # Create output directory for this item
            if self.config.create_subdirs:
                item_output_dir = self.output_dir / item.output_subdir
            else:
                item_output_dir = self.output_dir
            item_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already processed (if skip_existing is enabled)
            if self.config.skip_existing:
                existing_pdf = item_output_dir / "report.pdf"
                if existing_pdf.exists():
                    with self.lock:
                        item.status = 'skipped'
                        item.end_time = time.time()
                        item.duration_seconds = item.end_time - item.start_time
                        self.skipped_count += 1
                    self.log(f"Skipped (already exists): {item.identifier}")
                    return True
            
            # Prepare arguments for this specific video
            item_args = self._build_args_for_item(item, item_output_dir)
            
            # Process the video
            start_time = time.time()
            
            # Override sys.argv temporarily to pass arguments to main()
            original_argv = sys.argv.copy()
            sys.argv = ['main.py'] + item_args
            
            try:
                process_single_video()
                success = True
            except Exception as e:
                self.log(f"Error processing {item.identifier}: {str(e)}", 'ERROR')
                success = False
                with self.lock:
                    item.error_message = str(e)
            finally:
                sys.argv = original_argv
            
            end_time = time.time()
            
            with self.lock:
                item.end_time = end_time
                item.duration_seconds = end_time - start_time
                
                if success:
                    item.status = 'completed'
                    self.completed_count += 1
                    
                    # Record output files
                    self._record_output_files(item, item_output_dir)
                else:
                    item.status = 'failed'
                    self.failed_count += 1
            
            self.log(f"Completed: {item.identifier} ({item.duration_seconds:.1f}s)")
            return success
            
        except Exception as e:
            with self.lock:
                item.status = 'failed'
                item.error_message = str(e)
                item.end_time = time.time()
                if item.start_time:
                    item.duration_seconds = item.end_time - item.start_time
                self.failed_count += 1
            
            self.log(f"Failed: {item.identifier} - {str(e)}", 'ERROR')
            return False
    
    def _build_args_for_item(self, item: BatchItem, output_dir: Path) -> List[str]:
        """Build command line arguments for a specific item."""
        args = []
        
        # Input specification
        if item.input_type == 'url':
            args.extend(['--url', item.identifier])
        else:
            args.extend(['--video', item.identifier])
        
        # Output directory
        args.extend(['--out', str(output_dir)])
        
        # Processing options
        if self.config.transcribe_only:
            args.append('--transcribe-only')
        if self.config.language != 'auto':
            args.extend(['--language', self.config.language])
        if self.config.whisper_model != 'medium':
            args.extend(['--whisper-model', self.config.whisper_model])
        if self.config.beam_size != 5:
            args.extend(['--beam-size', str(self.config.beam_size)])
        if self.config.export_srt:
            args.append('--export-srt')
        if self.config.report_style != 'book':
            args.extend(['--report-style', self.config.report_style])
        
        # Keyframe options
        if self.config.kf_method != 'scene':
            args.extend(['--kf-method', self.config.kf_method])
        if self.config.max_fps != 1.0:
            args.extend(['--max-fps', str(self.config.max_fps)])
        if self.config.max_frames > 0:
            args.extend(['--frame-max-frames', str(self.config.max_frames)])
        if self.config.scene_threshold != 0.45:
            args.extend(['--min-scene-diff', str(self.config.scene_threshold)])
        if self.config.interval_sec != 5.0:
            args.extend(['--kf-interval-sec', str(self.config.interval_sec)])
        
        # Download options
        if self.config.cookies_from_browser:
            args.extend(['--cookies-from-browser', self.config.cookies_from_browser])
        if self.config.browser_profile:
            args.extend(['--browser-profile', self.config.browser_profile])
        if self.config.cookies_file:
            args.extend(['--cookies-file', self.config.cookies_file])
        if self.config.use_android_client:
            args.append('--use-android-client')
        
        # Add resume flag to avoid cleaning existing work
        args.append('--resume')
        
        return args
    
    def _record_output_files(self, item: BatchItem, output_dir: Path):
        """Record output files for a completed item."""
        output_files = {
            'pdf': 'report.pdf',
            'transcript': 'transcript/transcript.txt',
            'audio': 'audio.wav',
            'segments': 'transcript/segments.json'
        }
        
        for file_type, rel_path in output_files.items():
            full_path = output_dir / rel_path
            if full_path.exists():
                item.output_files[file_type] = str(full_path)
    
    def print_progress(self):
        """Print current progress status."""
        with self.lock:
            total = len(self.items)
            completed = self.completed_count
            failed = self.failed_count
            skipped = self.skipped_count
            cancelled = self.cancelled_count
            processing = sum(1 for item in self.items if item.status == 'processing')
            pending = sum(1 for item in self.items if item.status == 'pending')
            
            elapsed = time.time() - self.start_time if self.start_time else 0
            
            print(f"\n{'='*70}")
            print(f"Advanced Batch Processing Progress")
            print(f"{'='*70}")
            print(f"Total items: {total}")
            print(f"Completed: {completed} ({completed/total*100:.1f}%)")
            print(f"Failed: {failed} ({failed/total*100:.1f}%)")
            print(f"Skipped: {skipped} ({skipped/total*100:.1f}%)")
            print(f"Cancelled: {cancelled} ({cancelled/total*100:.1f}%)")
            print(f"Processing: {processing}")
            print(f"Pending: {pending}")
            print(f"Elapsed time: {elapsed:.1f}s")
            
            if completed > 0:
                avg_time = sum(item.duration_seconds or 0 for item in self.items if item.status == 'completed') / completed
                print(f"Average processing time: {avg_time:.1f}s")
            
            if processing > 0:
                print(f"\nCurrently processing:")
                for item in self.items:
                    if item.status == 'processing':
                        elapsed_item = time.time() - (item.start_time or 0)
                        print(f"  - {item.identifier} ({elapsed_item:.1f}s)")
            
            if failed > 0:
                print(f"\nFailed items:")
                for item in self.items:
                    if item.status == 'failed':
                        print(f"  - {item.identifier}: {item.error_message}")
            
            print(f"{'='*70}\n")
    
    def process_all(self):
        """Process all items in the batch."""
        if not self.items:
            self.log("No items to process", 'WARNING')
            return
        
        # Sort items by priority (higher priority first)
        self.items.sort(key=lambda x: x.priority, reverse=True)
        
        self.start_time = time.time()
        self.log(f"Starting advanced batch processing of {len(self.items)} items")
        self.log(f"Configuration: parallel={self.config.max_parallel}, retry_failed={self.config.retry_failed}")
        
        # Process items
        if self.config.max_parallel == 1:
            # Sequential processing
            for i, item in enumerate(self.items):
                if self.is_cancelled:
                    self.log("Processing cancelled by user", 'WARNING')
                    break
                
                if self.config.stop_on_error and self.failed_count > 0:
                    self.log("Stopping due to previous error", 'WARNING')
                    break
                
                if item.status in ['completed', 'skipped']:
                    continue
                
                self.log(f"Processing item {i+1}/{len(self.items)}: {item.identifier}")
                self.process_single_item(item)
                
                # Save state periodically
                if i % 10 == 0:
                    self.save_state()
                
                # Progress update
                if time.time() - self.last_progress_time >= self.config.progress_interval:
                    self.print_progress()
                    self.last_progress_time = time.time()
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
                # Submit all tasks
                future_to_item = {}
                for item in self.items:
                    if item.status not in ['completed', 'skipped']:
                        future = executor.submit(self.process_single_item, item)
                        future_to_item[future] = item
                
                # Process completed tasks
                for future in as_completed(future_to_item):
                    if self.is_cancelled:
                        self.log("Processing cancelled by user", 'WARNING')
                        # Cancel remaining futures
                        for f in future_to_item:
                            f.cancel()
                        break
                    
                    item = future_to_item[future]
                    
                    try:
                        success = future.result(timeout=self.config.timeout_per_video)
                    except Exception as e:
                        self.log(f"Unexpected error processing {item.identifier}: {str(e)}", 'ERROR')
                        with self.lock:
                            item.status = 'failed'
                            item.error_message = str(e)
                            self.failed_count += 1
                    
                    # Save state periodically
                    if (self.completed_count + self.failed_count) % 10 == 0:
                        self.save_state()
                    
                    # Progress update
                    if time.time() - self.last_progress_time >= self.config.progress_interval:
                        self.print_progress()
                        self.last_progress_time = time.time()
                    
                    # Stop on error if configured
                    if self.config.stop_on_error and self.failed_count > 0:
                        self.log("Stopping due to error", 'WARNING')
                        # Cancel remaining futures
                        for f in future_to_item:
                            f.cancel()
                        break
        
        # Retry failed items if configured
        if self.config.retry_failed and self.failed_count > 0 and not self.is_cancelled:
            self.log(f"Retrying {self.failed_count} failed items...")
            failed_items = [item for item in self.items if item.status == 'failed']
            
            for retry_attempt in range(self.config.max_retries):
                if not failed_items or self.is_cancelled:
                    break
                
                self.log(f"Retry attempt {retry_attempt + 1}/{self.config.max_retries}")
                retry_items = failed_items.copy()
                failed_items = []
                
                for item in retry_items:
                    if self.is_cancelled:
                        break
                    
                    # Reset item status
                    item.status = 'pending'
                    item.error_message = ''
                    item.retry_count += 1
                    
                    success = self.process_single_item(item)
                    if not success:
                        failed_items.append(item)
                
                if not failed_items:
                    self.log("All retries successful!")
                    break
        
        # Final results
        self.print_progress()
        self.save_results()
        
        # Cleanup
        if self.config.cleanup_temp:
            self._cleanup_temp_files()
        
        # Final summary
        total_time = time.time() - self.start_time
        self.log(f"Advanced batch processing completed in {total_time:.1f}s")
        self.log(f"Results: {self.completed_count} completed, {self.failed_count} failed, {self.skipped_count} skipped, {self.cancelled_count} cancelled")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        self.log("Cleaning up temporary files...")
        # Add cleanup logic here if needed
    
    def save_results(self):
        """Save batch processing results to JSON file."""
        results = {
            'config': asdict(self.config),
            'summary': {
                'total_items': len(self.items),
                'completed': self.completed_count,
                'failed': self.failed_count,
                'skipped': self.skipped_count,
                'cancelled': self.cancelled_count,
                'start_time': self.start_time,
                'end_time': time.time(),
                'total_duration': time.time() - self.start_time if self.start_time else 0
            },
            'items': [asdict(item) for item in self.items]
        }
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def load_config_from_yaml(config_path: Path) -> BatchConfig:
    """Load batch configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Extract batch settings
    batch_settings = config_data.get('batch', {})
    processing_settings = config_data.get('processing', {})
    keyframe_settings = config_data.get('keyframes', {})
    download_settings = config_data.get('download', {})
    
    # Create BatchConfig object
    config = BatchConfig()
    
    # Apply batch settings
    for key, value in batch_settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Apply processing settings
    for key, value in processing_settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Apply keyframe settings
    for key, value in keyframe_settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Apply download settings
    for key, value in download_settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def parse_advanced_args():
    """Parse command line arguments for advanced batch processing."""
    parser = argparse.ArgumentParser(
        description="Advanced batch process multiple videos into PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with YAML configuration
  python batch_process_advanced.py --config batch_config.yaml
  
  # Process videos from a text file
  python batch_process_advanced.py --input-file videos.txt --output-dir ./batch_output --parallel 2
  
  # Process all videos in a directory
  python batch_process_advanced.py --input-dir ./videos --output-dir ./batch_output --parallel 3 --recursive
  
  # Process with custom settings
  python batch_process_advanced.py --input-file videos.txt --output-dir ./batch_output \\
    --parallel 3 --transcribe-only --whisper-model small --beam-size 1
        """
    )
    
    # Configuration file
    parser.add_argument('--config', type=str, help='YAML configuration file')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--input-file', type=str, help='Text file containing video URLs/paths (one per line)')
    input_group.add_argument('--input-dir', type=str, help='Directory containing video files')
    input_group.add_argument('--urls', type=str, help='Comma-separated list of video URLs')
    
    # Output options
    parser.add_argument('--output-dir', type=str, help='Output directory for batch results')
    
    # Batch processing options
    parser.add_argument('--parallel', type=int, help='Number of parallel processes')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed items')
    parser.add_argument('--max-retries', type=int, help='Maximum retry attempts for failed items')
    parser.add_argument('--skip-existing', action='store_true', help='Skip items that already have output')
    parser.add_argument('--stop-on-error', action='store_true', help='Stop processing on first error')
    parser.add_argument('--timeout', type=int, help='Timeout per video in seconds')
    parser.add_argument('--progress-interval', type=float, help='Progress update interval in seconds')
    parser.add_argument('--recursive', action='store_true', help='Recursively search input directory')
    
    # Processing options
    parser.add_argument('--transcribe-only', action='store_true', help='Skip frames/classification; transcript-only PDF')
    parser.add_argument('--language', type=str, help='Language code or auto')
    parser.add_argument('--whisper-model', type=str, help='faster-whisper model size')
    parser.add_argument('--beam-size', type=int, help='Beam size for decoding')
    parser.add_argument('--export-srt', action='store_true', help='Also export transcript in SubRip (.srt)')
    parser.add_argument('--report-style', type=str, choices=['minimal', 'book'], help='PDF layout style')
    
    # Keyframe options
    parser.add_argument('--kf-method', type=str, choices=['scene', 'iframe', 'interval'], help='Keyframe extraction method')
    parser.add_argument('--max-fps', type=float, help='Max FPS for keyframe detection')
    parser.add_argument('--max-frames', type=int, help='Cap total saved frames; 0 to disable')
    parser.add_argument('--scene-threshold', type=float, help='Scene change threshold [0-1]')
    parser.add_argument('--interval-sec', type=float, help='Interval seconds for interval method')
    
    # Download options
    parser.add_argument('--cookies-from-browser', type=str, help='Browser to read cookies from')
    parser.add_argument('--browser-profile', type=str, help='Specific browser profile name')
    parser.add_argument('--cookies-file', type=str, help='Path to cookies.txt file')
    parser.add_argument('--use-android-client', action='store_true', help='Use YouTube Android client fallback')
    
    return parser.parse_args()


def main():
    """Main entry point for advanced batch processing."""
    args = parse_advanced_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_yaml(Path(args.config))
        output_dir = Path(config.output_dir if hasattr(config, 'output_dir') else args.output_dir)
    else:
        # Create default config
        config = BatchConfig()
        output_dir = Path(args.output_dir) if args.output_dir else Path('./batch_output')
    
    # Override config with command line arguments
    if args.parallel is not None:
        config.max_parallel = args.parallel
    if args.retry_failed:
        config.retry_failed = args.retry_failed
    if args.max_retries is not None:
        config.max_retries = args.max_retries
    if args.skip_existing:
        config.skip_existing = args.skip_existing
    if args.stop_on_error:
        config.stop_on_error = args.stop_on_error
    if args.timeout is not None:
        config.timeout_per_video = args.timeout
    if args.progress_interval is not None:
        config.progress_interval = args.progress_interval
    
    # Processing options
    if args.transcribe_only:
        config.transcribe_only = args.transcribe_only
    if args.language:
        config.language = args.language
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    if args.beam_size is not None:
        config.beam_size = args.beam_size
    if args.export_srt:
        config.export_srt = args.export_srt
    if args.report_style:
        config.report_style = args.report_style
    
    # Keyframe options
    if args.kf_method:
        config.kf_method = args.kf_method
    if args.max_fps is not None:
        config.max_fps = args.max_fps
    if args.max_frames is not None:
        config.max_frames = args.max_frames
    if args.scene_threshold is not None:
        config.scene_threshold = args.scene_threshold
    if args.interval_sec is not None:
        config.interval_sec = args.interval_sec
    
    # Download options
    if args.cookies_from_browser:
        config.cookies_from_browser = args.cookies_from_browser
    if args.browser_profile:
        config.browser_profile = args.browser_profile
    if args.cookies_file:
        config.cookies_file = args.cookies_file
    if args.use_android_client:
        config.use_android_client = args.use_android_client
    
    # Create batch processor
    processor = AdvancedBatchProcessor(config, output_dir)
    
    # Try to load previous state
    processor.load_state()
    
    # Add items based on input type
    try:
        if args.input_file:
            processor.add_from_file(Path(args.input_file))
        elif args.input_dir:
            processor.add_from_directory(Path(args.input_dir), recursive=args.recursive)
        elif args.urls:
            urls = [url.strip() for url in args.urls.split(',') if url.strip()]
            processor.add_from_urls(urls)
    except Exception as e:
        print(f"Error loading input: {e}")
        sys.exit(1)
    
    if not processor.items:
        print("No items to process")
        sys.exit(1)
    
    # Process all items
    try:
        processor.process_all()
        
        # Exit with appropriate code
        if processor.failed_count > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nAdvanced batch processing interrupted by user")
        processor.save_results()
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error during advanced batch processing: {e}")
        processor.save_results()
        sys.exit(1)


if __name__ == '__main__':
    main()
