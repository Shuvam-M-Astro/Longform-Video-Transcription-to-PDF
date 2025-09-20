#!/usr/bin/env python3
"""
Batch Video Processing Script

Process multiple videos in batch mode with progress tracking, error handling,
and parallel processing capabilities.

Usage:
    python batch_process.py --input-file videos.txt --output-dir ./batch_output
    python batch_process.py --input-dir ./videos --output-dir ./batch_output --parallel 2
    python batch_process.py --urls "url1,url2,url3" --output-dir ./batch_output
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import threading
from datetime import datetime

# Import the main processing function
from main import main as process_single_video
from main import parse_args as parse_single_args


@dataclass
class BatchItem:
    """Represents a single video to be processed in batch mode."""
    identifier: str  # URL, file path, or custom identifier
    input_type: str  # 'url', 'file', or 'custom'
    output_subdir: str  # Subdirectory within batch output
    status: str = 'pending'  # pending, processing, completed, failed, skipped
    error_message: str = ''
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    output_files: Dict[str, str] = None  # Maps file type to path
    
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
    progress_interval: float = 5.0  # seconds
    timeout_per_video: Optional[int] = None  # seconds, None for no timeout


class BatchProcessor:
    """Handles batch processing of multiple videos."""
    
    def __init__(self, config: BatchConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe collections
        self.items: List[BatchItem] = []
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.lock = threading.Lock()
        
        # Progress tracking
        self.start_time = None
        self.last_progress_time = 0
        
        # Logging
        self.log_file = self.output_dir / "batch_log.txt"
        self.results_file = self.output_dir / "batch_results.json"
        
    def add_item(self, identifier: str, input_type: str, output_subdir: str = None):
        """Add a video item to the batch processing queue."""
        if output_subdir is None:
            # Generate subdir from identifier
            if input_type == 'url':
                # Extract video ID or use hash
                import hashlib
                safe_id = hashlib.md5(identifier.encode()).hexdigest()[:8]
                output_subdir = f"video_{safe_id}"
            else:
                # Use filename without extension
                output_subdir = Path(identifier).stem.replace(' ', '_').replace('-', '_')
        
        item = BatchItem(
            identifier=identifier,
            input_type=input_type,
            output_subdir=output_subdir
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
                
                # Determine if it's a URL or file path
                if line.startswith(('http://', 'https://')):
                    input_type = 'url'
                elif Path(line).exists():
                    input_type = 'file'
                else:
                    self.log(f"Warning: Line {line_num} is neither a valid URL nor existing file: {line}")
                    continue
                
                self.add_item(line, input_type)
    
    def add_from_directory(self, dir_path: Path, extensions: List[str] = None):
        """Add all video files from a directory."""
        if extensions is None:
            extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v']
        
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        for ext in extensions:
            for video_file in dir_path.glob(f"*{ext}"):
                self.add_item(str(video_file), 'file')
    
    def add_from_urls(self, urls: List[str]):
        """Add videos from a list of URLs."""
        for url in urls:
            self.add_item(url, 'url')
    
    def log(self, message: str, level: str = 'INFO'):
        """Thread-safe logging."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        with self.lock:
            print(log_entry, flush=True)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
    
    def process_single_item(self, item: BatchItem, base_args: List[str]) -> bool:
        """Process a single video item."""
        try:
            with self.lock:
                item.status = 'processing'
                item.start_time = time.time()
            
            self.log(f"Processing: {item.identifier}")
            
            # Create output directory for this item
            item_output_dir = self.output_dir / item.output_subdir
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
            item_args = base_args.copy()
            
            if item.input_type == 'url':
                item_args.extend(['--url', item.identifier])
            else:
                item_args.extend(['--video', item.identifier])
            
            item_args.extend(['--out', str(item_output_dir)])
            
            # Add resume flag to avoid cleaning existing work
            if '--resume' not in item_args:
                item_args.append('--resume')
            
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
                    if (item_output_dir / "report.pdf").exists():
                        item.output_files['pdf'] = str(item_output_dir / "report.pdf")
                    if (item_output_dir / "transcript" / "transcript.txt").exists():
                        item.output_files['transcript'] = str(item_output_dir / "transcript" / "transcript.txt")
                    if (item_output_dir / "audio.wav").exists():
                        item.output_files['audio'] = str(item_output_dir / "audio.wav")
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
    
    def save_results(self):
        """Save batch processing results to JSON file."""
        results = {
            'config': asdict(self.config),
            'summary': {
                'total_items': len(self.items),
                'completed': self.completed_count,
                'failed': self.failed_count,
                'skipped': self.skipped_count,
                'start_time': self.start_time,
                'end_time': time.time(),
                'total_duration': time.time() - self.start_time if self.start_time else 0
            },
            'items': [asdict(item) for item in self.items]
        }
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def print_progress(self):
        """Print current progress status."""
        with self.lock:
            total = len(self.items)
            completed = self.completed_count
            failed = self.failed_count
            skipped = self.skipped_count
            processing = sum(1 for item in self.items if item.status == 'processing')
            
            elapsed = time.time() - self.start_time if self.start_time else 0
            
            print(f"\n{'='*60}")
            print(f"Batch Processing Progress")
            print(f"{'='*60}")
            print(f"Total items: {total}")
            print(f"Completed: {completed}")
            print(f"Failed: {failed}")
            print(f"Skipped: {skipped}")
            print(f"Processing: {processing}")
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
            
            print(f"{'='*60}\n")
    
    def process_all(self, base_args: List[str]):
        """Process all items in the batch."""
        if not self.items:
            self.log("No items to process", 'WARNING')
            return
        
        self.start_time = time.time()
        self.log(f"Starting batch processing of {len(self.items)} items")
        self.log(f"Configuration: parallel={self.config.max_parallel}, retry_failed={self.config.retry_failed}")
        
        # Process items
        if self.config.max_parallel == 1:
            # Sequential processing
            for i, item in enumerate(self.items):
                if self.config.stop_on_error and self.failed_count > 0:
                    self.log("Stopping due to previous error", 'WARNING')
                    break
                
                self.log(f"Processing item {i+1}/{len(self.items)}: {item.identifier}")
                self.process_single_item(item, base_args)
                
                # Progress update
                if time.time() - self.last_progress_time >= self.config.progress_interval:
                    self.print_progress()
                    self.last_progress_time = time.time()
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(self.process_single_item, item, base_args): item
                    for item in self.items
                }
                
                # Process completed tasks
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    
                    try:
                        success = future.result(timeout=self.config.timeout_per_video)
                    except Exception as e:
                        self.log(f"Unexpected error processing {item.identifier}: {str(e)}", 'ERROR')
                        with self.lock:
                            item.status = 'failed'
                            item.error_message = str(e)
                            self.failed_count += 1
                    
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
        
        # Final results
        self.print_progress()
        self.save_results()
        
        # Retry failed items if configured
        if self.config.retry_failed and self.failed_count > 0:
            self.log(f"Retrying {self.failed_count} failed items...")
            failed_items = [item for item in self.items if item.status == 'failed']
            
            for retry_attempt in range(self.config.max_retries):
                if not failed_items:
                    break
                
                self.log(f"Retry attempt {retry_attempt + 1}/{self.config.max_retries}")
                retry_items = failed_items.copy()
                failed_items = []
                
                for item in retry_items:
                    # Reset item status
                    item.status = 'pending'
                    item.error_message = ''
                    
                    success = self.process_single_item(item, base_args)
                    if not success:
                        failed_items.append(item)
                
                if not failed_items:
                    self.log("All retries successful!")
                    break
        
        # Final summary
        total_time = time.time() - self.start_time
        self.log(f"Batch processing completed in {total_time:.1f}s")
        self.log(f"Results: {self.completed_count} completed, {self.failed_count} failed, {self.skipped_count} skipped")
        
        if self.failed_count > 0:
            self.log("Failed items:", 'WARNING')
            for item in self.items:
                if item.status == 'failed':
                    self.log(f"  - {item.identifier}: {item.error_message}", 'WARNING')


def parse_batch_args():
    """Parse command line arguments for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple videos into PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process videos from a text file
  python batch_process.py --input-file videos.txt --output-dir ./batch_output
  
  # Process all videos in a directory
  python batch_process.py --input-dir ./videos --output-dir ./batch_output --parallel 2
  
  # Process specific URLs
  python batch_process.py --urls "url1,url2,url3" --output-dir ./batch_output
  
  # Process with custom settings
  python batch_process.py --input-file videos.txt --output-dir ./batch_output \\
    --parallel 3 --transcribe-only --whisper-model small --beam-size 1
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-file', type=str, help='Text file containing video URLs/paths (one per line)')
    input_group.add_argument('--input-dir', type=str, help='Directory containing video files')
    input_group.add_argument('--urls', type=str, help='Comma-separated list of video URLs')
    
    # Output options
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for batch results')
    
    # Batch processing options
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes (default: 1)')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed items')
    parser.add_argument('--max-retries', type=int, default=2, help='Maximum retry attempts for failed items')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip items that already have output (default: True)')
    parser.add_argument('--stop-on-error', action='store_true', help='Stop processing on first error')
    parser.add_argument('--timeout', type=int, help='Timeout per video in seconds (default: no timeout)')
    parser.add_argument('--progress-interval', type=float, default=5.0, help='Progress update interval in seconds')
    
    # Pass through options for main processing
    parser.add_argument('--transcribe-only', action='store_true', help='Skip frames/classification; transcript-only PDF')
    parser.add_argument('--language', type=str, default='auto', help='Language code or auto')
    parser.add_argument('--whisper-model', type=str, default='medium', help='faster-whisper model size')
    parser.add_argument('--beam-size', type=int, default=5, help='Beam size for decoding')
    parser.add_argument('--export-srt', action='store_true', help='Also export transcript in SubRip (.srt)')
    parser.add_argument('--report-style', type=str, choices=['minimal', 'book'], default='book', help='PDF layout style')
    
    # Video processing options
    parser.add_argument('--kf-method', type=str, choices=['scene', 'iframe', 'interval'], default='scene', help='Keyframe extraction method')
    parser.add_argument('--max-fps', type=float, default=1.0, help='Max FPS for keyframe detection')
    parser.add_argument('--frame-max-frames', type=int, default=0, help='Cap total saved frames; 0 to disable')
    
    # Download options
    parser.add_argument('--cookies-from-browser', type=str, help='Browser to read cookies from')
    parser.add_argument('--browser-profile', type=str, help='Specific browser profile name')
    parser.add_argument('--cookies-file', type=str, help='Path to cookies.txt file')
    parser.add_argument('--use-android-client', action='store_true', help='Use YouTube Android client fallback')
    
    return parser.parse_args()


def main():
    """Main entry point for batch processing."""
    args = parse_batch_args()
    
    # Create batch configuration
    config = BatchConfig(
        max_parallel=args.parallel,
        retry_failed=args.retry_failed,
        max_retries=args.max_retries,
        skip_existing=args.skip_existing,
        stop_on_error=args.stop_on_error,
        timeout_per_video=args.timeout,
        progress_interval=args.progress_interval
    )
    
    # Create batch processor
    processor = BatchProcessor(config, Path(args.output_dir))
    
    # Add items based on input type
    try:
        if args.input_file:
            processor.add_from_file(Path(args.input_file))
        elif args.input_dir:
            processor.add_from_directory(Path(args.input_dir))
        elif args.urls:
            urls = [url.strip() for url in args.urls.split(',') if url.strip()]
            processor.add_from_urls(urls)
    except Exception as e:
        print(f"Error loading input: {e}")
        sys.exit(1)
    
    if not processor.items:
        print("No items to process")
        sys.exit(1)
    
    # Prepare base arguments for video processing
    base_args = []
    
    # Add processing options
    if args.transcribe_only:
        base_args.append('--transcribe-only')
    if args.language != 'auto':
        base_args.extend(['--language', args.language])
    if args.whisper_model != 'medium':
        base_args.extend(['--whisper-model', args.whisper_model])
    if args.beam_size != 5:
        base_args.extend(['--beam-size', str(args.beam_size)])
    if args.export_srt:
        base_args.append('--export-srt')
    if args.report_style != 'book':
        base_args.extend(['--report-style', args.report_style])
    
    # Add video processing options
    if args.kf_method != 'scene':
        base_args.extend(['--kf-method', args.kf_method])
    if args.max_fps != 1.0:
        base_args.extend(['--max-fps', str(args.max_fps)])
    if args.frame_max_frames > 0:
        base_args.extend(['--frame-max-frames', str(args.frame_max_frames)])
    
    # Add download options
    if args.cookies_from_browser:
        base_args.extend(['--cookies-from-browser', args.cookies_from_browser])
    if args.browser_profile:
        base_args.extend(['--browser-profile', args.browser_profile])
    if args.cookies_file:
        base_args.extend(['--cookies-file', args.cookies_file])
    if args.use_android_client:
        base_args.append('--use-android-client')
    
    # Process all items
    try:
        processor.process_all(base_args)
        
        # Exit with appropriate code
        if processor.failed_count > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user")
        processor.save_results()
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error during batch processing: {e}")
        processor.save_results()
        sys.exit(1)


if __name__ == '__main__':
    main()
