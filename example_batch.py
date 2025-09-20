#!/usr/bin/env python3
"""
Example Batch Processing Script

This script demonstrates how to use the batch processing functionality
programmatically with Python code.
"""

from pathlib import Path
from batch_process_advanced import AdvancedBatchProcessor, BatchConfig


def example_basic_batch():
    """Example of basic batch processing."""
    print("=== Basic Batch Processing Example ===")
    
    # Create configuration
    config = BatchConfig(
        max_parallel=2,
        retry_failed=True,
        max_retries=2,
        skip_existing=True,
        transcribe_only=False,
        whisper_model='small',  # Faster processing for demo
        beam_size=1
    )
    
    # Create processor
    processor = AdvancedBatchProcessor(config, Path('./example_output'))
    
    # Add some example videos (replace with real paths/URLs)
    example_videos = [
        # Add your video paths or URLs here
        # "https://www.youtube.com/watch?v=example1",
        # "./videos/lecture1.mp4",
        # "./videos/tutorial2.mkv"
    ]
    
    if not example_videos:
        print("No example videos provided. Add some video paths or URLs to the example_videos list.")
        return
    
    # Add videos to processor
    for video in example_videos:
        if video.startswith(('http://', 'https://')):
            processor.add_item(video, 'url')
        else:
            processor.add_item(video, 'file')
    
    # Process all videos
    processor.process_all()
    
    print(f"Batch processing completed!")
    print(f"Results: {processor.completed_count} completed, {processor.failed_count} failed")


def example_priority_batch():
    """Example of batch processing with priorities."""
    print("\n=== Priority Batch Processing Example ===")
    
    config = BatchConfig(
        max_parallel=1,  # Sequential to show priority order
        retry_failed=True,
        transcribe_only=True  # Faster for demo
    )
    
    processor = AdvancedBatchProcessor(config, Path('./example_priority_output'))
    
    # Add videos with different priorities
    videos_with_priority = [
        ("https://www.youtube.com/watch?v=high_priority", 'url', 10),
        ("./videos/medium_priority.mp4", 'file', 5),
        ("https://www.youtube.com/watch?v=low_priority", 'url', 1),
    ]
    
    for video, input_type, priority in videos_with_priority:
        processor.add_item(video, input_type, priority=priority)
    
    print("Videos will be processed in priority order (highest first)")
    processor.process_all()


def example_resume_batch():
    """Example of resumable batch processing."""
    print("\n=== Resumable Batch Processing Example ===")
    
    config = BatchConfig(
        max_parallel=1,
        skip_existing=True,  # This enables resume functionality
        transcribe_only=True
    )
    
    processor = AdvancedBatchProcessor(config, Path('./example_resume_output'))
    
    # Try to load previous state
    if processor.load_state():
        print("Resumed from previous batch processing")
    else:
        print("Starting new batch processing")
        # Add videos here
        # processor.add_item("video1.mp4", 'file')
        # processor.add_item("video2.mp4", 'file')
    
    if processor.items:
        processor.process_all()
    else:
        print("No videos to process")


def example_custom_config():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create a custom configuration
    config = BatchConfig(
        max_parallel=3,
        retry_failed=True,
        max_retries=3,
        skip_existing=True,
        stop_on_error=False,
        timeout_per_video=1800,  # 30 minutes per video
        progress_interval=10.0,   # Update every 10 seconds
        
        # Processing options
        transcribe_only=False,
        language='en',
        whisper_model='medium',
        beam_size=5,
        export_srt=True,
        report_style='book',
        
        # Keyframe options
        kf_method='scene',
        max_fps=0.5,  # Slower extraction for better quality
        max_frames=50,
        scene_threshold=0.3,
        
        # Download options
        use_android_client=True
    )
    
    processor = AdvancedBatchProcessor(config, Path('./example_custom_output'))
    
    # Add videos from a directory
    video_dir = Path('./videos')
    if video_dir.exists():
        processor.add_from_directory(video_dir, recursive=True)
        print(f"Found {len(processor.items)} videos in {video_dir}")
    else:
        print(f"Video directory {video_dir} not found")
    
    if processor.items:
        processor.process_all()
    else:
        print("No videos found to process")


def main():
    """Run all examples."""
    print("Batch Processing Examples")
    print("=" * 50)
    
    # Uncomment the examples you want to run
    
    # example_basic_batch()
    # example_priority_batch()
    # example_resume_batch()
    # example_custom_config()
    
    print("\nTo run examples:")
    print("1. Uncomment the example functions you want to run")
    print("2. Add real video paths or URLs to the example_videos lists")
    print("3. Run: python example_batch.py")


if __name__ == '__main__':
    main()
