#!/usr/bin/env python3
"""
Batch Processing Utilities

Helper functions and utilities for batch video processing.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def create_video_list_from_directory(
    directory: Path, 
    output_file: Path, 
    extensions: List[str] = None,
    recursive: bool = True,
    include_priority: bool = False
):
    """Create a video list file from a directory of videos."""
    if extensions is None:
        extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v', '.flv', '.wmv']
    
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    pattern = "**/*" if recursive else "*"
    video_files = []
    
    for ext in extensions:
        for video_file in directory.glob(f"{pattern}{ext}"):
            video_files.append(video_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Video list generated from directory\n")
        f.write(f"# Directory: {directory}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total videos: {len(video_files)}\n\n")
        
        for video_file in sorted(video_files):
            if include_priority:
                f.write(f"0:{video_file}\n")  # Default priority 0
            else:
                f.write(f"{video_file}\n")
    
    print(f"Created video list with {len(video_files)} videos: {output_file}")


def create_video_list_from_urls(
    urls: List[str], 
    output_file: Path,
    include_priority: bool = False
):
    """Create a video list file from a list of URLs."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Video list generated from URLs\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total URLs: {len(urls)}\n\n")
        
        for url in urls:
            if include_priority:
                f.write(f"0:{url}\n")  # Default priority 0
            else:
                f.write(f"{url}\n")
    
    print(f"Created video list with {len(urls)} URLs: {output_file}")


def analyze_batch_results(results_file: Path) -> Dict[str, Any]:
    """Analyze batch processing results and return statistics."""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    summary = results.get('summary', {})
    items = results.get('items', [])
    
    # Calculate additional statistics
    total_duration = sum(item.get('duration_seconds', 0) for item in items if item.get('status') == 'completed')
    avg_duration = total_duration / summary.get('completed', 1) if summary.get('completed', 0) > 0 else 0
    
    # Group by status
    status_counts = {}
    for item in items:
        status = item.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Find longest and shortest processing times
    completed_items = [item for item in items if item.get('status') == 'completed' and item.get('duration_seconds')]
    if completed_items:
        longest = max(completed_items, key=lambda x: x.get('duration_seconds', 0))
        shortest = min(completed_items, key=lambda x: x.get('duration_seconds', 0))
    else:
        longest = shortest = None
    
    analysis = {
        'summary': summary,
        'status_counts': status_counts,
        'total_processing_time': total_duration,
        'average_processing_time': avg_duration,
        'longest_processing': longest,
        'shortest_processing': shortest,
        'success_rate': summary.get('completed', 0) / summary.get('total_items', 1) * 100,
        'failure_rate': summary.get('failed', 0) / summary.get('total_items', 1) * 100
    }
    
    return analysis


def print_batch_summary(results_file: Path):
    """Print a formatted summary of batch processing results."""
    analysis = analyze_batch_results(results_file)
    
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    
    summary = analysis['summary']
    print(f"Total items: {summary.get('total_items', 0)}")
    print(f"Completed: {summary.get('completed', 0)} ({analysis['success_rate']:.1f}%)")
    print(f"Failed: {summary.get('failed', 0)} ({analysis['failure_rate']:.1f}%)")
    print(f"Skipped: {summary.get('skipped', 0)}")
    print(f"Cancelled: {summary.get('cancelled', 0)}")
    
    if summary.get('total_duration'):
        print(f"Total processing time: {summary['total_duration']:.1f}s")
    
    if analysis['total_processing_time'] > 0:
        print(f"Video processing time: {analysis['total_processing_time']:.1f}s")
        print(f"Average per video: {analysis['average_processing_time']:.1f}s")
    
    if analysis['longest_processing']:
        longest = analysis['longest_processing']
        print(f"Longest processing: {longest['identifier']} ({longest['duration_seconds']:.1f}s)")
    
    if analysis['shortest_processing']:
        shortest = analysis['shortest_processing']
        print(f"Shortest processing: {shortest['identifier']} ({shortest['duration_seconds']:.1f}s)")
    
    print("\nStatus breakdown:")
    for status, count in analysis['status_counts'].items():
        percentage = count / summary.get('total_items', 1) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
    
    print("="*70)


def export_results_to_csv(results_file: Path, output_csv: Path):
    """Export batch processing results to CSV format."""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    items = results.get('items', [])
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        if not items:
            return
        
        # Get all possible fieldnames
        fieldnames = set()
        for item in items:
            fieldnames.update(item.keys())
        
        # Add output file fields
        for item in items:
            if 'output_files' in item and item['output_files']:
                for file_type in item['output_files'].keys():
                    fieldnames.add(f'output_{file_type}')
        
        fieldnames = sorted(fieldnames)
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in items:
            row = item.copy()
            
            # Flatten output_files
            if 'output_files' in row and row['output_files']:
                for file_type, file_path in row['output_files'].items():
                    row[f'output_{file_type}'] = file_path
                del row['output_files']
            
            writer.writerow(row)
    
    print(f"Exported results to CSV: {output_csv}")


def create_batch_template(output_file: Path):
    """Create a batch processing template file."""
    template = """# Batch Processing Template
# Copy this file and modify as needed

# Input videos (one per line)
# Format: [priority:]identifier
# Priority is optional (higher number = higher priority)
# Examples:
# 5:https://www.youtube.com/watch?v=example1
# 0:./videos/lecture1.mp4
# 10:https://vimeo.com/123456789

# Add your videos here:
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"Created batch template: {output_file}")


def main():
    """Command line interface for batch utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch processing utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create video list from directory
    dir_parser = subparsers.add_parser('create-list', help='Create video list from directory')
    dir_parser.add_argument('directory', type=str, help='Directory containing videos')
    dir_parser.add_argument('output', type=str, help='Output file path')
    dir_parser.add_argument('--recursive', action='store_true', help='Search recursively')
    dir_parser.add_argument('--priority', action='store_true', help='Include priority column')
    dir_parser.add_argument('--extensions', nargs='+', help='Video file extensions')
    
    # Create video list from URLs
    url_parser = subparsers.add_parser('create-urls', help='Create video list from URLs')
    url_parser.add_argument('urls', nargs='+', help='Video URLs')
    url_parser.add_argument('output', type=str, help='Output file path')
    url_parser.add_argument('--priority', action='store_true', help='Include priority column')
    
    # Analyze results
    analyze_parser = subparsers.add_parser('analyze', help='Analyze batch results')
    analyze_parser.add_argument('results', type=str, help='Results JSON file')
    analyze_parser.add_argument('--csv', type=str, help='Export to CSV file')
    
    # Create template
    template_parser = subparsers.add_parser('template', help='Create batch template')
    template_parser.add_argument('output', type=str, help='Output template file')
    
    args = parser.parse_args()
    
    if args.command == 'create-list':
        extensions = args.extensions or ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v', '.flv', '.wmv']
        create_video_list_from_directory(
            Path(args.directory), 
            Path(args.output), 
            extensions, 
            args.recursive, 
            args.priority
        )
    
    elif args.command == 'create-urls':
        create_video_list_from_urls(args.urls, Path(args.output), args.priority)
    
    elif args.command == 'analyze':
        print_batch_summary(Path(args.results))
        if args.csv:
            export_results_to_csv(Path(args.results), Path(args.csv))
    
    elif args.command == 'template':
        create_batch_template(Path(args.output))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
