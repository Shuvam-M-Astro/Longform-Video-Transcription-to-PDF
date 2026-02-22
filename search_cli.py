#!/usr/bin/env python3
"""
Command-line interface for searching and managing transcript indexes.

This tool allows you to:
- Check indexing status for processed videos
- Manually trigger re-indexing
- Search across all indexed transcripts
- Export search results
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Optional, List
from datetime import datetime

try:
    from src.video_doc.search import get_search_service, SearchIndex
    from src.video_doc.database import get_db_session
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    print("Warning: Search functionality not available. Install required dependencies.", file=sys.stderr)


def get_job_id_from_output_dir(output_dir: Path) -> str:
    """Generate a stable job_id from output directory path."""
    output_path_str = str(output_dir.absolute())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, output_path_str))


def check_index_status(output_dir: Optional[Path] = None, job_id: Optional[str] = None) -> None:
    """Check the indexing status for a processed video."""
    if not SEARCH_AVAILABLE:
        print("Error: Search functionality not available.", file=sys.stderr)
        sys.exit(1)
    
    if not output_dir and not job_id:
        print("Error: Must provide either --output-dir or --job-id", file=sys.stderr)
        sys.exit(1)
    
    if output_dir:
        job_id = get_job_id_from_output_dir(Path(output_dir))
        print(f"Output directory: {output_dir}")
    
    print(f"Job ID: {job_id}")
    
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        print(f"Error: Invalid job ID format: {job_id}", file=sys.stderr)
        sys.exit(1)
    
    db = get_db_session()
    try:
        search_index = db.query(SearchIndex).filter(SearchIndex.job_id == job_uuid).first()
        
        if not search_index:
            print("\nStatus: Not indexed")
            print("This transcript has not been indexed yet.")
            print("\nTo index it, run:")
            if output_dir:
                print(f"  python search_cli.py index --output-dir \"{output_dir}\"")
            else:
                print(f"  python search_cli.py index --job-id \"{job_id}\"")
            return
        
        print(f"\nStatus: {search_index.status}")
        print(f"Total chunks: {search_index.total_chunks}")
        print(f"Indexed chunks: {search_index.indexed_chunks}")
        
        if search_index.indexed_at:
            print(f"Indexed at: {search_index.indexed_at.isoformat()}")
        else:
            print("Indexed at: Not yet indexed")
        
        if search_index.error_message:
            print(f"\nError: {search_index.error_message}")
        
        if search_index.status == 'completed':
            print("\n✓ Transcript is fully indexed and searchable")
        elif search_index.status == 'indexing':
            print("\n⏳ Transcript is currently being indexed...")
        elif search_index.status == 'failed':
            print("\n✗ Indexing failed. You can try re-indexing.")
    finally:
        db.close()


def index_transcript(
    output_dir: Optional[Path] = None, 
    job_id: Optional[str] = None,
    segments_json_path: Optional[Path] = None
) -> None:
    """Manually trigger indexing for a processed video."""
    if not SEARCH_AVAILABLE:
        print("Error: Search functionality not available.", file=sys.stderr)
        sys.exit(1)
    
    if segments_json_path:
        segments_json = Path(segments_json_path)
        if not segments_json.exists():
            print(f"Error: Transcript file not found at {segments_json}", file=sys.stderr)
            sys.exit(1)
        # Try to infer output_dir from segments_json path
        if segments_json.parent.name == "transcript":
            output_dir = segments_json.parent.parent
        # Generate job_id from output_dir if available, otherwise use segments_json path
        if output_dir:
            job_id = get_job_id_from_output_dir(output_dir)
        else:
            # Use segments_json path to generate job_id
            job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(segments_json.absolute())))
    elif output_dir:
        output_dir = Path(output_dir)
        job_id = get_job_id_from_output_dir(output_dir)
        segments_json = output_dir / "transcript" / "segments.json"
        
        if not segments_json.exists():
            print(f"Error: Transcript not found at {segments_json}", file=sys.stderr)
            print("Make sure the video has been processed first.", file=sys.stderr)
            sys.exit(1)
    elif job_id:
        # If only job_id provided, we need to find the segments.json
        # Try common output locations
        common_outputs = [
            Path("./outputs") / job_id / "transcript" / "segments.json",
            Path("./outputs/run") / "transcript" / "segments.json",
        ]
        
        # Also check environment variable
        import os
        if os.getenv('OUTPUT_FOLDER'):
            common_outputs.insert(0, Path(os.getenv('OUTPUT_FOLDER')) / job_id / "transcript" / "segments.json")
        
        segments_json = None
        for path in common_outputs:
            if path.exists():
                segments_json = path
                break
        
        if not segments_json:
            print("Error: Could not find segments.json file.", file=sys.stderr)
            print("Please provide --output-dir or --segments-json to specify the transcript location.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: Must provide --output-dir, --job-id, or --segments-json", file=sys.stderr)
        sys.exit(1)
    
    print(f"Output directory: {output_dir}")
    print(f"Job ID: {job_id}")
    print(f"Transcript file: {segments_json}")
    print("\nStarting indexing...")
    
    try:
        search_service = get_search_service()
        success = search_service.index_transcript(
            job_id=job_id,
            segments_json_path=segments_json
        )
        
        if success:
            print("✓ Indexing completed successfully!")
            print("\nYou can now search this transcript using:")
            print(f"  python search_cli.py search --query \"your search query\"")
        else:
            print("✗ Indexing failed. Check the error messages above.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error during indexing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def search_transcripts(
    query: str,
    target_language: Optional[str] = None,
    job_ids: Optional[List[str]] = None,
    limit: int = 10,
    min_score: float = 0.5,
    search_mode: str = 'semantic',
    output_file: Optional[Path] = None,
    format: str = 'text'
) -> None:
    """Search across all indexed transcripts."""
    if not SEARCH_AVAILABLE:
        print("Error: Search functionality not available.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Searching for: '{query}'")
    if target_language:
        print(f"Target language: {target_language}")
    print(f"Search mode: {search_mode}")
    print(f"Limit: {limit}")
    print(f"Min score: {min_score}")
    print()
    
    try:
        search_service = get_search_service()
        results = search_service.search(
            query=query,
            target_language=target_language,
            job_ids=job_ids,
            limit=limit,
            min_score=min_score,
            search_mode=search_mode
        )
        
        total = results.get('total', 0)
        search_results = results.get('results', [])
        
        print(f"Found {total} result(s)\n")
        
        if total == 0:
            print("No results found. Try:")
            print("  - Lowering --min-score (current: {})".format(min_score))
            print("  - Using a different search mode (--mode keyword or --mode hybrid)")
            print("  - Checking that transcripts are indexed (python search_cli.py status)")
            return
        
        # Format output
        if format == 'json':
            output_data = {
                'query': query,
                'total': total,
                'results': search_results
            }
            output_str = json.dumps(output_data, indent=2, ensure_ascii=False)
        else:  # text format
            output_lines = []
            for i, result in enumerate(search_results, 1):
                score = result.get('score', 0)
                text = result.get('text', '')
                start_time = result.get('start_time', 0)
                end_time = result.get('end_time', 0)
                job_id = result.get('job_id', '')
                language = result.get('original_language', 'unknown')
                
                # Format timestamp
                start_min = int(start_time // 60)
                start_sec = int(start_time % 60)
                end_min = int(end_time // 60)
                end_sec = int(end_time % 60)
                
                output_lines.append(f"\n[{i}] Score: {score:.3f} | Job: {job_id[:8]}... | Lang: {language}")
                output_lines.append(f"    Time: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}")
                output_lines.append(f"    Text: {text}")
            
            output_str = '\n'.join(output_lines)
        
        # Output to file or stdout
        if output_file:
            output_file.write_text(output_str, encoding='utf-8')
            print(f"\nResults saved to: {output_file}")
        else:
            print(output_str)
            
    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def list_indexed_jobs() -> None:
    """List all jobs that have been indexed."""
    if not SEARCH_AVAILABLE:
        print("Error: Search functionality not available.", file=sys.stderr)
        sys.exit(1)
    
    db = get_db_session()
    try:
        indexes = db.query(SearchIndex).order_by(SearchIndex.indexed_at.desc()).all()
        
        if not indexes:
            print("No indexed transcripts found.")
            return
        
        print(f"Found {len(indexes)} indexed job(s):\n")
        print(f"{'Job ID':<40} {'Status':<12} {'Chunks':<10} {'Indexed At':<20}")
        print("-" * 82)
        
        for idx in indexes:
            job_id_str = str(idx.job_id)
            status = idx.status
            chunks = f"{idx.indexed_chunks}/{idx.total_chunks}"
            indexed_at = idx.indexed_at.isoformat() if idx.indexed_at else "N/A"
            
            print(f"{job_id_str:<40} {status:<12} {chunks:<10} {indexed_at:<20}")
            
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Search and manage transcript indexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check indexing status
  python search_cli.py status --output-dir ./outputs/run1
  
  # Manually index a transcript
  python search_cli.py index --output-dir ./outputs/run1
  python search_cli.py index --segments-json ./outputs/run1/transcript/segments.json
  
  # Search transcripts
  python search_cli.py search --query "machine learning"
  python search_cli.py search --query "deep learning" --limit 20 --min-score 0.3
  
  # Search with translation
  python search_cli.py search --query "deep learning" --target-language es
  
  # Export results to JSON
  python search_cli.py search --query "neural networks" --output results.json --format json
  
  # List all indexed jobs
  python search_cli.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check indexing status')
    status_group = status_parser.add_mutually_exclusive_group(required=True)
    status_group.add_argument('--output-dir', type=str, help='Output directory from video processing')
    status_group.add_argument('--job-id', type=str, help='Job ID (UUID)')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Manually trigger indexing')
    index_parser.add_argument('--output-dir', type=str, help='Output directory from video processing')
    index_parser.add_argument('--job-id', type=str, help='Job ID (UUID)')
    index_parser.add_argument('--segments-json', type=str, help='Direct path to segments.json file')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search across indexed transcripts')
    search_parser.add_argument('--query', '-q', type=str, required=True, help='Search query')
    search_parser.add_argument('--target-language', '-t', type=str, help='Target language for results (e.g., es, fr)')
    search_parser.add_argument('--job-ids', type=str, help='Comma-separated job IDs to search within')
    search_parser.add_argument('--limit', '-n', type=int, default=10, help='Maximum number of results (default: 10)')
    search_parser.add_argument('--min-score', type=float, default=0.5, help='Minimum similarity score (0-1, default: 0.5)')
    search_parser.add_argument('--mode', type=str, choices=['semantic', 'keyword', 'hybrid'], default='semantic',
                              help='Search mode (default: semantic)')
    search_parser.add_argument('--output', '-o', type=str, help='Output file path (default: stdout)')
    search_parser.add_argument('--format', type=str, choices=['text', 'json'], default='text',
                              help='Output format (default: text)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all indexed jobs')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'status':
        check_index_status(
            output_dir=Path(args.output_dir) if args.output_dir else None,
            job_id=args.job_id
        )
    elif args.command == 'index':
        index_transcript(
            output_dir=Path(args.output_dir) if args.output_dir else None,
            job_id=args.job_id,
            segments_json_path=Path(args.segments_json) if args.segments_json else None
        )
    elif args.command == 'search':
        job_ids = None
        if args.job_ids:
            job_ids = [jid.strip() for jid in args.job_ids.split(',')]
        
        search_transcripts(
            query=args.query,
            target_language=args.target_language,
            job_ids=job_ids,
            limit=args.limit,
            min_score=args.min_score,
            search_mode=args.mode,
            output_file=Path(args.output) if args.output else None,
            format=args.format
        )
    elif args.command == 'list':
        list_indexed_jobs()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

