#!/usr/bin/env python3
"""
Database initialization script for video processing pipeline.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.video_doc.database import create_tables, check_database_health, engine
from src.video_doc.monitoring import get_logger

logger = get_logger(__name__)


def init_database():
    """Initialize the database with all tables."""
    try:
        # Check database health first
        health = check_database_health()
        if health["status"] != "healthy":
            logger.error("Database is not healthy", health=health)
            return False
        
        # Create all tables
        logger.info("Creating database tables...")
        create_tables()
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        return False


def create_sample_data():
    """Create sample data for testing."""
    try:
        from src.video_doc.database import get_db_session, JobManager, ProcessingStatus
        
        db = get_db_session()
        job_manager = JobManager(db)
        
        # Create a sample job
        sample_config = {
            "language": "auto",
            "whisper_model": "medium",
            "beam_size": 5,
            "transcribe_only": False,
            "streaming": False,
            "kf_method": "scene",
            "max_fps": 1.0,
            "min_scene_diff": 0.45,
            "report_style": "book"
        }
        
        job = job_manager.create_job("url", "https://example.com/video", sample_config)
        logger.info("Created sample job", job_id=str(job.id))
        
        # Add some sample processing steps
        step1 = job_manager.add_processing_step(str(job.id), "download", 1, ProcessingStatus.COMPLETED)
        step2 = job_manager.add_processing_step(str(job.id), "transcription", 2, ProcessingStatus.PROCESSING)
        
        logger.info("Created sample processing steps")
        
        # Add sample quality checks
        job_manager.add_quality_check(
            str(job.id), "config_validation", "validation", "passed",
            message="Configuration is valid"
        )
        
        logger.info("Created sample quality checks")
        
        db.close()
        return True
        
    except Exception as e:
        logger.error("Failed to create sample data", error=str(e))
        return False


def main():
    """Main initialization function."""
    print("Initializing video processing database...")
    
    # Initialize database
    if not init_database():
        print("❌ Database initialization failed")
        sys.exit(1)
    
    print("✅ Database initialization completed")
    
    # Create sample data if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--with-sample-data":
        print("Creating sample data...")
        if create_sample_data():
            print("✅ Sample data created")
        else:
            print("❌ Sample data creation failed")
            sys.exit(1)
    
    print("🎉 Database setup complete!")


if __name__ == "__main__":
    main()
