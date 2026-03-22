"""
Tests for database models and functionality.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.video_doc.database import Base, ProcessingJob, ProcessingStatus, get_db_session


class TestDatabase:
    """Test database models and operations."""

    @pytest.fixture
    def db_session(self):
        """Create a test database session."""
        # Use in-memory SQLite for testing
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def test_processing_job_creation(self, db_session):
        """Test creating a processing job."""
        job = ProcessingJob(
            job_type="url",
            identifier="https://example.com/video.mp4",
            status=ProcessingStatus.PENDING,
            config={"language": "auto", "whisper_model": "tiny"}
        )
        db_session.add(job)
        db_session.commit()

        # Verify job was created
        assert job.id is not None
        assert job.job_type == "url"
        assert job.status == ProcessingStatus.PENDING
        assert job.config["language"] == "auto"

    def test_processing_job_status_update(self, db_session):
        """Test updating job status."""
        job = ProcessingJob(
            job_type="file",
            identifier="/path/to/video.mp4",
            status=ProcessingStatus.PENDING
        )
        db_session.add(job)
        db_session.commit()

        # Update status
        job.status = ProcessingStatus.PROCESSING
        db_session.commit()

        # Verify status was updated
        updated_job = db_session.query(ProcessingJob).filter(ProcessingJob.id == job.id).first()
        assert updated_job.status == ProcessingStatus.PROCESSING