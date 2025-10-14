"""
Database configuration and models for video processing metadata storage.
"""

import os
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    Boolean, Float, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

# Database configuration
DATABASE_URL = os.getenv(
    'DATABASE_URL', 
    'postgresql://video_doc:password@localhost:5432/video_doc_db'
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=os.getenv('SQL_ECHO', 'False').lower() == 'true'
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ProcessingJob(Base):
    """Main processing job table."""
    __tablename__ = "processing_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(20), nullable=False)  # 'url' or 'file'
    identifier = Column(Text, nullable=False)  # URL or file path
    status = Column(String(20), nullable=False, default=ProcessingStatus.PENDING)
    
    # Processing configuration
    config = Column(JSON, nullable=False, default=dict)
    
    # Progress tracking
    progress = Column(Float, default=0.0)
    current_step = Column(String(100))
    error_message = Column(Text)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Output tracking
    output_files = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    processing_steps = relationship("ProcessingStep", back_populates="job", cascade="all, delete-orphan")
    quality_checks = relationship("QualityCheck", back_populates="job", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_job_status', 'status'),
        Index('idx_job_created', 'created_at'),
        Index('idx_job_type', 'job_type'),
    )


class ProcessingStep(Base):
    """Individual processing steps within a job."""
    __tablename__ = "processing_steps"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey('processing_jobs.id'), nullable=False)
    
    step_name = Column(String(100), nullable=False)
    step_order = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default=ProcessingStatus.PENDING)
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    
    # Progress and metrics
    progress = Column(Float, default=0.0)
    metrics = Column(JSON, default=dict)
    error_message = Column(Text)
    
    # Relationships
    job = relationship("ProcessingJob", back_populates="processing_steps")
    
    # Indexes
    __table_args__ = (
        Index('idx_step_job', 'job_id'),
        Index('idx_step_order', 'job_id', 'step_order'),
    )


class QualityCheck(Base):
    """Data quality checks and validation results."""
    __tablename__ = "quality_checks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey('processing_jobs.id'), nullable=False)
    
    check_name = Column(String(100), nullable=False)
    check_type = Column(String(50), nullable=False)  # 'validation', 'quality', 'integrity'
    status = Column(String(20), nullable=False)  # 'passed', 'failed', 'warning'
    
    # Check details
    expected_value = Column(Text)
    actual_value = Column(Text)
    threshold = Column(Float)
    message = Column(Text)
    
    # Timing
    executed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("ProcessingJob", back_populates="quality_checks")
    
    # Indexes
    __table_args__ = (
        Index('idx_quality_job', 'job_id'),
        Index('idx_quality_status', 'status'),
        Index('idx_quality_type', 'check_type'),
    )


class SystemMetrics(Base):
    """System performance and resource metrics."""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # 'counter', 'gauge', 'histogram'
    
    # Metric values
    value = Column(Float, nullable=False)
    labels = Column(JSON, default=dict)
    
    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_metrics_name', 'metric_name'),
        Index('idx_metrics_timestamp', 'timestamp'),
        Index('idx_metrics_type', 'metric_type'),
    )


class AuditLog(Base):
    """Audit trail for all operations."""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Operation details
    operation = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100))
    
    # User and context
    user_id = Column(String(100))
    session_id = Column(String(100))
    ip_address = Column(String(45))
    
    # Operation data
    old_values = Column(JSON)
    new_values = Column(JSON)
    metadata = Column(JSON, default=dict)
    
    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_operation', 'operation'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_user', 'user_id'),
    )


# Database utility functions
def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables."""
    Base.metadata.drop_all(bind=engine)


def get_db_session() -> Session:
    """Get a database session for direct use."""
    return SessionLocal()


# Database health check
def check_database_health() -> Dict[str, Any]:
    """Check database connectivity and health."""
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT 1").fetchone()
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Job management utilities
class JobManager:
    """High-level job management utilities."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_job(
        self, 
        job_type: str, 
        identifier: str, 
        config: Dict[str, Any]
    ) -> ProcessingJob:
        """Create a new processing job."""
        job = ProcessingJob(
            job_type=job_type,
            identifier=identifier,
            config=config,
            status=ProcessingStatus.PENDING
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job
    
    def update_job_status(
        self, 
        job_id: str, 
        status: ProcessingStatus,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update job status and progress."""
        try:
            job = self.db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if not job:
                return False
            
            job.status = status
            job.last_activity = datetime.utcnow()
            
            if progress is not None:
                job.progress = progress
            if current_step is not None:
                job.current_step = current_step
            if error_message is not None:
                job.error_message = error_message
            
            if status == ProcessingStatus.PROCESSING and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]:
                job.completed_at = datetime.utcnow()
            
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            return False
    
    def add_processing_step(
        self,
        job_id: str,
        step_name: str,
        step_order: int,
        status: ProcessingStatus = ProcessingStatus.PENDING
    ) -> ProcessingStep:
        """Add a processing step to a job."""
        step = ProcessingStep(
            job_id=job_id,
            step_name=step_name,
            step_order=step_order,
            status=status
        )
        self.db.add(step)
        self.db.commit()
        self.db.refresh(step)
        return step
    
    def update_step_status(
        self,
        step_id: str,
        status: ProcessingStatus,
        progress: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update processing step status."""
        try:
            step = self.db.query(ProcessingStep).filter(ProcessingStep.id == step_id).first()
            if not step:
                return False
            
            step.status = status
            
            if progress is not None:
                step.progress = progress
            if metrics is not None:
                step.metrics = metrics
            if error_message is not None:
                step.error_message = error_message
            
            if status == ProcessingStatus.PROCESSING and not step.started_at:
                step.started_at = datetime.utcnow()
            elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                step.completed_at = datetime.utcnow()
                if step.started_at:
                    step.duration_seconds = (step.completed_at - step.started_at).total_seconds()
            
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            return False
    
    def add_quality_check(
        self,
        job_id: str,
        check_name: str,
        check_type: str,
        status: str,
        expected_value: Optional[str] = None,
        actual_value: Optional[str] = None,
        threshold: Optional[float] = None,
        message: Optional[str] = None
    ) -> QualityCheck:
        """Add a quality check result."""
        check = QualityCheck(
            job_id=job_id,
            check_name=check_name,
            check_type=check_type,
            status=status,
            expected_value=expected_value,
            actual_value=actual_value,
            threshold=threshold,
            message=message
        )
        self.db.add(check)
        self.db.commit()
        self.db.refresh(check)
        return check
    
    def get_job_stats(self) -> Dict[str, Any]:
        """Get job statistics."""
        total_jobs = self.db.query(ProcessingJob).count()
        completed_jobs = self.db.query(ProcessingJob).filter(
            ProcessingJob.status == ProcessingStatus.COMPLETED
        ).count()
        failed_jobs = self.db.query(ProcessingJob).filter(
            ProcessingJob.status == ProcessingStatus.FAILED
        ).count()
        processing_jobs = self.db.query(ProcessingJob).filter(
            ProcessingJob.status == ProcessingStatus.PROCESSING
        ).count()
        
        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "processing_jobs": processing_jobs,
            "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0
        }


# Import authentication models
from .auth import User, UserSession, APIKey