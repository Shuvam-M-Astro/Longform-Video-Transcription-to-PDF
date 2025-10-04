"""
Comprehensive logging and monitoring system for video processing pipeline.
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

# Configure structured logging
def configure_logging(log_level: str = "INFO", log_format: str = "json"):
    """Configure structured logging with correlation IDs."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if log_format == "json" 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('video_processing.log')
        ]
    )


# Correlation ID management
class CorrelationContext:
    """Thread-local correlation ID context."""
    
    def __init__(self):
        self._local = threading.local()
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current thread."""
        self._local.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get correlation ID for current thread."""
        return getattr(self._local, 'correlation_id', None)
    
    def clear_correlation_id(self):
        """Clear correlation ID for current thread."""
        if hasattr(self._local, 'correlation_id'):
            delattr(self._local, 'correlation_id')


correlation_context = CorrelationContext()


def get_logger(name: str) -> structlog.BoundLogger:
    """Get structured logger with correlation ID."""
    logger = structlog.get_logger(name)
    correlation_id = correlation_context.get_correlation_id()
    if correlation_id:
        logger = logger.bind(correlation_id=correlation_id)
    return logger


@contextmanager
def correlation_id_context(correlation_id: str):
    """Context manager for correlation ID."""
    correlation_context.set_correlation_id(correlation_id)
    try:
        yield
    finally:
        correlation_context.clear_correlation_id()


# Metrics collection
class MetricsCollector:
    """Prometheus metrics collector for video processing."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize all metrics."""
        
        # Job metrics
        self.jobs_total = Counter(
            'video_processing_jobs_total',
            'Total number of processing jobs',
            ['job_type', 'status'],
            registry=self.registry
        )
        
        self.job_duration = Histogram(
            'video_processing_job_duration_seconds',
            'Duration of processing jobs',
            ['job_type'],
            buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600],
            registry=self.registry
        )
        
        # Step metrics
        self.steps_total = Counter(
            'video_processing_steps_total',
            'Total number of processing steps',
            ['step_name', 'status'],
            registry=self.registry
        )
        
        self.step_duration = Histogram(
            'video_processing_step_duration_seconds',
            'Duration of processing steps',
            ['step_name'],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 300],
            registry=self.registry
        )
        
        # Quality metrics
        self.quality_checks_total = Counter(
            'video_processing_quality_checks_total',
            'Total number of quality checks',
            ['check_type', 'status'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'video_processing_errors_total',
            'Total number of errors',
            ['error_type', 'severity'],
            registry=self.registry
        )
        
        # System metrics
        self.active_jobs = Gauge(
            'video_processing_active_jobs',
            'Number of currently active jobs',
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'video_processing_queue_size',
            'Number of jobs in queue',
            registry=self.registry
        )
        
        # Resource metrics
        self.cpu_usage = Gauge(
            'video_processing_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'video_processing_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'video_processing_disk_usage_bytes',
            'Disk usage in bytes',
            ['path'],
            registry=self.registry
        )
    
    def increment_job(self, job_type: str, status: str):
        """Increment job counter."""
        self.jobs_total.labels(job_type=job_type, status=status).inc()
    
    def observe_job_duration(self, job_type: str, duration: float):
        """Observe job duration."""
        self.job_duration.labels(job_type=job_type).observe(duration)
    
    def increment_step(self, step_name: str, status: str):
        """Increment step counter."""
        self.steps_total.labels(step_name=step_name, status=status).inc()
    
    def observe_step_duration(self, step_name: str, duration: float):
        """Observe step duration."""
        self.step_duration.labels(step_name=step_name).observe(duration)
    
    def increment_quality_check(self, check_type: str, status: str):
        """Increment quality check counter."""
        self.quality_checks_total.labels(check_type=check_type, status=status).inc()
    
    def increment_error(self, error_type: str, severity: str):
        """Increment error counter."""
        self.errors_total.labels(error_type=error_type, severity=severity).inc()
    
    def set_active_jobs(self, count: int):
        """Set active jobs gauge."""
        self.active_jobs.set(count)
    
    def set_queue_size(self, size: int):
        """Set queue size gauge."""
        self.queue_size.set(size)
    
    def set_cpu_usage(self, usage: float):
        """Set CPU usage gauge."""
        self.cpu_usage.set(usage)
    
    def set_memory_usage(self, usage: int):
        """Set memory usage gauge."""
        self.memory_usage.set(usage)
    
    def set_disk_usage(self, path: str, usage: int):
        """Set disk usage gauge."""
        self.disk_usage.labels(path=path).set(usage)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)


# Global metrics collector
metrics = MetricsCollector()


# Performance monitoring
@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Finish timing the operation."""
        self.end_time = datetime.now(timezone.utc)
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error_message


class PerformanceMonitor:
    """Performance monitoring context manager."""
    
    def __init__(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.metrics = PerformanceMetrics(
            operation=operation,
            start_time=datetime.now(timezone.utc),
            metadata=metadata
        )
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        self.logger.info(
            "Starting operation",
            operation=self.metrics.operation,
            metadata=self.metrics.metadata
        )
        return self.metrics
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None
        
        self.metrics.finish(success, error_message)
        
        # Log performance metrics
        self.logger.info(
            "Operation completed",
            operation=self.metrics.operation,
            duration=self.metrics.duration,
            success=success,
            error_message=error_message,
            metadata=self.metrics.metadata
        )
        
        # Update Prometheus metrics
        if self.metrics.operation.startswith('job_'):
            job_type = self.metrics.metadata.get('job_type', 'unknown')
            metrics.observe_job_duration(job_type, self.metrics.duration or 0)
        elif self.metrics.operation.startswith('step_'):
            step_name = self.metrics.metadata.get('step_name', 'unknown')
            metrics.observe_step_duration(step_name, self.metrics.duration or 0)


def monitor_performance(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for performance monitoring."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceMonitor(operation, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Audit logging
class AuditLogger:
    """Audit logger for compliance and security."""
    
    def __init__(self):
        self.logger = get_logger('audit')
    
    def log_operation(
        self,
        operation: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an audit event."""
        audit_event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation': operation,
            'resource_type': resource_type,
            'resource_id': resource_id,
            'user_id': user_id,
            'session_id': session_id,
            'ip_address': ip_address,
            'old_values': old_values,
            'new_values': new_values,
            'metadata': metadata or {}
        }
        
        self.logger.info("Audit event", **audit_event)


# Global audit logger
audit_logger = AuditLogger()


# Health check monitoring
@dataclass
class HealthStatus:
    """Health status data class."""
    service: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class HealthMonitor:
    """Health monitoring for system components."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.health_checks: Dict[str, callable] = {}
    
    def register_health_check(self, name: str, check_func: callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    def check_health(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                if isinstance(result, HealthStatus):
                    results[name] = result
                else:
                    # Convert simple boolean/string to HealthStatus
                    status = 'healthy' if result else 'unhealthy'
                    results[name] = HealthStatus(
                        service=name,
                        status=status,
                        timestamp=datetime.now(timezone.utc)
                    )
            except Exception as e:
                results[name] = HealthStatus(
                    service=name,
                    status='unhealthy',
                    timestamp=datetime.now(timezone.utc),
                    error_message=str(e)
                )
                self.logger.error(
                    "Health check failed",
                    service=name,
                    error=str(e)
                )
        
        return results
    
    def get_overall_health(self) -> str:
        """Get overall system health."""
        results = self.check_health()
        
        if not results:
            return 'unknown'
        
        statuses = [result.status for result in results.values()]
        
        if all(status == 'healthy' for status in statuses):
            return 'healthy'
        elif any(status == 'unhealthy' for status in statuses):
            return 'unhealthy'
        else:
            return 'degraded'


# Global health monitor
health_monitor = HealthMonitor()


# Log aggregation and analysis
class LogAnalyzer:
    """Log analysis and pattern detection."""
    
    def __init__(self, log_file: str = 'video_processing.log'):
        self.log_file = log_file
        self.logger = get_logger(__name__)
    
    def analyze_errors(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns in logs."""
        # This would typically integrate with ELK stack or similar
        # For now, return mock data
        return {
            'total_errors': 0,
            'error_types': {},
            'error_trends': [],
            'most_common_errors': []
        }
    
    def analyze_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance patterns in logs."""
        return {
            'average_duration': 0,
            'slowest_operations': [],
            'performance_trends': []
        }


# System resource monitoring
class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        try:
            import psutil
            return psutil.virtual_memory().used
        except ImportError:
            return 0
    
    def get_disk_usage(self, path: str) -> int:
        """Get disk usage in bytes."""
        try:
            import psutil
            return psutil.disk_usage(path).used
        except ImportError:
            return 0
    
    def update_metrics(self):
        """Update resource metrics."""
        try:
            cpu_usage = self.get_cpu_usage()
            memory_usage = self.get_memory_usage()
            
            metrics.set_cpu_usage(cpu_usage)
            metrics.set_memory_usage(memory_usage)
            
            # Monitor common paths
            for path in ['/tmp', '/var/log', os.getcwd()]:
                disk_usage = self.get_disk_usage(path)
                metrics.set_disk_usage(path, disk_usage)
                
        except Exception as e:
            self.logger.error("Failed to update resource metrics", error=str(e))


# Global resource monitor
resource_monitor = ResourceMonitor()


# Logging utilities
def log_job_start(job_id: str, job_type: str, config: Dict[str, Any]):
    """Log job start."""
    logger = get_logger(__name__)
    logger.info(
        "Job started",
        job_id=job_id,
        job_type=job_type,
        config=config
    )
    metrics.increment_job(job_type, 'started')


def log_job_completion(job_id: str, job_type: str, duration: float, success: bool):
    """Log job completion."""
    logger = get_logger(__name__)
    status = 'completed' if success else 'failed'
    
    logger.info(
        "Job completed",
        job_id=job_id,
        job_type=job_type,
        duration=duration,
        success=success
    )
    
    metrics.increment_job(job_type, status)
    if success:
        metrics.observe_job_duration(job_type, duration)


def log_step_start(step_name: str, job_id: str):
    """Log step start."""
    logger = get_logger(__name__)
    logger.info(
        "Step started",
        step_name=step_name,
        job_id=job_id
    )
    metrics.increment_step(step_name, 'started')


def log_step_completion(step_name: str, job_id: str, duration: float, success: bool):
    """Log step completion."""
    logger = get_logger(__name__)
    status = 'completed' if success else 'failed'
    
    logger.info(
        "Step completed",
        step_name=step_name,
        job_id=job_id,
        duration=duration,
        success=success
    )
    
    metrics.increment_step(step_name, status)
    if success:
        metrics.observe_step_duration(step_name, duration)


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log error with context."""
    logger = get_logger(__name__)
    logger.error(
        "Error occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {}
    )
    
    metrics.increment_error(type(error).__name__, 'medium')


def log_quality_check(check_name: str, check_type: str, status: str, details: Optional[Dict[str, Any]] = None):
    """Log quality check result."""
    logger = get_logger(__name__)
    logger.info(
        "Quality check completed",
        check_name=check_name,
        check_type=check_type,
        status=status,
        details=details or {}
    )
    
    metrics.increment_quality_check(check_type, status)


# Initialize logging configuration
configure_logging(
    log_level=os.getenv('LOG_LEVEL', 'INFO'),
    log_format=os.getenv('LOG_FORMAT', 'json')
)
