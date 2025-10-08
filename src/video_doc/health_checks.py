"""
Comprehensive health check system for all video processing services.
"""

import os
import time
import psutil
import subprocess
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .monitoring import get_logger, HealthStatus, health_monitor, metrics
from .database import get_db_session, check_database_health
from .error_handling import ProcessingError, ErrorSeverity

logger = get_logger(__name__)


@dataclass
class ServiceHealth:
    """Detailed health status for a service."""
    name: str
    status: str  # 'healthy', 'degraded', 'unhealthy', 'unknown'
    timestamp: datetime
    response_time_ms: Optional[float] = None
    version: Optional[str] = None
    uptime_seconds: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: str
    timestamp: datetime
    services: Dict[str, ServiceHealth]
    system_resources: Dict[str, Any]
    active_jobs: int
    queue_size: int
    uptime_seconds: float
    version: str


class HealthChecker:
    """Comprehensive health checker for all system components."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.start_time = time.time()
        self._register_health_checks()
    
    def _register_health_checks(self):
        """Register all health check functions."""
        health_monitor.register_health_check('database', self._check_database)
        health_monitor.register_health_check('redis', self._check_redis)
        health_monitor.register_health_check('celery', self._check_celery)
        health_monitor.register_health_check('ffmpeg', self._check_ffmpeg)
        health_monitor.register_health_check('whisper', self._check_whisper)
        health_monitor.register_health_check('disk_space', self._check_disk_space)
        health_monitor.register_health_check('memory', self._check_memory)
        health_monitor.register_health_check('cpu', self._check_cpu)
        health_monitor.register_health_check('network', self._check_network)
        health_monitor.register_health_check('file_system', self._check_file_system)
    
    def _check_database(self) -> HealthStatus:
        """Check database connectivity and performance."""
        start_time = time.time()
        try:
            # Test database connection
            db_health = check_database_health()
            response_time = (time.time() - start_time) * 1000
            
            if db_health:
                return HealthStatus(
                    service='database',
                    status='healthy',
                    timestamp=datetime.now(timezone.utc),
                    details={
                        'response_time_ms': response_time,
                        'connection': 'active',
                        'type': 'postgresql'
                    }
                )
            else:
                return HealthStatus(
                    service='database',
                    status='unhealthy',
                    timestamp=datetime.now(timezone.utc),
                    error_message='Database connection failed'
                )
        except Exception as e:
            return HealthStatus(
                service='database',
                status='unhealthy',
                timestamp=datetime.now(timezone.utc),
                error_message=f'Database check failed: {str(e)}'
            )
    
    def _check_redis(self) -> HealthStatus:
        """Check Redis connectivity."""
        start_time = time.time()
        try:
            import redis
            redis_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
            r = redis.from_url(redis_url)
            
            # Test Redis connection
            r.ping()
            response_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = r.info()
            
            return HealthStatus(
                service='redis',
                status='healthy',
                timestamp=datetime.now(timezone.utc),
                details={
                    'response_time_ms': response_time,
                    'version': info.get('redis_version'),
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients')
                }
            )
        except Exception as e:
            return HealthStatus(
                service='redis',
                status='unhealthy',
                timestamp=datetime.now(timezone.utc),
                error_message=f'Redis check failed: {str(e)}'
            )
    
    def _check_celery(self) -> HealthStatus:
        """Check Celery worker status."""
        try:
            from .celery_app import celery_app
            
            # Get active workers
            inspect = celery_app.control.inspect()
            active_workers = inspect.active()
            stats = inspect.stats()
            
            if active_workers:
                worker_count = len(active_workers)
                return HealthStatus(
                    service='celery',
                    status='healthy',
                    timestamp=datetime.now(timezone.utc),
                    details={
                        'active_workers': worker_count,
                        'workers': list(active_workers.keys()),
                        'stats': stats
                    }
                )
            else:
                return HealthStatus(
                    service='celery',
                    status='degraded',
                    timestamp=datetime.now(timezone.utc),
                    error_message='No active Celery workers found'
                )
        except Exception as e:
            return HealthStatus(
                service='celery',
                status='unhealthy',
                timestamp=datetime.now(timezone.utc),
                error_message=f'Celery check failed: {str(e)}'
            )
    
    def _check_ffmpeg(self) -> HealthStatus:
        """Check FFmpeg availability and version."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                return HealthStatus(
                    service='ffmpeg',
                    status='healthy',
                    timestamp=datetime.now(timezone.utc),
                    details={
                        'version': version_line,
                        'available': True
                    }
                )
            else:
                return HealthStatus(
                    service='ffmpeg',
                    status='unhealthy',
                    timestamp=datetime.now(timezone.utc),
                    error_message='FFmpeg not available'
                )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return HealthStatus(
                service='ffmpeg',
                status='unhealthy',
                timestamp=datetime.now(timezone.utc),
                error_message=f'FFmpeg check failed: {str(e)}'
            )
    
    def _check_whisper(self) -> HealthStatus:
        """Check Whisper model availability."""
        try:
            from faster_whisper import WhisperModel
            
            # Try to load a small model
            model = WhisperModel('tiny', device='cpu')
            return HealthStatus(
                service='whisper',
                status='healthy',
                timestamp=datetime.now(timezone.utc),
                details={
                    'model_loaded': True,
                    'device': 'cpu',
                    'available': True
                }
            )
        except Exception as e:
            return HealthStatus(
                service='whisper',
                status='unhealthy',
                timestamp=datetime.now(timezone.utc),
                error_message=f'Whisper check failed: {str(e)}'
            )
    
    def _check_disk_space(self) -> HealthStatus:
        """Check available disk space."""
        try:
            # Check multiple important paths
            paths_to_check = [
                ('/', 'root'),
                (os.getcwd(), 'current_dir'),
                ('/tmp', 'temp'),
                ('uploads', 'uploads'),
                ('outputs', 'outputs')
            ]
            
            disk_info = {}
            warnings = []
            
            for path, name in paths_to_check:
                try:
                    if not os.path.exists(path):
                        continue
                        
                    usage = psutil.disk_usage(path)
                    free_gb = usage.free / (1024**3)
                    total_gb = usage.total / (1024**3)
                    used_percent = (usage.used / usage.total) * 100
                    
                    disk_info[name] = {
                        'free_gb': round(free_gb, 2),
                        'total_gb': round(total_gb, 2),
                        'used_percent': round(used_percent, 2)
                    }
                    
                    # Add warnings for low disk space
                    if used_percent > 90:
                        warnings.append(f'{name}: {used_percent:.1f}% used (critical)')
                    elif used_percent > 80:
                        warnings.append(f'{name}: {used_percent:.1f}% used (warning)')
                        
                except Exception as e:
                    warnings.append(f'{name}: Error checking disk space - {str(e)}')
            
            status = 'healthy'
            if any('critical' in w for w in warnings):
                status = 'unhealthy'
            elif warnings:
                status = 'degraded'
            
            return HealthStatus(
                service='disk_space',
                status=status,
                timestamp=datetime.now(timezone.utc),
                details=disk_info,
                warnings=warnings
            )
        except Exception as e:
            return HealthStatus(
                service='disk_space',
                status='unhealthy',
                timestamp=datetime.now(timezone.utc),
                error_message=f'Disk space check failed: {str(e)}'
            )
    
    def _check_memory(self) -> HealthStatus:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_percent = memory.percent
            swap_percent = swap.percent
            
            status = 'healthy'
            warnings = []
            
            if memory_percent > 95:
                status = 'unhealthy'
                warnings.append(f'Memory usage critical: {memory_percent:.1f}%')
            elif memory_percent > 85:
                status = 'degraded'
                warnings.append(f'Memory usage high: {memory_percent:.1f}%')
            
            if swap_percent > 80:
                warnings.append(f'Swap usage high: {swap_percent:.1f}%')
            
            return HealthStatus(
                service='memory',
                status=status,
                timestamp=datetime.now(timezone.utc),
                details={
                    'memory_percent': memory_percent,
                    'memory_available_gb': round(memory.available / (1024**3), 2),
                    'memory_total_gb': round(memory.total / (1024**3), 2),
                    'swap_percent': swap_percent,
                    'swap_total_gb': round(swap.total / (1024**3), 2)
                },
                warnings=warnings
            )
        except Exception as e:
            return HealthStatus(
                service='memory',
                status='unhealthy',
                timestamp=datetime.now(timezone.utc),
                error_message=f'Memory check failed: {str(e)}'
            )
    
    def _check_cpu(self) -> HealthStatus:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else None
            
            status = 'healthy'
            warnings = []
            
            if cpu_percent > 95:
                status = 'unhealthy'
                warnings.append(f'CPU usage critical: {cpu_percent:.1f}%')
            elif cpu_percent > 85:
                status = 'degraded'
                warnings.append(f'CPU usage high: {cpu_percent:.1f}%')
            
            return HealthStatus(
                service='cpu',
                status=status,
                timestamp=datetime.now(timezone.utc),
                details={
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'load_avg': load_avg
                },
                warnings=warnings
            )
        except Exception as e:
            return HealthStatus(
                service='cpu',
                status='unhealthy',
                timestamp=datetime.now(timezone.utc),
                error_message=f'CPU check failed: {str(e)}'
            )
    
    def _check_network(self) -> HealthStatus:
        """Check network connectivity."""
        try:
            import socket
            
            # Test DNS resolution
            socket.gethostbyname('google.com')
            
            # Test HTTP connectivity (if requests available)
            try:
                import requests
                response = requests.get('https://httpbin.org/get', timeout=5)
                http_ok = response.status_code == 200
            except ImportError:
                http_ok = True  # Skip if requests not available
            
            return HealthStatus(
                service='network',
                status='healthy',
                timestamp=datetime.now(timezone.utc),
                details={
                    'dns_resolution': True,
                    'http_connectivity': http_ok
                }
            )
        except Exception as e:
            return HealthStatus(
                service='network',
                status='unhealthy',
                timestamp=datetime.now(timezone.utc),
                error_message=f'Network check failed: {str(e)}'
            )
    
    def _check_file_system(self) -> HealthStatus:
        """Check file system permissions and accessibility."""
        try:
            test_paths = [
                'uploads',
                'outputs',
                'logs',
                '/tmp'
            ]
            
            fs_info = {}
            warnings = []
            
            for path in test_paths:
                try:
                    path_obj = Path(path)
                    
                    # Check if path exists or can be created
                    if not path_obj.exists():
                        path_obj.mkdir(parents=True, exist_ok=True)
                    
                    # Test write permissions
                    test_file = path_obj / '.health_check_test'
                    test_file.write_text('test')
                    test_file.unlink()
                    
                    fs_info[path] = {
                        'exists': True,
                        'writable': True,
                        'readable': True
                    }
                    
                except Exception as e:
                    fs_info[path] = {
                        'exists': False,
                        'writable': False,
                        'readable': False,
                        'error': str(e)
                    }
                    warnings.append(f'{path}: {str(e)}')
            
            status = 'healthy'
            if warnings:
                status = 'degraded'
            
            return HealthStatus(
                service='file_system',
                status=status,
                timestamp=datetime.now(timezone.utc),
                details=fs_info,
                warnings=warnings
            )
        except Exception as e:
            return HealthStatus(
                service='file_system',
                status='unhealthy',
                timestamp=datetime.now(timezone.utc),
                error_message=f'File system check failed: {str(e)}'
            )
    
    def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status."""
        try:
            # Get all service health checks
            service_results = health_monitor.check_health()
            
            # Convert to ServiceHealth objects
            services = {}
            for name, health_status in service_results.items():
                services[name] = ServiceHealth(
                    name=name,
                    status=health_status.status,
                    timestamp=health_status.timestamp,
                    details=health_status.details,
                    error_message=health_status.error_message,
                    warnings=getattr(health_status, 'warnings', [])
                )
            
            # Get system resources
            system_resources = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': {
                    name: details for name, details in 
                    services.get('disk_space', ServiceHealth('disk_space', 'unknown', datetime.now(timezone.utc))).details.items()
                    if isinstance(details, dict)
                }
            }
            
            # Get job statistics
            active_jobs = 0
            queue_size = 0
            try:
                # This would integrate with your job management system
                # For now, return mock values
                active_jobs = len([j for j in services.values() if j.status == 'healthy'])
                queue_size = 0
            except Exception:
                pass
            
            # Determine overall status
            overall_status = health_monitor.get_overall_health()
            
            return SystemHealth(
                overall_status=overall_status,
                timestamp=datetime.now(timezone.utc),
                services=services,
                system_resources=system_resources,
                active_jobs=active_jobs,
                queue_size=queue_size,
                uptime_seconds=time.time() - self.start_time,
                version=self._get_version()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {str(e)}")
            # Return minimal health status
            return SystemHealth(
                overall_status='unknown',
                timestamp=datetime.now(timezone.utc),
                services={},
                system_resources={},
                active_jobs=0,
                queue_size=0,
                uptime_seconds=time.time() - self.start_time,
                version=self._get_version()
            )
    
    def _get_version(self) -> str:
        """Get application version."""
        try:
            # Try to read from version file or git
            version_file = Path('VERSION')
            if version_file.exists():
                return version_file.read_text().strip()
            
            # Try git
            result = subprocess.run(
                ['git', 'describe', '--tags', '--always'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
            
            return 'unknown'
        except Exception:
            return 'unknown'


# Global health checker instance
health_checker = HealthChecker()


def get_health_status() -> Dict[str, Any]:
    """Get health status for API endpoints."""
    system_health = health_checker.get_system_health()
    
    # Convert to JSON-serializable format
    return {
        'status': system_health.overall_status,
        'timestamp': system_health.timestamp.isoformat(),
        'uptime_seconds': system_health.uptime_seconds,
        'version': system_health.version,
        'services': {
            name: {
                'status': service.status,
                'timestamp': service.timestamp.isoformat(),
                'details': service.details,
                'error_message': service.error_message,
                'warnings': service.warnings
            }
            for name, service in system_health.services.items()
        },
        'system_resources': system_health.system_resources,
        'active_jobs': system_health.active_jobs,
        'queue_size': system_health.queue_size
    }


def get_health_summary() -> Dict[str, Any]:
    """Get a simplified health summary."""
    system_health = health_checker.get_system_health()
    
    return {
        'status': system_health.overall_status,
        'timestamp': system_health.timestamp.isoformat(),
        'uptime_seconds': system_health.uptime_seconds,
        'version': system_health.version,
        'service_count': len(system_health.services),
        'healthy_services': len([s for s in system_health.services.values() if s.status == 'healthy']),
        'degraded_services': len([s for s in system_health.services.values() if s.status == 'degraded']),
        'unhealthy_services': len([s for s in system_health.services.values() if s.status == 'unhealthy']),
        'active_jobs': system_health.active_jobs,
        'queue_size': system_health.queue_size
    }


def get_service_health(service_name: str) -> Optional[Dict[str, Any]]:
    """Get health status for a specific service."""
    system_health = health_checker.get_system_health()
    
    if service_name not in system_health.services:
        return None
    
    service = system_health.services[service_name]
    return {
        'name': service.name,
        'status': service.status,
        'timestamp': service.timestamp.isoformat(),
        'details': service.details,
        'error_message': service.error_message,
        'warnings': service.warnings
    }
