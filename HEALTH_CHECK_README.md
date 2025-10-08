# Health Check System

This document describes the comprehensive health check system implemented for the Video Processing application.

## Overview

The health check system provides real-time monitoring of all system components, including:
- Database connectivity and performance
- Redis cache status
- Celery worker availability
- FFmpeg availability
- Whisper model loading
- System resources (CPU, memory, disk)
- Network connectivity
- File system permissions

## Health Check Endpoints

### Web Interface Endpoints

#### 1. Comprehensive Health Check
- **URL**: `/health`
- **Method**: GET
- **Description**: Complete health status of all services
- **Response**: Detailed JSON with all service statuses

#### 2. Health Summary
- **URL**: `/health/summary`
- **Method**: GET
- **Description**: Simplified health overview
- **Response**: Summary JSON with key metrics

#### 3. Service-Specific Health
- **URL**: `/health/service/<service_name>`
- **Method**: GET
- **Description**: Health status for a specific service
- **Response**: Service-specific health data

#### 4. Kubernetes Probes
- **URL**: `/health/live`
- **Method**: GET
- **Description**: Liveness probe for Kubernetes
- **Response**: Simple alive/dead status

- **URL**: `/health/ready`
- **Method**: GET
- **Description**: Readiness probe for Kubernetes
- **Response**: Ready/not-ready status

#### 5. Metrics Endpoint
- **URL**: `/metrics`
- **Method**: GET
- **Description**: Prometheus metrics
- **Response**: Prometheus-formatted metrics

#### 6. Health Dashboard
- **URL**: `/health-dashboard`
- **Method**: GET
- **Description**: Visual health monitoring dashboard
- **Response**: HTML dashboard with real-time updates

## Command Line Health Check

### Basic Usage

```bash
# Quick health check
python health_check.py

# Detailed health check
python health_check.py --detailed

# Check specific service
python health_check.py --service database

# Watch mode (continuous monitoring)
python health_check.py --watch

# JSON output
python health_check.py --json
```

### Watch Mode

Watch mode provides continuous monitoring with automatic refresh:

```bash
# Default 30-second refresh
python health_check.py --watch

# Custom refresh interval
python health_check.py --watch --interval 10
```

## Health Status Levels

### Status Types
- **healthy**: Service is working normally
- **degraded**: Service is working but with warnings
- **unhealthy**: Service has critical issues
- **unknown**: Service status cannot be determined

### HTTP Status Codes
- **200**: Healthy or degraded (operational)
- **503**: Unhealthy (service unavailable)
- **404**: Service not found (for service-specific checks)

## Monitored Services

### Core Services
1. **Database**: PostgreSQL connectivity and performance
2. **Redis**: Cache and message broker status
3. **Celery**: Worker availability and statistics
4. **FFmpeg**: Video processing tool availability
5. **Whisper**: AI model loading capability

### System Resources
1. **CPU**: Usage percentage and load average
2. **Memory**: RAM usage and swap status
3. **Disk Space**: Available space on key directories
4. **Network**: DNS resolution and connectivity
5. **File System**: Permissions and accessibility

## Health Dashboard Features

The web-based health dashboard provides:

- **Real-time Updates**: Auto-refresh every 30 seconds
- **Visual Status Indicators**: Color-coded service status
- **System Metrics**: CPU, memory, and disk usage
- **Warning Display**: Clear visibility of issues
- **Service Details**: Expandable service information
- **Uptime Tracking**: Application uptime display

## Integration Examples

### Kubernetes Health Checks

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: video-processor
    livenessProbe:
      httpGet:
        path: /health/live
        port: 5000
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 5000
      initialDelaySeconds: 5
      periodSeconds: 5
```

### Monitoring with curl

```bash
# Quick health check
curl http://localhost:5000/health/summary

# Detailed health check
curl http://localhost:5000/health

# Check specific service
curl http://localhost:5000/health/service/database

# Get metrics
curl http://localhost:5000/metrics
```

### Prometheus Monitoring

The `/metrics` endpoint provides Prometheus-compatible metrics:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'video-processor'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

## Troubleshooting

### Common Issues

1. **Database Unhealthy**
   - Check PostgreSQL connection
   - Verify database credentials
   - Check network connectivity

2. **Redis Unhealthy**
   - Verify Redis server is running
   - Check connection URL
   - Verify Redis configuration

3. **FFmpeg Unhealthy**
   - Ensure FFmpeg is installed
   - Check PATH environment variable
   - Verify FFmpeg permissions

4. **High Resource Usage**
   - Monitor CPU and memory usage
   - Check for memory leaks
   - Optimize processing parameters

### Debug Mode

Enable debug logging for detailed health check information:

```bash
export LOG_LEVEL=DEBUG
python web_app.py
```

## Custom Health Checks

You can add custom health checks by extending the `HealthChecker` class:

```python
from src.video_doc.health_checks import health_checker

def custom_service_check():
    # Your custom health check logic
    return HealthStatus(
        service='custom_service',
        status='healthy',
        timestamp=datetime.now(timezone.utc),
        details={'custom_metric': 'value'}
    )

# Register the custom check
health_checker.register_health_check('custom_service', custom_service_check)
```

## Best Practices

1. **Regular Monitoring**: Set up automated health checks
2. **Alerting**: Configure alerts for unhealthy services
3. **Logging**: Monitor health check logs for patterns
4. **Capacity Planning**: Use metrics for resource planning
5. **Testing**: Include health checks in your testing strategy

## API Response Examples

### Health Summary Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "service_count": 10,
  "healthy_services": 9,
  "degraded_services": 1,
  "unhealthy_services": 0,
  "active_jobs": 2,
  "queue_size": 0
}
```

### Service Health Response
```json
{
  "name": "database",
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "response_time_ms": 15.2,
    "connection": "active",
    "type": "postgresql"
  },
  "warnings": []
}
```
