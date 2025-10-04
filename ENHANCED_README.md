# Enhanced Video Documentation Builder

A production-ready video processing pipeline that converts long-form videos into comprehensive PDF documents with advanced data engineering features.

## 🚀 New Features

### Data Engineering Infrastructure
- **PostgreSQL Database**: Metadata storage, job tracking, and audit logging
- **Error Handling & Retry Logic**: Circuit breakers, exponential backoff, and graceful degradation
- **Comprehensive Monitoring**: Structured logging, Prometheus metrics, and health checks
- **Data Quality Validation**: Input validation, quality checks, and compliance tracking

### Enhanced Processing Pipeline
- **Database Integration**: All jobs and steps tracked in PostgreSQL
- **Quality Assurance**: Automated validation of audio, video, transcript, and output quality
- **Performance Monitoring**: Real-time metrics and performance tracking
- **Audit Trail**: Complete operation logging for compliance

## 📋 Prerequisites

### System Requirements
- Python 3.10+
- PostgreSQL 12+
- Redis (optional, for caching)
- FFmpeg installed and available in PATH
- NVIDIA GPU + CUDA (optional, for acceleration)

### Database Setup
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE video_doc_db;
CREATE USER video_doc WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE video_doc_db TO video_doc;
\q
```

## 🛠️ Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Longform-Video-Transcription-to-PDF
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Initialize Database
```bash
# Initialize database tables
python init_database.py

# Optional: Create sample data
python init_database.py --with-sample-data
```

### 4. Environment Configuration
Create a `.env` file:
```bash
# Database
DATABASE_URL=postgresql://video_doc:password@localhost:5432/video_doc_db

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Processing
MAX_CONCURRENT_JOBS=3
JOB_TIMEOUT=3600

# Web Interface
SECRET_KEY=your-secret-key-here
HOST=0.0.0.0
PORT=5000
```

## 🎯 Usage

### Enhanced Command Line Interface
```bash
# Process a video URL with enhanced pipeline
python enhanced_main.py --url "https://youtube.com/watch?v=VIDEO_ID" --out "./outputs/enhanced_run"

# Process with custom configuration
python enhanced_main.py \
  --url "https://youtube.com/watch?v=VIDEO_ID" \
  --out "./outputs/custom_run" \
  --whisper-model large \
  --beam-size 5 \
  --language en \
  --report-style book
```

### Enhanced Web Interface
```bash
# Start enhanced web server
python enhanced_web_app.py
```

Access the web interface at `http://localhost:5000`

### Health Monitoring
- **Health Check**: `http://localhost:5000/health`
- **Metrics**: `http://localhost:5000/metrics` (Prometheus format)
- **Job Quality**: `http://localhost:5000/job/<job_id>/quality`

## 📊 Monitoring & Observability

### Structured Logging
All operations are logged with structured JSON format including:
- Correlation IDs for request tracing
- Performance metrics
- Error context and stack traces
- Audit trail for compliance

### Prometheus Metrics
Available metrics include:
- `video_processing_jobs_total`: Total processing jobs by type and status
- `video_processing_job_duration_seconds`: Job processing duration
- `video_processing_steps_total`: Processing steps by name and status
- `video_processing_errors_total`: Error counts by type and severity
- `video_processing_active_jobs`: Currently active jobs
- `video_processing_queue_size`: Queue size

### Health Checks
The system provides comprehensive health monitoring:
- Database connectivity
- System resource usage
- Error rates and patterns
- Circuit breaker status

## 🔧 Configuration

### Database Configuration
```python
# src/video_doc/database.py
DATABASE_URL = 'postgresql://user:password@host:port/database'
```

### Error Handling Configuration
```python
# src/video_doc/error_handling.py
@retry_on_failure(
    max_attempts=3,
    base_delay=2.0,
    max_delay=60.0,
    retryable_exceptions=[ConnectionError, TimeoutError]
)
```

### Monitoring Configuration
```python
# src/video_doc/monitoring.py
configure_logging(
    log_level="INFO",
    log_format="json"  # or "console"
)
```

## 🏗️ Architecture

### Data Flow
```
Input (URL/File) → Validation → Database Job Creation → Processing Pipeline → Quality Checks → Output Generation → Audit Logging
```

### Components
- **EnhancedVideoProcessor**: Main processing orchestrator
- **Database Layer**: PostgreSQL for metadata and job tracking
- **Error Handling**: Circuit breakers and retry logic
- **Monitoring**: Structured logging and metrics collection
- **Data Validation**: Input/output quality assurance

### Database Schema
- `processing_jobs`: Main job tracking
- `processing_steps`: Individual step tracking
- `quality_checks`: Data quality validation results
- `system_metrics`: Performance metrics
- `audit_logs`: Compliance and audit trail

## 🔍 Quality Assurance

### Input Validation
- File format and size validation
- URL format and accessibility checks
- Configuration parameter validation
- Audio/video quality assessment

### Processing Validation
- Audio duration and quality checks
- Video resolution and frame rate validation
- Transcript completeness and accuracy
- Output file integrity verification

### Quality Metrics
- Processing success rates
- Error frequency and types
- Performance benchmarks
- Resource utilization

## 🚨 Error Handling

### Retry Strategies
- Exponential backoff with jitter
- Circuit breaker pattern for external services
- Graceful degradation for non-critical failures
- Dead letter queues for failed operations

### Error Categories
- **Retryable**: Network timeouts, temporary service unavailability
- **Non-retryable**: Invalid input, configuration errors
- **Critical**: Database failures, system resource exhaustion

### Recovery Mechanisms
- Automatic retry with increasing delays
- Fallback processing methods
- Resource cleanup on failures
- Detailed error reporting and logging

## 📈 Performance Optimization

### Parallel Processing
- Multi-threaded processing steps
- Concurrent job execution
- Resource pooling and reuse

### Caching
- Redis integration for session management
- Model caching for Whisper
- Result caching for repeated operations

### Resource Management
- Connection pooling for database
- Memory usage monitoring
- Disk space management
- CPU utilization tracking

## 🔒 Security & Compliance

### Data Protection
- Input sanitization and validation
- Secure file handling
- Access control and authentication
- Audit trail for all operations

### Compliance Features
- Complete operation logging
- Data lineage tracking
- Quality assurance documentation
- Error reporting and analysis

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests
pytest tests/unit/

# Run with coverage
pytest --cov=src tests/unit/
```

### Integration Tests
```bash
# Run integration tests
pytest tests/integration/

# Test with database
pytest tests/integration/ --database-url=postgresql://test:test@localhost:5432/test_db
```

### Load Testing
```bash
# Run load tests
pytest tests/load/
```

## 📚 API Documentation

### REST Endpoints
- `POST /upload`: File upload processing
- `POST /process_url`: URL processing
- `GET /jobs`: List all jobs
- `GET /job/<id>`: Get job details
- `GET /job/<id>/quality`: Get quality checks
- `GET /health`: Health check
- `GET /metrics`: Prometheus metrics

### WebSocket Events
- `job_status`: Job status updates
- `job_progress`: Progress updates
- `job_step`: Step updates
- `error`: Error notifications

## 🐛 Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database status
python -c "from src.video_doc.database import check_database_health; print(check_database_health())"

# Reset database
python init_database.py
```

#### Memory Issues
```bash
# Monitor memory usage
curl http://localhost:5000/metrics | grep memory

# Adjust concurrent jobs
export MAX_CONCURRENT_JOBS=1
```

#### Processing Failures
```bash
# Check error logs
tail -f video_processing.log | grep ERROR

# View job quality checks
curl http://localhost:5000/job/<job_id>/quality
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export SQL_ECHO=true

# Run with debug mode
python enhanced_web_app.py
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- FFmpeg for video processing
- PostgreSQL for data storage
- Prometheus for metrics collection
- Flask for web framework
