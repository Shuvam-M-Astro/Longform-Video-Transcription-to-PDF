# Development Setup Guide

This guide helps you set up the Video Processing System for development.

## Quick Start

### 1. Automated Setup
Run the automated setup script:
```bash
python setup_dev.py
```

This will:
- Check Python version compatibility
- Create virtual environment
- Install all development dependencies
- Set up pre-commit hooks
- Create configuration files
- Run initial tests

### 2. Manual Setup

#### Prerequisites
- Python 3.8 or higher
- PostgreSQL 12+
- Redis 6+
- FFmpeg
- Git

#### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Longform-Video-Transcription-to-PDF
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Unix/Linux/macOS
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```

5. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

6. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

7. **Set up database**
   ```bash
   # Create PostgreSQL database
   createdb video_doc_dev
   
   # Run migrations (if available)
   python -m alembic upgrade head
   ```

8. **Run tests**
   ```bash
   pytest
   ```

9. **Start development server**
   ```bash
   python web_app.py
   ```

## Development Tools

### Code Quality
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pylint**: Advanced linting
- **bandit**: Security analysis

### Testing
- **pytest**: Test framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **pytest-xdist**: Parallel testing
- **factory-boy**: Test data factories

### Documentation
- **Sphinx**: API documentation
- **MkDocs**: User documentation
- **mkdocs-material**: Material theme

### Development Utilities
- **ipython**: Enhanced REPL
- **jupyter**: Notebook environment
- **rich**: Rich text formatting
- **tqdm**: Progress bars

## Development Workflow

### 1. Code Quality Checks
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linting
flake8 src/ tests/

# Type checking
mypy src/

# Security check
bandit -r src/
```

### 2. Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_auth.py

# Run tests in parallel
pytest -n auto
```

### 3. Documentation
```bash
# Build Sphinx docs
cd docs/
make html

# Serve MkDocs locally
mkdocs serve
```

### 4. Database Management
```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Configuration

### Environment Variables
Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/video_doc_dev
REDIS_URL=redis://localhost:6379/1

# Application
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key

# Processing
MAX_CONCURRENT_JOBS=2
JOB_TIMEOUT=1800
MAX_FILE_SIZE=536870912

# Authentication
SESSION_DURATION_HOURS=8
ACCOUNT_LOCKOUT_MINUTES=5
MAX_FAILED_LOGIN_ATTEMPTS=10

# Monitoring
ENABLE_METRICS=True
ENABLE_HEALTH_CHECKS=True
LOG_LEVEL=DEBUG
```

### Development Configuration
The `dev_config.ini` file contains development-specific settings that override default values.

## Testing

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── fixtures/      # Test fixtures
└── conftest.py    # Pytest configuration
```

### Running Tests
```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With coverage report
pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src --cov-report=html
```

### Test Data
Use `factory-boy` for generating test data:
```python
from factory import Factory, Faker
from src.video_doc.auth import User

class UserFactory(Factory):
    class Meta:
        model = User
    
    username = Faker('user_name')
    email = Faker('email')
    role = 'user'
```

## Debugging

### Debug Tools
- **ipdb**: Enhanced debugger
- **pdbpp**: Advanced debugger
- **rich**: Rich debugging output

### Profiling
```bash
# Memory profiling
python -m memory_profiler script.py

# Line profiling
kernprof -l -v script.py

# CPU profiling
python -m cProfile script.py
```

### Logging
Development logging is configured in `dev_config.ini`:
```ini
LOG_LEVEL = DEBUG
ENABLE_SQL_LOGGING = True
```

## API Development

### Interactive Documentation
- **API Docs**: http://localhost:5000/api/docs
- **OpenAPI Spec**: http://localhost:5000/api/openapi.json

### Testing API Endpoints
```bash
# Using curl
curl -X POST http://localhost:5000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Using httpx (Python)
python -c "
import httpx
response = httpx.post('http://localhost:5000/auth/login', 
                     json={'username': 'admin', 'password': 'admin123'})
print(response.json())
"
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   ```bash
   # Check PostgreSQL status
   pg_ctl status
   
   # Start PostgreSQL
   pg_ctl start
   ```

2. **Redis Connection Error**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Start Redis
   redis-server
   ```

3. **FFmpeg Not Found**
   ```bash
   # Install FFmpeg
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

4. **Permission Errors**
   ```bash
   # Fix file permissions
   chmod +x setup_dev.py
   chmod +x scripts/*.py
   ```

### Debug Mode
Enable debug mode for detailed error messages:
```bash
export FLASK_DEBUG=True
export LOG_LEVEL=DEBUG
python web_app.py
```

## Contributing

### Pre-commit Hooks
Pre-commit hooks are automatically installed and will run on every commit:
- Code formatting (black)
- Import sorting (isort)
- Linting (flake8)
- Type checking (mypy)
- Security checks (bandit)

### Commit Messages
Follow conventional commit format:
```
feat: add new authentication system
fix: resolve database connection issue
docs: update API documentation
test: add unit tests for auth module
```

### Pull Request Process
1. Create feature branch
2. Make changes
3. Run tests and quality checks
4. Update documentation
5. Submit pull request

## Resources

### Documentation
- [Flask Documentation](https://flask.palletsprojects.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)

### Tools
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)

### Development Resources
- [Python Best Practices](https://docs.python-guide.org/)
- [Flask Best Practices](https://exploreflask.com/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
