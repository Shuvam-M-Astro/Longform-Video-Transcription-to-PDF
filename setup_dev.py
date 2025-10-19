#!/usr/bin/env python3
"""
Development setup script for Video Processing System.
This script helps set up the development environment.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def setup_virtual_environment():
    """Set up virtual environment."""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command("python -m venv .venv", "Creating virtual environment")


def install_dependencies():
    """Install development dependencies."""
    # Determine the correct pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = ".venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = ".venv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install development requirements
    if not run_command(f"{pip_cmd} install -r requirements-dev.txt", "Installing development dependencies"):
        return False
    
    return True


def setup_pre_commit():
    """Set up pre-commit hooks."""
    if os.name == 'nt':  # Windows
        pre_commit_cmd = ".venv\\Scripts\\pre-commit"
    else:  # Unix/Linux/macOS
        pre_commit_cmd = ".venv/bin/pre-commit"
    
    return run_command(f"{pre_commit_cmd} install", "Installing pre-commit hooks")


def create_env_file():
    """Create .env file from template."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    env_template = """# Environment Configuration
# Copy this file and modify as needed

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/video_doc
REDIS_URL=redis://localhost:6379/0

# Application Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Processing Configuration
MAX_CONCURRENT_JOBS=3
JOB_TIMEOUT=3600
MAX_FILE_SIZE=1073741824

# Authentication Configuration
SESSION_DURATION_HOURS=1
ACCOUNT_LOCKOUT_MINUTES=15
MAX_FAILED_LOGIN_ATTEMPTS=5

# Monitoring Configuration
ENABLE_METRICS=True
ENABLE_HEALTH_CHECKS=True
LOG_LEVEL=INFO

# External Services
WHISPER_MODEL_PATH=./models
FFMPEG_PATH=ffmpeg
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_template)
        print("✅ Created .env file from template")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def create_gitignore():
    """Create .gitignore file."""
    gitignore_file = Path(".gitignore")
    if gitignore_file.exists():
        print("✅ .gitignore file already exists")
        return True
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
.env
*.log
outputs/
temp/
cache/
models/
*.db
*.sqlite

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/

# Documentation
docs/_build/
site/

# Jupyter
.ipynb_checkpoints/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json
"""
    
    try:
        with open(gitignore_file, 'w') as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore file")
        return True
    except Exception as e:
        print(f"❌ Failed to create .gitignore file: {e}")
        return False


def run_tests():
    """Run the test suite."""
    if os.name == 'nt':  # Windows
        pytest_cmd = ".venv\\Scripts\\pytest"
    else:  # Unix/Linux/macOS
        pytest_cmd = ".venv/bin/pytest"
    
    return run_command(f"{pytest_cmd} --version", "Checking pytest installation")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Set up development environment")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-pre-commit", action="store_true", help="Skip pre-commit setup")
    args = parser.parse_args()
    
    print("🚀 Setting up Video Processing System development environment...")
    print("=" * 60)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Setting up virtual environment", setup_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Creating .env file", create_env_file),
        ("Creating .gitignore", create_gitignore),
    ]
    
    if not args.skip_pre_commit:
        steps.append(("Setting up pre-commit hooks", setup_pre_commit))
    
    if not args.skip_tests:
        steps.append(("Running tests", run_tests))
    
    success_count = 0
    for description, func in steps:
        if func():
            success_count += 1
        print()
    
    print("=" * 60)
    if success_count == len(steps):
        print("🎉 Development environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate virtual environment:")
        if os.name == 'nt':
            print("   .venv\\Scripts\\activate")
        else:
            print("   source .venv/bin/activate")
        print("2. Configure .env file with your settings")
        print("3. Run the application:")
        print("   python web_app.py")
        print("4. Visit http://localhost:5000")
    else:
        print(f"⚠️  Setup completed with {len(steps) - success_count} errors")
        print("Please check the error messages above and fix any issues")


if __name__ == "__main__":
    main()
