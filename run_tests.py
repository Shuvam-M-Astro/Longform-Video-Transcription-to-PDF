#!/usr/bin/env python3
"""
Test runner script for the video processing system.
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run the test suite."""
    print("Running test suite...")

    # Check if we're in the right directory
    if not Path("src").exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)

    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v",
        "tests/"
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)