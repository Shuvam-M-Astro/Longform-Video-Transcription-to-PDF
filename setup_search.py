#!/usr/bin/env python3
"""
Setup script for cross-language search feature.

This script:
1. Creates the database tables for search functionality
2. Verifies dependencies are installed
3. Downloads required models (if needed)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    missing = []
    
    required = [
        ('sentence_transformers', 'sentence-transformers'),
        ('langdetect', 'langdetect'),
    ]
    
    optional = [
        ('transformers', 'transformers'),
        ('googletrans', 'googletrans'),
    ]
    
    for module, package in required:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING - REQUIRED)")
            missing.append(package)
    
    for module, package in optional:
        try:
            __import__(module)
            print(f"  ✓ {package} (optional)")
        except ImportError:
            print(f"  ⚠ {package} (optional - not installed)")
    
    return missing


def create_tables():
    """Create database tables for search."""
    print("\nCreating database tables...")
    try:
        from src.video_doc.database import create_tables, engine, Base
        from src.video_doc.search import TranscriptChunk, SearchIndex
        from src.video_doc.auth import User, UserSession, APIKey
        
        # Import all models to ensure they're registered
        print("  Importing models...")
        
        # Create tables
        print("  Creating tables in database...")
        Base.metadata.create_all(bind=engine)
        print("  ✓ Tables created successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error creating tables: {e}")
        return False


def verify_models():
    """Verify that embedding and translation models can be loaded."""
    print("\nVerifying models...")
    
    # Test embedding service
    try:
        from src.video_doc.search import get_embedding_service
        print("  Testing embedding service...")
        service = get_embedding_service()
        if service.model:
            print(f"  ✓ Embedding model loaded: {service.embedding_dim} dimensions")
        else:
            print("  ⚠ Embedding model not loaded (will download on first use)")
    except Exception as e:
        print(f"  ⚠ Embedding service error: {e}")
    
    # Test translation service
    try:
        from src.video_doc.search import get_translation_service
        print("  Testing translation service...")
        service = get_translation_service()
        if service.model or service.translator:
            print("  ✓ Translation service available")
        else:
            print("  ⚠ Translation service not available (install transformers or googletrans)")
    except Exception as e:
        print(f"  ⚠ Translation service error: {e}")


def main():
    """Main setup function."""
    print("=" * 60)
    print("Cross-Language Search Setup")
    print("=" * 60)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"\n⚠ Missing required dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Create tables
    if not create_tables():
        print("\n✗ Setup failed. Please check database connection and try again.")
        return
    
    # Verify models (non-blocking)
    verify_models()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Process a video to create transcripts")
    print("2. Transcripts will be automatically indexed for search")
    print("3. Use the search panel in the web interface to search across transcripts")
    print("\nFor more information, see CROSS_LANGUAGE_SEARCH_README.md")


if __name__ == '__main__':
    main()

