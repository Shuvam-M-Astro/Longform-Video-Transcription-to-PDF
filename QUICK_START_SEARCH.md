# Quick Start: Cross-Language Search

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `sentence-transformers` - For multilingual embeddings
- `transformers` - For translation (optional but recommended)
- `langdetect` - For language detection
- `googletrans` - Fallback translation service

## Step 2: Setup Database Tables

Run the setup script:

```bash
python setup_search.py
```

Or manually create tables using Alembic:

```bash
# Generate migration
alembic revision --autogenerate -m "Add cross-language search tables"

# Apply migration
alembic upgrade head
```

## Step 3: Verify Setup

Check that everything is working:

```python
# Test in Python shell
from src.video_doc.search import get_search_service, get_embedding_service

# Test embedding service
embedding_service = get_embedding_service()
if embedding_service.model:
    print("✓ Embedding service ready")
    
# Test translation service  
from src.video_doc.search import get_translation_service
translation_service = get_translation_service()
if translation_service.model or translation_service.translator:
    print("✓ Translation service ready")
```

## Step 4: Process and Search

1. **Process a video** (via web interface or CLI)
   - Upload a video or provide a URL
   - Wait for transcription to complete
   - Indexing happens automatically in the background

2. **Search across transcripts**
   - Open the web interface
   - Use the "Cross-Language Search" panel
   - Enter a query in any language
   - Select target language for translated results (optional)

## Step 5: Test the API

```bash
# Search across all transcripts
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "machine learning",
    "target_language": "es",
    "limit": 10
  }'
```

## Troubleshooting

### Models Downloading Slowly
Models are downloaded automatically on first use. This may take several minutes:
- Embedding model: ~420MB (`paraphrase-multilingual-mpnet-base-v2`)
- Translation model: ~2GB (`facebook/mbart-large-50-many-to-many-mmt`) - optional

### Database Connection Issues
Make sure PostgreSQL is running and accessible:
```bash
# Check connection
psql -U video_doc -d video_doc_db -h localhost
```

### Import Errors
If you get import errors, ensure all dependencies are installed:
```bash
pip install --upgrade sentence-transformers transformers langdetect googletrans
```

### GPU Acceleration (Optional)
For faster processing, enable GPU:
```bash
export USE_GPU=true
```

## Example Usage

### Search in English, get Spanish results
```python
from src.video_doc.search import get_search_service

service = get_search_service()
results = service.search(
    query="neural networks",
    target_language="es",  # Translate to Spanish
    limit=10
)

for result in results:
    print(f"[{result['start_time']}s] {result['text']}")
```

### Search in Spanish, find English content
```python
results = service.search(
    query="redes neuronales",  # Spanish query
    target_language="en",      # Get English results
    limit=10
)
```

## Next Steps

- See `CROSS_LANGUAGE_SEARCH_README.md` for detailed documentation
- Check indexing status: `GET /api/search/index-status/<job_id>`
- List indexed jobs: `GET /api/search/jobs`

