# Cross-Language Search Feature

## Overview

The cross-language search feature allows users to search across video transcripts in any language and receive results in their preferred language. The system uses semantic search with multilingual embeddings to find relevant content regardless of the query language.

## Features

- **Multilingual Semantic Search**: Search using any language, find results from transcripts in any language
- **Automatic Translation**: Results can be automatically translated to your preferred language
- **Semantic Matching**: Uses vector embeddings to find semantically similar content, not just keyword matches
- **Automatic Indexing**: Transcripts are automatically indexed after processing completes
- **Real-time Search**: Fast search across all indexed transcripts

## How It Works

### 1. Indexing Process

When a video is processed and transcription completes:

1. Transcript segments are chunked together (3 segments per chunk by default, with 1 segment overlap)
2. Language is automatically detected for each transcript
3. Multilingual embeddings are generated using `paraphrase-multilingual-mpnet-base-v2`
4. Chunks are stored in the database with their embeddings, timestamps, and metadata

### 2. Search Process

When a user performs a search:

1. Query is embedded using the same multilingual model
2. Vector similarity search finds matching chunks across all indexed transcripts
3. Results are sorted by similarity score
4. Optionally, results are translated to the target language if requested
5. Results are returned with timestamps, similarity scores, and metadata

## Architecture

### Database Models

- **TranscriptChunk**: Stores individual transcript chunks with embeddings
- **SearchIndex**: Tracks indexing status for each processing job

### Services

- **TranslationService**: Handles translation between languages (supports 50+ languages via mBART or googletrans)
- **EmbeddingService**: Generates multilingual embeddings using sentence-transformers
- **SearchService**: Performs semantic search and manages indexing

## API Endpoints

### POST /api/search

Perform cross-language semantic search.

**Request:**
```json
{
  "query": "machine learning algorithms",
  "target_language": "es",  // Optional: translate results to Spanish
  "job_ids": ["job-id-1", "job-id-2"],  // Optional: filter by specific jobs
  "limit": 10,
  "min_score": 0.5
}
```

**Response:**
```json
{
  "query": "machine learning algorithms",
  "target_language": "es",
  "results": [
    {
      "chunk_id": "uuid",
      "job_id": "uuid",
      "text": "algoritmos de aprendizaje automático",  // Translated if requested
      "original_text": "machine learning algorithms",  // Original text
      "original_language": "en",
      "start_time": 120.5,
      "end_time": 135.2,
      "similarity": 0.87,
      "chunk_index": 5,
      "metadata": {}
    }
  ],
  "count": 1
}
```

### POST /api/search/index/<job_id>

Manually trigger indexing for a completed job.

### GET /api/search/index-status/<job_id>

Get indexing status for a job.

### GET /api/search/jobs

List all indexed jobs.

## Configuration

### Environment Variables

- `EMBEDDING_MODEL`: Embedding model to use (default: `paraphrase-multilingual-mpnet-base-v2`)
- `TRANSLATION_MODEL`: Translation model to use (default: `facebook/mbart-large-50-many-to-many-mmt`)
- `USE_GPU`: Enable GPU acceleration (default: `false`)

### Supported Languages

The system supports 50+ languages including:
- English, Spanish, French, German, Italian, Portuguese
- Russian, Japanese, Korean, Chinese, Arabic, Hindi
- And many more...

## Usage

### Web Interface

1. Process a video (upload or URL)
2. Wait for transcription to complete (indexing happens automatically)
3. Use the "Cross-Language Search" panel on the right side
4. Enter your query in any language
5. Optionally select a target language for translated results
6. View results with timestamps and similarity scores

### Programmatic Access

```python
import requests

# Search across all transcripts
response = requests.post('http://localhost:5000/api/search', 
    json={
        'query': 'deep learning',
        'target_language': 'es',
        'limit': 10
    },
    headers={'Authorization': 'Bearer YOUR_TOKEN'}
)

results = response.json()['results']
for result in results:
    print(f"Found at {result['start_time']}s: {result['text']}")
```

## Dependencies

The following packages are required:

- `sentence-transformers>=2.2.2`: For multilingual embeddings
- `transformers>=4.30.0`: For translation models (optional)
- `langdetect>=1.0.9`: For language detection
- `googletrans>=4.0.0rc1`: Fallback translation service

Install with:
```bash
pip install sentence-transformers transformers langdetect googletrans
```

## Performance

- **Indexing**: ~1-2 seconds per minute of transcript
- **Search**: <100ms for typical queries across thousands of chunks
- **Embedding Generation**: ~10-50ms per chunk (GPU) or ~50-200ms (CPU)

## Limitations

- Indexing happens automatically but may take a few minutes for long videos
- Translation quality depends on the model used (mBART provides better quality than googletrans)
- GPU acceleration is recommended for large-scale deployments
- First search may be slower as models are loaded into memory

## Future Enhancements

- [ ] Support for PGVector for better performance at scale
- [ ] Advanced filtering (by date, speaker, video type)
- [ ] Search result highlighting in video player
- [ ] Batch search and export
- [ ] Search analytics and popular queries
- [ ] Multi-vector search (combining semantic and keyword search)

## Troubleshooting

### Search returns no results

1. Check that transcripts have been indexed: `/api/search/index-status/<job_id>`
2. Verify embeddings were generated successfully
3. Try lowering `min_score` threshold
4. Check database connectivity

### Translation not working

1. Verify translation dependencies are installed
2. Check model files are downloaded (happens automatically on first use)
3. For GPU acceleration, set `USE_GPU=true`

### Slow indexing

1. Enable GPU acceleration with `USE_GPU=true`
2. Reduce chunk size for faster processing (edit `chunk_size` in indexing)
3. Process indexing in background (already implemented)

## Security

- Search respects user permissions (requires `VIEW_JOB` permission)
- Results are filtered by user access rights
- Indexing is automatic but can be disabled per job if needed


