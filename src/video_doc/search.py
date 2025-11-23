"""
Cross-language search functionality for video transcripts.

This module provides:
- Vector embeddings for semantic search
- Cross-language translation capabilities
- Search indexing and querying
- Multilingual support
"""

import os
import json
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, ForeignKey, Index, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from .database import Base, get_db_session
from .monitoring import get_logger

logger = get_logger(__name__)


class TranscriptChunk(Base):
    """Store transcript chunks with embeddings for search."""
    __tablename__ = "transcript_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey('processing_jobs.id'), nullable=False, index=True)
    
    # Text content
    text = Column(Text, nullable=False)
    original_language = Column(String(10))  # Language code from transcription
    
    # Timestamps from video
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    
    # Embedding vector (stored as JSON array for compatibility)
    # For PostgreSQL with PGVector, this would be a vector type
    embedding = Column(JSONB)  # Store as JSON array of floats
    
    # Metadata
    chunk_index = Column(Integer, nullable=False)  # Order within job
    metadata = Column(JSONB, default=dict)  # Additional metadata (speaker, etc.)
    
    # Indexing info
    indexed_at = Column(DateTime, default=datetime.utcnow)
    embedding_model = Column(String(50))  # Which model generated the embedding
    
    # Indexes
    __table_args__ = (
        Index('idx_chunk_job', 'job_id'),
        Index('idx_chunk_time', 'job_id', 'start_time'),
        Index('idx_chunk_index', 'job_id', 'chunk_index'),
    )


class SearchIndex(Base):
    """Track indexing status for jobs."""
    __tablename__ = "search_indexes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey('processing_jobs.id'), nullable=False, unique=True, index=True)
    
    status = Column(String(20), nullable=False, default='pending')  # pending, indexing, completed, failed
    total_chunks = Column(Integer, default=0)
    indexed_chunks = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    indexed_at = Column(DateTime)
    error_message = Column(Text)
    
    # Indexes
    __table_args__ = (
        Index('idx_index_status', 'status'),
    )


def detect_language(text: str) -> str:
    """Detect language of text using simple heuristics or a language detection library."""
    if not text or not text.strip():
        return 'en'
    
    # Simple heuristic: check for common patterns
    # For production, use langdetect or polyglot
    try:
        from langdetect import detect, LangDetectException
        try:
            return detect(text)
        except LangDetectException:
            return 'en'  # Default to English
    except ImportError:
        # Fallback: simple character-based detection
        # Check for common non-ASCII patterns
        sample = text[:200] if len(text) > 200 else text
        if any(ord(c) > 127 for c in sample):
            # Contains non-ASCII - likely not English
            # This is very basic; proper detection needs a library
            return 'unknown'
        return 'en'


class TranslationService:
    """Service for translating text between languages."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None
        self.device = 'cpu'
        self._initialize_translator()
    
    def _initialize_translator(self):
        """Initialize translation model."""
        try:
            # Use googletrans as a fallback, or mBart for better quality
            # For production, consider using HuggingFace Transformers with mBART
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
                
                # Use mBART50 which supports 50 languages
                model_name = os.getenv('TRANSLATION_MODEL', 'facebook/mbart-large-50-many-to-many-mmt')
                
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    self.device = 'cuda' if os.getenv('USE_GPU', 'false').lower() == 'true' else 'cpu'
                    self.model.to(self.device)
                    self.use_transformers = True
                    self.logger.info(f"Initialized transformer-based translator: {model_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load transformer model: {e}, using googletrans fallback")
                    self.use_transformers = False
                    
            except ImportError:
                self.use_transformers = False
                self.logger.info("Transformers not available, using googletrans fallback")
                
            if not self.use_transformers:
                try:
                    from googletrans import Translator
                    self.translator = Translator()
                    self.logger.info("Initialized googletrans translator")
                except ImportError:
                    self.logger.warning("No translation library available. Install googletrans or transformers.")
                    self.translator = None
                    
        except Exception as e:
            self.logger.error(f"Error initializing translator: {e}")
            self.translator = None
            self.model = None
    
    def translate(self, text: str, target_language: str = 'en', source_language: str = 'auto') -> str:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'en', 'es', 'fr')
            source_language: Source language code or 'auto' for auto-detection
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text
        
        if target_language == 'auto' or target_language == source_language:
            return text
        
        try:
            if self.use_transformers and self.model:
                return self._translate_with_transformers(text, source_language, target_language)
            elif self.translator:
                return self._translate_with_googletrans(text, target_language, source_language)
            else:
                self.logger.warning("No translation service available, returning original text")
                return text
        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            return text
    
    def _translate_with_transformers(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate using transformers model."""
        try:
            # Map language codes to mBART language codes
            lang_map = {
                'en': 'en_XX', 'es': 'es_XX', 'fr': 'fr_XX', 'de': 'de_DE',
                'it': 'it_IT', 'pt': 'pt_XX', 'ru': 'ru_RU', 'ja': 'ja_XX',
                'ko': 'ko_KR', 'zh': 'zh_CN', 'ar': 'ar_AR', 'hi': 'hi_IN'
            }
            
            src_code = lang_map.get(src_lang, 'en_XX')
            tgt_code = lang_map.get(tgt_lang, 'en_XX')
            
            self.tokenizer.src_lang = src_code
            encoded = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code]
            )
            
            translated = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translated
        except Exception as e:
            self.logger.error(f"Transformers translation error: {e}")
            return text
    
    def _translate_with_googletrans(self, text: str, target_lang: str, source_lang: str) -> str:
        """Translate using googletrans."""
        try:
            if source_lang == 'auto':
                result = self.translator.translate(text, dest=target_lang)
            else:
                result = self.translator.translate(text, src=source_lang, dest=target_lang)
            return result.text
        except Exception as e:
            self.logger.error(f"Googletrans error: {e}")
            return text


class EmbeddingService:
    """Service for generating multilingual embeddings."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None
        self.tokenizer = None
        self.device = 'cpu'
        self.embedding_dim = 768  # Default for sentence-transformers
        self._initialize_embedder()
    
    def _initialize_embedder(self):
        """Initialize embedding model."""
        try:
            # Use sentence-transformers which are optimized for multilingual embeddings
            from sentence_transformers import SentenceTransformer
            
            # Use multilingual model that supports 50+ languages
            model_name = os.getenv(
                'EMBEDDING_MODEL',
                'paraphrase-multilingual-mpnet-base-v2'  # Supports 50+ languages
            )
            
            try:
                self.device = 'cuda' if os.getenv('USE_GPU', 'false').lower() == 'true' else 'cpu'
                self.model = SentenceTransformer(model_name, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"Initialized embedding model: {model_name} (dim={self.embedding_dim})")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                self.model = None
                
        except ImportError:
            self.logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            self.logger.error(f"Error initializing embedder: {e}")
            self.model = None
    
    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        if not texts:
            return np.array([])
        
        try:
            # The model handles batching internally
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Embedding error: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.embed([query])[0]


# Global service instances
_translation_service = None
_embedding_service = None


def get_translation_service() -> TranslationService:
    """Get or create translation service instance."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


class SearchService:
    """Service for performing cross-language search."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.translation_service = get_translation_service()
        self.embedding_service = get_embedding_service()
    
    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """
        Calculate keyword matching score between query and text.
        
        Args:
            query: Search query
            text: Text to match against
            
        Returns:
            Score between 0 and 1
        """
        if not query or not text:
            return 0.0
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Extract words from query
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        if not query_words:
            return 0.0
        
        # Count matches
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        matched_words = query_words & text_words
        
        # Calculate score based on:
        # 1. Ratio of matched words to query words
        # 2. Exact phrase matches (bonus)
        word_match_ratio = len(matched_words) / len(query_words) if query_words else 0.0
        
        # Check for exact phrase match (case-insensitive)
        phrase_bonus = 0.0
        if query_lower in text_lower:
            phrase_bonus = 0.3  # Bonus for exact phrase
        
        # Combine scores (max 1.0)
        score = min(1.0, word_match_ratio * 0.7 + phrase_bonus)
        return score
    
    def _search_keywords(
        self,
        query: str,
        job_ids: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based text search.
        
        Args:
            query: Search query
            job_ids: Filter by specific job IDs
            limit: Maximum number of results
            min_score: Minimum keyword score (0-1)
            
        Returns:
            List of search results with chunks and metadata
        """
        try:
            db = get_db_session()
            try:
                # Build query
                chunk_query = db.query(TranscriptChunk)
                
                if job_ids:
                    chunk_query = chunk_query.filter(
                        TranscriptChunk.job_id.in_([uuid.UUID(jid) for jid in job_ids])
                    )
                
                # Get all chunks
                chunks = chunk_query.all()
                
                if not chunks:
                    return []
                
                # Calculate keyword scores
                results = []
                for chunk in chunks:
                    keyword_score = self._calculate_keyword_score(query, chunk.text)
                    
                    if keyword_score >= min_score:
                        results.append({
                            'chunk_id': str(chunk.id),
                            'job_id': str(chunk.job_id),
                            'text': chunk.text,
                            'original_text': chunk.text,
                            'original_language': chunk.original_language,
                            'start_time': chunk.start_time,
                            'end_time': chunk.end_time,
                            'similarity': float(keyword_score),
                            'keyword_score': float(keyword_score),
                            'semantic_score': 0.0,
                            'chunk_index': chunk.chunk_index,
                            'metadata': chunk.metadata or {}
                        })
                
                # Sort by keyword score and limit
                results.sort(key=lambda x: x['keyword_score'], reverse=True)
                return results[:limit]
            finally:
                db.close()
                
        except Exception as e:
            self.logger.error(f"Keyword search error: {e}")
            return []
    
    def search(
        self,
        query: str,
        target_language: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.5,
        search_mode: str = 'semantic'  # 'semantic', 'keyword', or 'hybrid'
    ) -> List[Dict[str, Any]]:
        """
        Perform cross-language search with multiple modes.
        
        Args:
            query: Search query in any language
            target_language: Language to return results in (None = original)
            job_ids: Filter by specific job IDs
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            search_mode: 'semantic', 'keyword', or 'hybrid'
            
        Returns:
            List of search results with chunks and metadata
        """
        if search_mode == 'keyword':
            return self._search_keywords(query, job_ids, limit, min_score)
        elif search_mode == 'hybrid':
            return self._search_hybrid(query, target_language, job_ids, limit, min_score)
        else:  # semantic (default)
            return self._search_semantic(query, target_language, job_ids, limit, min_score)
    
    def _search_semantic(
        self,
        query: str,
        target_language: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query: Search query in any language
            target_language: Language to return results in (None = original)
            job_ids: Filter by specific job IDs
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of search results with chunks and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            
            # Perform vector search in database
            db = get_db_session()
            try:
                # Build query
                chunk_query = db.query(TranscriptChunk).filter(
                    TranscriptChunk.embedding.isnot(None)
                )
                
                if job_ids:
                    chunk_query = chunk_query.filter(
                        TranscriptChunk.job_id.in_([uuid.UUID(jid) for jid in job_ids])
                    )
                
                # Get all matching chunks
                chunks = chunk_query.all()
                
                if not chunks:
                    return []
                
                # Calculate cosine similarity for each chunk
                results = []
                for chunk in chunks:
                    if not chunk.embedding:
                        continue
                    
                    chunk_embedding = np.array(chunk.embedding)
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    
                    if similarity >= min_score:
                        # Translate text if needed
                        result_text = chunk.text
                        if target_language and target_language != chunk.original_language:
                            try:
                                result_text = self.translation_service.translate(
                                    chunk.text,
                                    target_language=target_language,
                                    source_language=chunk.original_language or 'auto'
                                )
                            except Exception as e:
                                self.logger.warning(f"Translation failed for chunk {chunk.id}: {e}")
                        
                        results.append({
                            'chunk_id': str(chunk.id),
                            'job_id': str(chunk.job_id),
                            'text': result_text,
                            'original_text': chunk.text,
                            'original_language': chunk.original_language,
                            'start_time': chunk.start_time,
                            'end_time': chunk.end_time,
                            'similarity': float(similarity),
                            'semantic_score': float(similarity),
                            'keyword_score': 0.0,
                            'chunk_index': chunk.chunk_index,
                            'metadata': chunk.metadata or {}
                        })
                
                # Sort by similarity and limit
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results[:limit]
            finally:
                db.close()
                
        except Exception as e:
            self.logger.error(f"Semantic search error: {e}")
            return []
    
    def _search_hybrid(
        self,
        query: str,
        target_language: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query in any language
            target_language: Language to return results in (None = original)
            job_ids: Filter by specific job IDs
            limit: Maximum number of results
            min_score: Minimum combined score (0-1)
            semantic_weight: Weight for semantic score (0-1)
            keyword_weight: Weight for keyword score (0-1)
            
        Returns:
            List of search results with chunks and metadata
        """
        try:
            # Normalize weights
            total_weight = semantic_weight + keyword_weight
            if total_weight > 0:
                semantic_weight /= total_weight
                keyword_weight /= total_weight
            else:
                semantic_weight = 0.5
                keyword_weight = 0.5
            
            # Get semantic results
            semantic_results = self._search_semantic(
                query, target_language, job_ids, limit * 2, min_score * 0.5
            )
            
            # Get keyword results
            keyword_results = self._search_keywords(
                query, job_ids, limit * 2, min_score * 0.5
            )
            
            # Combine results by chunk_id
            combined_results: Dict[str, Dict[str, Any]] = {}
            
            # Add semantic results
            for result in semantic_results:
                chunk_id = result['chunk_id']
                result['semantic_score'] = result.get('similarity', 0.0)
                result['keyword_score'] = 0.0
                combined_results[chunk_id] = result
            
            # Merge keyword results
            for result in keyword_results:
                chunk_id = result['chunk_id']
                if chunk_id in combined_results:
                    # Update existing result with keyword score
                    combined_results[chunk_id]['keyword_score'] = result.get('keyword_score', 0.0)
                else:
                    # Add new result
                    result['semantic_score'] = 0.0
                    combined_results[chunk_id] = result
            
            # Calculate combined scores
            final_results = []
            for result in combined_results.values():
                semantic_score = result.get('semantic_score', 0.0)
                keyword_score = result.get('keyword_score', 0.0)
                
                # Combine scores
                combined_score = (semantic_score * semantic_weight) + (keyword_score * keyword_weight)
                
                if combined_score >= min_score:
                    result['similarity'] = combined_score
                    result['combined_score'] = combined_score
                    
                    # Translate text if needed
                    if target_language and target_language != result.get('original_language'):
                        try:
                            result['text'] = self.translation_service.translate(
                                result['original_text'],
                                target_language=target_language,
                                source_language=result.get('original_language') or 'auto'
                            )
                        except Exception as e:
                            self.logger.warning(f"Translation failed: {e}")
                    
                    final_results.append(result)
            
            # Sort by combined score and limit
            final_results.sort(key=lambda x: x['combined_score'], reverse=True)
            return final_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Hybrid search error: {e}")
            return []
    
    def index_transcript(
        self,
        job_id: str,
        segments_json_path: Path,
        chunk_size: int = 3,  # Number of segments per chunk
        overlap: int = 1  # Number of overlapping segments
    ) -> bool:
        """
        Index transcript segments for search.
        
        Args:
            job_id: Processing job ID
            segments_json_path: Path to segments.json file
            chunk_size: Number of segments to combine per chunk
            overlap: Number of overlapping segments between chunks
            
        Returns:
            True if successful
        """
        try:
            # Load segments
            if not segments_json_path.exists():
                self.logger.error(f"Segments file not found: {segments_json_path}")
                return False
            
            with open(segments_json_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            
            if not segments:
                self.logger.warning(f"No segments found for job {job_id}")
                return False
            
            # Detect language from first few segments
            sample_text = ' '.join([s.get('text', '') for s in segments[:5]])
            detected_language = detect_language(sample_text)
            
            # Create chunks with overlap
            chunks = []
            for i in range(0, len(segments), chunk_size - overlap):
                chunk_segments = segments[i:i + chunk_size]
                if not chunk_segments:
                    break
                
                chunk_text = ' '.join([s.get('text', '') for s in chunk_segments])
                start_time = chunk_segments[0].get('start', 0.0)
                end_time = chunk_segments[-1].get('end', 0.0)
                
                chunks.append({
                    'text': chunk_text,
                    'start_time': start_time,
                    'end_time': end_time,
                    'index': len(chunks)
                })
            
            if not chunks:
                self.logger.warning(f"No chunks created for job {job_id}")
                return False
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_service.embed(chunk_texts)
            
            # Store chunks in database
            db = get_db_session()
            try:
                job_uuid = uuid.UUID(job_id)
                
                # Update index status
                search_index = db.query(SearchIndex).filter(SearchIndex.job_id == job_uuid).first()
                if not search_index:
                    search_index = SearchIndex(
                        job_id=job_uuid,
                        status='indexing',
                        total_chunks=len(chunks)
                    )
                    db.add(search_index)
                else:
                    search_index.status = 'indexing'
                    search_index.total_chunks = len(chunks)
                    search_index.indexed_chunks = 0
                
                db.commit()
                
                # Insert chunks
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_record = TranscriptChunk(
                        job_id=job_uuid,
                        text=chunk['text'],
                        original_language=detected_language,
                        start_time=chunk['start_time'],
                        end_time=chunk['end_time'],
                        chunk_index=chunk['index'],
                        embedding=embedding.tolist(),
                        embedding_model=os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-mpnet-base-v2'),
                        metadata={}
                    )
                    db.add(chunk_record)
                    search_index.indexed_chunks = i + 1
                
                search_index.status = 'completed'
                search_index.indexed_at = datetime.utcnow()
                db.commit()
                
                self.logger.info(f"Indexed {len(chunks)} chunks for job {job_id}")
                return True
            finally:
                db.close()
                
        except Exception as e:
            self.logger.error(f"Indexing error for job {job_id}: {e}")
            # Update index status to failed
            try:
                db = get_db_session()
                try:
                    search_index = db.query(SearchIndex).filter(SearchIndex.job_id == uuid.UUID(job_id)).first()
                    if search_index:
                        search_index.status = 'failed'
                        search_index.error_message = str(e)
                        db.commit()
                finally:
                    db.close()
            except:
                pass
            return False


# Global search service instance
_search_service = None


def get_search_service() -> SearchService:
    """Get or create search service instance."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service

