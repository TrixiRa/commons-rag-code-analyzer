"""
Simple vector store using NumPy for cosine similarity search.
Uses sentence-transformers for embeddings with automatic GPU detection.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

from config import (
    INDEX_DIR, EMBEDDING_MODELS, DEFAULT_MODEL, detect_device,
    PACKAGE_INFO_BOOST, FOLDER_MATCH_BOOST, PARTIAL_FOLDER_BOOST,
    EXACT_FILE_BOOST, CLASS_IN_PATH_BOOST, TERM_MATCH_BOOST,
    MAIN_SOURCE_BOOST, TEST_SOURCE_BOOST, CHANGELOG_PENALTY
)
from utils.logging_config import get_logger
from utils.lazy_imports import lazy_import

logger = get_logger(__name__)

# Lazy load numpy
_np = lazy_import('numpy')


class SimpleVectorStore:
    """
    Minimal vector store using NumPy for cosine similarity search.
    Uses sentence-transformers for embeddings with automatic GPU detection.
    """
    
    def __init__(
        self, 
        index_dir: Path = INDEX_DIR, 
        model_name: str = DEFAULT_MODEL
    ) -> None:
        """
        Initialize the vector store.
        
        Args:
            index_dir: Directory to store/load the index
            model_name: Embedding model key or full model name
        """
        self.index_dir = index_dir
        self._embeddings: Optional[Any] = None  # numpy array
        self._documents: list[dict[str, Any]] = []
        self._model: Optional[Any] = None
        self._model_key = model_name
        self._device: Optional[str] = None
    
    @property
    def embeddings(self) -> Optional[Any]:
        """Get the embeddings array."""
        return self._embeddings
    
    @embeddings.setter
    def embeddings(self, value: Any) -> None:
        """Set the embeddings array."""
        self._embeddings = value
    
    @property
    def documents(self) -> list[dict[str, Any]]:
        """Get all indexed documents."""
        return self._documents
    
    @documents.setter
    def documents(self, value: list[dict[str, Any]]) -> None:
        """Set the documents list."""
        self._documents = value
    
    def _get_model(self) -> Any:
        """Lazy load the embedding model with automatic device detection."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            
            model_name = EMBEDDING_MODELS.get(self._model_key, self._model_key)
            self._device = detect_device()
            
            logger.info(f"Loading embedding model '{model_name}' on {self._device}")
            self._model = SentenceTransformer(model_name, device=self._device)
        
        return self._model
    
    def _embed(self, texts: list[str], batch_size: int = 32) -> Any:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        model = self._get_model()
        return model.encode(
            texts, 
            show_progress_bar=False, 
            normalize_embeddings=True,
            batch_size=batch_size
        )
    
    def build_index(self, documents: list[dict[str, Any]], batch_size: int = 32) -> None:
        """
        Build the vector index from documents.
        
        Args:
            documents: List of document dicts with 'text' field
            batch_size: Batch size for embedding (lower = less memory)
        """
        np = _np._load()
        
        logger.info("=" * 60)
        logger.info(f"Building index from {len(documents)} chunks...")
        logger.info("=" * 60)
        
        self._documents = documents
        texts = [doc["text"] for doc in documents]
        
        logger.info(f"Step 1/3: Preparing {len(texts)} text chunks (batch_size={batch_size})")
        
        logger.info("Step 2/3: Generating embeddings...")
        if self._device == "cpu" or self._device is None:
            logger.info("(This may take 30-60 seconds on CPU)")
        else:
            logger.info(f"(Using {self._device} for fast embedding)")
        
        self._embeddings = self._embed(texts, batch_size=batch_size)
        logger.info(f"Generated {self._embeddings.shape[0]} embeddings of dimension {self._embeddings.shape[1]}")
        
        logger.info("Step 3/3: Saving index to disk...")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "embeddings.npy", self._embeddings)
        with open(self.index_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2)
        
        logger.info(f"Index saved to {self.index_dir}")
        logger.info("=" * 60)
        logger.info("Index build complete!")
        logger.info("=" * 60)
    
    def load_index(self) -> bool:
        """
        Load existing index from disk.
        
        Returns:
            True if successful, False if index doesn't exist
        """
        np = _np._load()
        
        embeddings_path = self.index_dir / "embeddings.npy"
        documents_path = self.index_dir / "documents.json"
        
        if embeddings_path.exists() and documents_path.exists():
            self._embeddings = np.load(embeddings_path)
            with open(documents_path, "r", encoding="utf-8") as f:
                self._documents = json.load(f)
            logger.info(f"Loaded index with {len(self._documents)} documents")
            return True
        
        logger.warning("No existing index found")
        return False
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        prefer_source_code: bool = True
    ) -> dict[str, Any]:
        """
        Search for most relevant documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            prefer_source_code: If True, boost scores for source code files
            
        Returns:
            Dict with 'results', 'is_folder_query', 'folder_name'
        """
        np = _np._load()
        
        if self._embeddings is None or len(self._documents) == 0:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        query_embedding = self._embed([query])[0]
        
        # Cosine similarity (embeddings are already normalized)
        similarities = np.dot(self._embeddings, query_embedding)
        
        # Extract key terms
        query_lower = query.lower()
        key_terms = [t for t in re.findall(r'\b[A-Z][a-zA-Z]+\b', query) if len(t) > 3]
        key_terms_lower = [t.lower() for t in key_terms]
        
        # Detect folder queries
        folder_indicators = ['folder', 'package', 'directory', 'module', 'dir']
        is_folder_query = any(ind in query_lower for ind in folder_indicators)
        
        common_words = {
            'the', 'what', 'how', 'does', 'which', 'where', 'when', 'why',
            'kind', 'type', 'stored', 'contains', 'have', 'has', 'are', 'is',
            'functionality', 'function', 'class', 'classes', 'method', 'methods',
            'folder', 'package', 'directory', 'module', 'about', 'from', 'for',
            'with', 'this', 'that', 'java', 'code', 'files', 'file'
        }
        folder_terms = [
            w for w in re.findall(r'\b[a-z]{3,}\b', query_lower)
            if w not in common_words
        ]
        
        detected_folder: Optional[str] = None
        
        # Apply boosting
        for i, doc in enumerate(self._documents):
            similarities[i] = self._apply_boosts(
                similarities[i], doc, query_lower, key_terms, 
                key_terms_lower, folder_terms, is_folder_query, prefer_source_code
            )
            
            # Track detected folder
            if is_folder_query or folder_terms:
                path_lower = doc.get('path', '').lower()
                path_parts = path_lower.replace('\\', '/').split('/')
                for folder in folder_terms:
                    if folder in path_parts:
                        detected_folder = folder
                        break
        
        # Get top results
        top_indices = np.argsort(similarities)[-(top_k * 3):][::-1]
        
        # Handle folder query deduplication
        if is_folder_query and detected_folder:
            results = self._deduplicate_folder_results(
                top_indices, similarities, top_k
            )
        else:
            results = [
                (self._documents[idx], float(similarities[idx]))
                for idx in top_indices[:top_k]
            ]
        
        return {
            'results': results,
            'is_folder_query': is_folder_query and detected_folder is not None,
            'folder_name': detected_folder
        }
    
    def _apply_boosts(
        self,
        similarity: float,
        doc: dict[str, Any],
        query_lower: str,
        key_terms: list[str],
        key_terms_lower: list[str],
        folder_terms: list[str],
        is_folder_query: bool,
        prefer_source_code: bool
    ) -> float:
        """Apply various boosting factors to similarity score."""
        doc_type = doc.get('type', '')
        path = doc.get('path', '')
        path_lower = path.lower()
        text_lower = doc.get('text', '').lower()
        
        # Folder/package boost
        if is_folder_query or folder_terms:
            path_parts = path_lower.replace('\\', '/').split('/')
            for folder in folder_terms:
                if folder in path_parts:
                    if 'package-info.java' in path_lower:
                        similarity += PACKAGE_INFO_BOOST
                    else:
                        similarity += FOLDER_MATCH_BOOST
                    break
                elif folder in path_lower:
                    similarity += PARTIAL_FOLDER_BOOST
        
        # Exact file match boost
        for term in key_terms:
            filename = path.split('/')[-1] if '/' in path else path
            if filename.lower() == f"{term.lower()}.java":
                similarity += EXACT_FILE_BOOST
            elif term.lower() in path_lower:
                similarity *= CLASS_IN_PATH_BOOST
        
        # Keyword boost
        term_matches = sum(1 for t in key_terms_lower if t in text_lower)
        if term_matches > 0:
            similarity *= (1 + TERM_MATCH_BOOST * term_matches)
        
        # Source code prioritization
        if prefer_source_code:
            if doc_type == 'java' and 'src/main/java' in path:
                similarity *= MAIN_SOURCE_BOOST
            elif doc_type == 'java_test':
                similarity *= TEST_SOURCE_BOOST
            elif doc_type in ('changelog', 'documentation') or 'changes.xml' in path_lower:
                similarity *= CHANGELOG_PENALTY
        
        return similarity
    
    def _deduplicate_folder_results(
        self,
        top_indices: Any,
        similarities: Any,
        top_k: int
    ) -> list[tuple[dict[str, Any], float]]:
        """Deduplicate results for folder queries."""
        results: list[tuple[dict[str, Any], float]] = []
        seen_files: set[str] = set()
        
        for idx in top_indices:
            doc = self._documents[idx]
            path = doc.get('path', '')
            filename = path.split('/')[-1] if '/' in path else path
            
            if filename in seen_files:
                continue
            seen_files.add(filename)
            results.append((doc, float(similarities[idx])))
            
            if len(results) >= top_k:
                break
        
        return results
