"""
Embedding generation utilities using sentence-transformers.
Supports multilingual embeddings for English and Urdu.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from config import settings


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers.
    Uses a multilingual model to support English, Urdu, and mixed text.
    """
    
    _instance: Optional['EmbeddingGenerator'] = None
    _model: Optional[SentenceTransformer] = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding model."""
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        print(f"Loading embedding model: {settings.embedding_model}")
        self._model = SentenceTransformer(settings.embedding_model)
        print(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.dimension
        
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            return [[0.0] * self.dimension for _ in texts]
        
        # Generate embeddings for valid texts
        embeddings = self._model.encode(
            valid_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(valid_texts) > 100
        )
        
        # Create result list with zero vectors for empty texts
        result = [[0.0] * self.dimension for _ in texts]
        
        for i, embedding in zip(valid_indices, embeddings):
            result[i] = embedding.tolist()
        
        return result
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        May use different normalization than document embeddings.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embed_text(query)
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


# Global instance
embedding_generator = EmbeddingGenerator()
