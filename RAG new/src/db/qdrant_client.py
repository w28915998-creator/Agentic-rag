"""
Qdrant vector database client for storing and retrieving document embeddings.
"""

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import settings
from src.models import DocumentChunk, RetrievalResult


class QdrantManager:
    """
    Manages Qdrant vector database operations.
    Handles collection creation, document indexing, and similarity search.
    """
    
    _instance: Optional['QdrantManager'] = None
    _client: Optional[QdrantClient] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Qdrant client."""
        if self._client is None:
            self._connect()
    
    def _connect(self):
        """Establish connection to Qdrant server."""
        print(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=60
        )
        print("Connected to Qdrant successfully")
    
    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client."""
        return self._client
    
    def ensure_collection(self) -> bool:
        """
        Ensure the collection exists, create if not.
        
        Returns:
            True if collection exists or was created successfully
        """
        try:
            # Check if collection exists
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if settings.qdrant_collection in collection_names:
                print(f"Collection '{settings.qdrant_collection}' already exists")
                return True
            
            # Create collection
            self._client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=qdrant_models.VectorParams(
                    size=settings.embedding_dimension,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            print(f"Created collection '{settings.qdrant_collection}'")
            return True
            
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            return False
    
    def upsert_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Upsert document chunks with their embeddings.
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
            
        Returns:
            True if successful
        """
        if not chunks:
            return True
        
        # Ensure collection exists
        self.ensure_collection()
        
        # Prepare points for upsert
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                print(f"Warning: Chunk {chunk.chunk_id} has no embedding, skipping")
                continue
            
            point = qdrant_models.PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "language": chunk.language.value,
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                    "temporal_context": chunk.temporal_context
                }
            )
            points.append(point)
        
        if not points:
            print("No valid points to upsert")
            return True
        
        try:
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self._client.upsert(
                    collection_name=settings.qdrant_collection,
                    points=batch
                )
            
            print(f"Upserted {len(points)} chunks to Qdrant")
            return True
            
        except Exception as e:
            print(f"Error upserting chunks: {e}")
            return False
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filter_conditions: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            List of RetrievalResult objects
        """
        if top_k is None:
            top_k = settings.top_k_results
        
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value)
                    )
                )
            query_filter = qdrant_models.Filter(must=must_conditions)
        
        try:
            results = self._client.query_points(
                collection_name=settings.qdrant_collection,
                query=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True
            ).points
            
            retrieval_results = []
            for result in results:
                payload = result.payload
                retrieval_results.append(RetrievalResult(
                    doc_id=payload.get("doc_id", ""),
                    chunk_id=payload.get("chunk_id", ""),
                    text=payload.get("text", ""),
                    score=result.score,
                    source="qdrant",
                    metadata={
                        "language": payload.get("language"),
                        "source_file": payload.get("source_file"),
                        "chunk_index": payload.get("chunk_index")
                    }
                ))
            
            return retrieval_results
            
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []
    
    def delete_by_doc_id(self, doc_id: str) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful
        """
        try:
            self._client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="doc_id",
                                match=qdrant_models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            )
            print(f"Deleted chunks for doc_id: {doc_id}")
            return True
            
        except Exception as e:
            print(f"Error deleting chunks: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self._client.get_collection(settings.qdrant_collection)
            return {
                "name": settings.qdrant_collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Delete and recreate the collection."""
        try:
            self._client.delete_collection(settings.qdrant_collection)
            print(f"Deleted collection '{settings.qdrant_collection}'")
            return self.ensure_collection()
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False


# Global instance
qdrant_manager = QdrantManager()
