"""
Semantic Indexer Agent
Handles storing document chunks and embeddings in Qdrant.
"""

from typing import List

from src.models import DocumentChunk, IngestionState
from src.db.qdrant_client import QdrantManager


class SemanticIndexerAgent:
    """
    Agent responsible for indexing documents in Qdrant.
    
    Tasks:
    1. Receive chunks from ingestion agent
    2. Store chunks with embeddings in Qdrant
    3. Manage Qdrant collection
    """
    
    def __init__(self):
        """Initialize the semantic indexer agent."""
        self.qdrant_manager = QdrantManager()
    
    def index_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Index document chunks in Qdrant.
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
            
        Returns:
            True if successful
        """
        if not chunks:
            print("No chunks to index")
            return True
        
        # Ensure collection exists
        self.qdrant_manager.ensure_collection()
        
        # Filter chunks with embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        
        if len(valid_chunks) < len(chunks):
            print(f"Warning: {len(chunks) - len(valid_chunks)} chunks missing embeddings")
        
        # Upsert to Qdrant
        success = self.qdrant_manager.upsert_chunks(valid_chunks)
        
        if success:
            print(f"Successfully indexed {len(valid_chunks)} chunks to Qdrant")
        
        return success
    
    def get_collection_status(self):
        """Get the current status of the Qdrant collection."""
        return self.qdrant_manager.get_collection_info()
    
    def clear_index(self) -> bool:
        """Clear all indexed documents."""
        return self.qdrant_manager.clear_collection()


# Agent instance
semantic_indexer_agent = SemanticIndexerAgent()


def run_semantic_indexer_agent(state: IngestionState) -> IngestionState:
    """
    LangGraph node function for semantic indexing.
    
    Args:
        state: Current ingestion state with chunks
        
    Returns:
        Updated state with indexing status
    """
    agent = SemanticIndexerAgent()
    
    try:
        success = agent.index_chunks(state.chunks)
        state.indexed_to_qdrant = success
    except Exception as e:
        state.errors.append(f"Error indexing to Qdrant: {e}")
        state.indexed_to_qdrant = False
    
    return state
