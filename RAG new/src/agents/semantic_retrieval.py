"""
Semantic Retrieval Agent
Handles semantic search using Qdrant vector database.
"""

from typing import List

from config import settings
from src.models import RetrievalResult, AgenticRAGState
from src.db.qdrant_client import QdrantManager
from src.utils.embeddings import EmbeddingGenerator
from src.utils.language import detect_language


class SemanticRetrievalAgent:
    """
    Agent responsible for semantic similarity search in Qdrant.
    
    Tasks:
    1. Convert user query to embedding
    2. Query Qdrant for top-k similar chunks
    3. Return results with relevance scores
    """
    
    def __init__(self):
        """Initialize the semantic retrieval agent."""
        self.qdrant_manager = QdrantManager()
        self.embedding_generator = EmbeddingGenerator()
    
    def search(
        self,
        query: str,
        top_k: int = None,
        filter_language: str = None
    ) -> List[RetrievalResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: User query text
            top_k: Number of results to return
            filter_language: Optional language filter
            
        Returns:
            List of RetrievalResult objects
        """
        if not query or not query.strip():
            return []
        
        if top_k is None:
            top_k = settings.top_k_results
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)
        
        # Build filter conditions
        filter_conditions = {}
        if filter_language:
            filter_conditions["language"] = filter_language
        
        # Search Qdrant
        results = self.qdrant_manager.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_conditions=filter_conditions if filter_conditions else None
        )
        
        print(f"Semantic search found {len(results)} results")
        return results


# Agent instance
semantic_retrieval_agent = SemanticRetrievalAgent()


def run_semantic_retrieval_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    LangGraph node function for semantic retrieval.
    
    Args:
        state: Current RAG state with query
        
    Returns:
        Updated state with Qdrant results
    """
    agent = SemanticRetrievalAgent()
    
    try:
        # Detect query language
        state.query_language = detect_language(state.query)
        
        # Search
        results = agent.search(
            query=state.query,
            top_k=settings.top_k_results
        )
        
        state.qdrant_results = results
        
    except Exception as e:
        state.errors.append(f"Error in semantic retrieval: {e}")
    
    return state
