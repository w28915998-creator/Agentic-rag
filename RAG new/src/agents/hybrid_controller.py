"""
Hybrid Retrieval Controller
Merges results from Qdrant semantic search and Neo4j graph traversal.
"""

from typing import List, Dict, Set

from src.models import RetrievalResult, AgenticRAGState
from src.db.qdrant_client import QdrantManager


class HybridRetrievalController:
    """
    Controller that merges and prioritizes retrieval results.
    
    Tasks:
    1. Merge Qdrant and Neo4j results
    2. Remove duplicates
    3. Prioritize graph-validated chunks
    4. Rank by combined relevance
    """
    
    def __init__(self):
        """Initialize the hybrid controller."""
        self.qdrant_manager = QdrantManager()
    
    def merge_results(
        self,
        qdrant_results: List[RetrievalResult],
        neo4j_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Merge and rank results from both sources.
        
        Args:
            qdrant_results: Results from semantic search
            neo4j_results: Results from graph traversal
            
        Returns:
            Merged and ranked results
        """
        # Track chunks by chunk_id for deduplication
        chunk_map: Dict[str, RetrievalResult] = {}
        graph_validated_ids: Set[str] = set()
        
        # Process Neo4j results first (to know which chunks are graph-validated)
        for result in neo4j_results:
            graph_validated_ids.add(result.chunk_id)
            # Mark as graph validated by default since it came from the graph
            if not result.metadata:
                result.metadata = {}
            result.metadata["graph_validated"] = True
            
            if result.chunk_id not in chunk_map:
                # Need to fetch actual text from Qdrant since Neo4j only has references
                chunk_map[result.chunk_id] = result
        
        # Process Qdrant results
        for result in qdrant_results:
            if result.chunk_id in chunk_map:
                # Chunk exists from Neo4j - boost its score
                existing = chunk_map[result.chunk_id]
                
                # Use Qdrant's text (more complete) and combine scores
                boosted_score = (result.score + existing.score) / 2 + 0.1  # Boost for graph validation
                
                chunk_map[result.chunk_id] = RetrievalResult(
                    doc_id=result.doc_id,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=min(boosted_score, 1.0),  # Cap at 1.0
                    source="hybrid",
                    metadata={
                        **result.metadata,
                        "graph_validated": True,
                        "neo4j_metadata": existing.metadata
                    }
                )
            else:
                # New chunk from Qdrant
                result.metadata["graph_validated"] = False
                chunk_map[result.chunk_id] = result
        
        # Convert to list and sort by score
        merged = list(chunk_map.values())
        merged.sort(key=lambda x: x.score, reverse=True)
        
        # Prioritize graph-validated results
        graph_validated = [r for r in merged if r.metadata.get("graph_validated")]
        non_validated = [r for r in merged if not r.metadata.get("graph_validated")]
        
        # Interleave: give priority to graph-validated but include semantic results
        final_results = []
        max_results = 10  # Limit total results
        
        gv_idx, nv_idx = 0, 0
        while len(final_results) < max_results:
            # Add 2 graph-validated for every 1 non-validated
            for _ in range(2):
                if gv_idx < len(graph_validated):
                    final_results.append(graph_validated[gv_idx])
                    gv_idx += 1
            
            if nv_idx < len(non_validated):
                final_results.append(non_validated[nv_idx])
                nv_idx += 1
            
            # Break if we've exhausted both lists
            if gv_idx >= len(graph_validated) and nv_idx >= len(non_validated):
                break
        
        print(f"Merged results: {len(graph_validated)} graph-validated, {len(non_validated)} semantic-only")
        return final_results
    
    def fetch_chunk_text(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Fetch full text for chunks that only have references.
        
        For Neo4j results that don't have full text.
        
        Args:
            results: Results that may need text fetching
            
        Returns:
            Results with full text
        """
        # Identify chunks needing text
        needs_text = [
            r for r in results 
            if r.source == "neo4j" and "Entity:" in r.text
        ]
        
        if not needs_text:
            return results
        
        # Batch fetch from Qdrant
        try:
            from config import settings
            
            chunk_ids = [r.chunk_id for r in needs_text]
            
            points = self.qdrant_manager.client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=chunk_ids,
                with_payload=True
            )
            
            # Map results
            text_map = {
                point.id: point.payload.get("text", "") 
                for point in points 
                if point.payload
            }
            
            for result in needs_text:
                if result.chunk_id in text_map and text_map[result.chunk_id]:
                    result.text = text_map[result.chunk_id]
                    
        except Exception as e:
            print(f"Error batch fetching texts: {e}")
            # Keep existing text if fetch fails
        
        return results


# Controller instance
hybrid_controller = HybridRetrievalController()


def run_hybrid_controller(state: AgenticRAGState) -> AgenticRAGState:
    """
    LangGraph node function for hybrid retrieval.
    
    Args:
        state: Current RAG state with both Qdrant and Neo4j results
        
    Returns:
        Updated state with merged evidence
    """
    controller = HybridRetrievalController()
    
    try:
        merged = controller.merge_results(
            qdrant_results=state.qdrant_results,
            neo4j_results=state.neo4j_results
        )
        
        # Fetch full text for chunks that only have graph references
        merged = controller.fetch_chunk_text(merged)
        
        state.merged_evidence = merged
        
    except Exception as e:
        state.errors.append(f"Error in hybrid retrieval: {e}")
        # Fallback to Qdrant results only
        state.merged_evidence = state.qdrant_results
    
    return state
