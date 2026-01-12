"""
Graph Retrieval Agent
Handles entity-based retrieval from Neo4j knowledge graph.
"""

from typing import List

from src.models import RetrievalResult, AgenticRAGState
from src.db.neo4j_client import Neo4jManager
from src.utils.ner import NERExtractor
from src.utils.language import detect_language


class GraphRetrievalAgent:
    """
    Agent responsible for graph-based retrieval from Neo4j.
    
    Tasks:
    1. Extract entities from user query
    2. Traverse Neo4j graph for related nodes
    3. Return chunk references from graph
    """
    
    def __init__(self):
        """Initialize the graph retrieval agent."""
        self.neo4j_manager = Neo4jManager()
        self.ner_extractor = NERExtractor()
    
    def search(self, query: str) -> List[RetrievalResult]:
        """
        Search for relevant chunks via graph traversal.
        
        Args:
            query: User query text
            
        Returns:
            List of RetrievalResult objects
        """
        if not query or not query.strip():
            return []
        
        # Detect language
        language = detect_language(query)
        
        # Extract entities from query
        entities = self.ner_extractor.extract_entities(
            text=query,
            doc_id="query",
            chunk_id="query",
            language=language
        )
        
        if not entities:
            # Try simple word matching for entity-like terms
            entity_names = self._extract_potential_entities(query)
        else:
            entity_names = [e.name for e in entities]
        
        if not entity_names:
            print("No entities found in query for graph search")
            return []
        
        print(f"Searching graph for entities: {entity_names}")
        
        # Get chunks associated with these entities
        results = self.neo4j_manager.get_chunks_for_entities(entity_names)
        
        # Also get related entities for richer context
        all_related_chunks = []
        for name in entity_names[:3]:  # Limit to first 3 entities
            related = self.neo4j_manager.find_related_entities(name, max_depth=2)
            for rel in related:
                if rel.get("doc_id") and rel.get("chunk_id"):
                    all_related_chunks.append(RetrievalResult(
                        doc_id=rel["doc_id"],
                        chunk_id=rel["chunk_id"],
                        text=f"Related entity: {rel['name']} ({rel['type']})",
                        score=0.8,  # Slightly lower score for related entities
                        source="neo4j",
                        metadata={
                            "entity_name": rel["name"],
                            "entity_type": rel["type"],
                            "relations": rel.get("relations", [])
                        }
                    ))
        
        results.extend(all_related_chunks)
        
        print(f"Graph search found {len(results)} results")
        return results
    
    def _extract_potential_entities(self, text: str) -> List[str]:
        """
        Extract potential entity names using simple heuristics.
        
        Args:
            text: Query text
            
        Returns:
            List of potential entity names
        """
        import re
        
        # Look for capitalized words (potential proper nouns)
        words = text.split()
        potential = []
        
        for word in words:
            # Skip common words
            if word.lower() in {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                               'what', 'who', 'where', 'when', 'how', 'why',
                               'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}:
                continue
            
            # Accept all non-stopword tokens as potential entities
            # This handles lowercase queries like "karachi"
            potential.append(word)
        
        return potential


# Agent instance
graph_retrieval_agent = GraphRetrievalAgent()


def run_graph_retrieval_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    LangGraph node function for graph retrieval.
    
    Args:
        state: Current RAG state with query
        
    Returns:
        Updated state with Neo4j results
    """
    agent = GraphRetrievalAgent()
    
    try:
        results = agent.search(state.query)
        state.neo4j_results = results
        
        # Extract entity names for reference
        state.query_entities = [
            r.metadata.get("entity_name", "") 
            for r in results 
            if r.metadata.get("entity_name")
        ]
        
    except Exception as e:
        state.errors.append(f"Error in graph retrieval: {e}")
    
    return state
