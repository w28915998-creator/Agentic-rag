"""
Knowledge Graph Agent
Handles NER, relationship extraction, and Neo4j graph construction.
"""

from typing import List, Tuple

from src.models import DocumentChunk, EntityNode, RelationshipEdge, IngestionState
from src.db.neo4j_client import Neo4jManager
from src.utils.ner import NERExtractor


class KnowledgeGraphAgent:
    """
    Agent responsible for building the knowledge graph in Neo4j.
    
    Tasks:
    1. Perform NER on document chunks
    2. Extract relationships between entities
    3. Deduplicate entities
    4. Create Neo4j nodes and edges
    """
    
    def __init__(self):
        """Initialize the knowledge graph agent."""
        self.neo4j_manager = Neo4jManager()
        self.ner_extractor = NERExtractor()
    
    def process_chunks(
        self,
        chunks: List[DocumentChunk]
    ) -> Tuple[List[EntityNode], List[RelationshipEdge]]:
        """
        Process chunks to extract entities and relationships.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Tuple of (entities, relationships)
        """
        all_entities = []
        all_relationships = []
        
        for chunk in chunks:
            # Extract entities
            entities = self.ner_extractor.extract_entities(
                text=chunk.text,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                language=chunk.language
            )
            
            # Extract relationships
            relationships = self.ner_extractor.extract_relationships(
                text=chunk.text,
                entities=entities,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id
            )
            
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        # Deduplicate entities globally
        deduplicated_entities = self._deduplicate_entities(all_entities)
        
        print(f"Extracted {len(deduplicated_entities)} unique entities and {len(all_relationships)} relationships")
        
        return deduplicated_entities, all_relationships
    
    def _deduplicate_entities(self, entities: List[EntityNode]) -> List[EntityNode]:
        """
        Deduplicate entities based on name and type.
        
        Args:
            entities: List of all extracted entities
            
        Returns:
            Deduplicated list of entities
        """
        seen = {}
        
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            
            if key not in seen:
                seen[key] = entity
            else:
                # Merge properties - keep the first occurrence but track all sources
                existing = seen[key]
                if 'source_chunks' not in existing.properties:
                    existing.properties['source_chunks'] = [existing.chunk_id]
                existing.properties['source_chunks'].append(entity.chunk_id)
        
        return list(seen.values())
    
    def build_graph(
        self,
        entities: List[EntityNode],
        relationships: List[RelationshipEdge]
    ) -> bool:
        """
        Build the knowledge graph in Neo4j.
        
        Args:
            entities: List of EntityNode objects
            relationships: List of RelationshipEdge objects
            
        Returns:
            True if successful
        """
        # Ensure constraints exist
        self.neo4j_manager.create_constraints()
        
        # Create entities
        entity_count = self.neo4j_manager.create_entities_batch(entities)
        
        # Create relationships
        rel_count = self.neo4j_manager.create_relationships_batch(relationships)
        
        print(f"Built knowledge graph: {entity_count} entities, {rel_count} relationships")
        
        return entity_count > 0 or rel_count > 0
    
    def get_graph_status(self):
        """Get the current status of the Neo4j graph."""
        return self.neo4j_manager.get_graph_stats()
    
    def clear_graph(self) -> bool:
        """Clear all graph data."""
        return self.neo4j_manager.clear_graph()


# Agent instance
knowledge_graph_agent = KnowledgeGraphAgent()


def run_knowledge_graph_agent(state: IngestionState) -> IngestionState:
    """
    LangGraph node function for knowledge graph construction.
    
    Args:
        state: Current ingestion state with chunks
        
    Returns:
        Updated state with graph status
    """
    agent = KnowledgeGraphAgent()
    
    try:
        # Extract entities and relationships
        entities, relationships = agent.process_chunks(state.chunks)
        state.entities = entities
        state.relationships = relationships
        
        # Build graph
        success = agent.build_graph(entities, relationships)
        state.indexed_to_neo4j = success
        
    except Exception as e:
        state.errors.append(f"Error building knowledge graph: {e}")
        state.indexed_to_neo4j = False
    
    return state
