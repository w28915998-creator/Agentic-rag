"""
Neo4j graph database client for storing and retrieving knowledge graph data.
"""

from typing import List, Optional, Dict, Any, Tuple
from neo4j import GraphDatabase, Driver
from contextlib import contextmanager

from config import settings
from src.models import EntityNode, RelationshipEdge, RetrievalResult


class Neo4jManager:
    """
    Manages Neo4j graph database operations.
    Handles node/edge creation and graph traversal queries.
    """
    
    _instance: Optional['Neo4jManager'] = None
    _driver: Optional[Driver] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Neo4j driver."""
        if self._driver is None:
            self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j server."""
        print(f"Connecting to Neo4j at {settings.neo4j_uri}")
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        # Verify connection
        try:
            self._driver.verify_connectivity()
            print("Connected to Neo4j successfully")
        except Exception as e:
            print(f"Warning: Could not verify Neo4j connection: {e}")
    
    @property
    def driver(self) -> Driver:
        """Get the Neo4j driver."""
        return self._driver
    
    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions."""
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def create_constraints(self):
        """Create unique constraints and indexes for the graph."""
        with self.session() as session:
            # Unique constraint on Entity id
            try:
                session.run("""
                    CREATE CONSTRAINT entity_id IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
            except Exception as e:
                print(f"Note: Constraint creation: {e}")
            
            # Index on entity name for faster lookups
            try:
                session.run("""
                    CREATE INDEX entity_name IF NOT EXISTS
                    FOR (e:Entity) ON (e.name)
                """)
            except Exception as e:
                print(f"Note: Index creation: {e}")
            
            # Index on entity type
            try:
                session.run("""
                    CREATE INDEX entity_type IF NOT EXISTS
                    FOR (e:Entity) ON (e.entity_type)
                """)
            except Exception as e:
                print(f"Note: Index creation: {e}")
        
        print("Neo4j constraints and indexes ensured")
    
    def create_entity(self, entity: EntityNode) -> bool:
        """
        Create or merge an entity node.
        
        Args:
            entity: EntityNode to create
            
        Returns:
            True if successful
        """
        query = """
            MERGE (e:Entity {name: $name, entity_type: $entity_type})
            ON CREATE SET
                e.id = $id,
                e.source_doc_id = $source_doc_id,
                e.chunk_id = $chunk_id,
                e.language = $language,
                e.properties = $properties,
                e.created_at = datetime()
            ON MATCH SET
                e.updated_at = datetime()
            RETURN e
        """
        
        try:
            with self.session() as session:
                session.run(
                    query,
                    id=entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    source_doc_id=entity.source_doc_id,
                    chunk_id=entity.chunk_id,
                    language=entity.language.value,
                    properties=str(entity.properties)
                )
            return True
        except Exception as e:
            print(f"Error creating entity: {e}")
            return False
    
    def create_entities_batch(self, entities: List[EntityNode]) -> int:
        """
        Create multiple entities in batch.
        
        Args:
            entities: List of EntityNode objects
            
        Returns:
            Number of entities created
        """
        if not entities:
            return 0
            
        # Group entities by type to apply specific labels
        from collections import defaultdict
        entities_by_type = defaultdict(list)
        for e in entities:
            entities_by_type[e.entity_type].append(e)
            
        total_created = 0
        
        try:
            with self.session() as session:
                for entity_type, type_entities in entities_by_type.items():
                    # Sanitize label (alphanumeric only)
                    safe_label = "".join([c for c in entity_type if c.isalnum()])
                    if not safe_label:
                        safe_label = "Unknown"
                        
                    query = f"""
                        UNWIND $entities AS entity
                        MERGE (e:Entity:{safe_label} {{name: entity.name}})
                        ON CREATE SET
                            e.id = entity.id,
                            e.entity_type = entity.entity_type, 
                            e.source_doc_id = entity.source_doc_id,
                            e.chunk_id = entity.chunk_id,
                            e.language = entity.language,
                            e.properties = entity.properties,
                            e.created_at = datetime()
                        ON MATCH SET
                            e.updated_at = datetime()
                        RETURN count(e) as count
                    """
                    
                    entity_data = [
                        {
                            "id": e.id,
                            "name": e.name,
                            "entity_type": e.entity_type,
                            "source_doc_id": e.source_doc_id,
                            "chunk_id": e.chunk_id,
                            "language": e.language.value,
                            "properties": str(e.properties)
                        }
                        for e in type_entities
                    ]
                    
                    result = session.run(query, entities=entity_data)
                    record = result.single()
                    count = record["count"] if record else 0
                    total_created += count
                    
                print(f"Created/updated {total_created} entities with labels in Neo4j")
                return total_created
                
        except Exception as e:
            print(f"Error creating entities batch: {e}")
            return 0
    
    def create_relationship(self, relationship: RelationshipEdge) -> bool:
        """
        Create a relationship between entities.
        
        Args:
            relationship: RelationshipEdge to create
            
        Returns:
            True if successful
        """
        # Dynamic relationship type - using parameterized approach
        query = """
            MATCH (from:Entity {name: $from_entity})
            MATCH (to:Entity {name: $to_entity})
            MERGE (from)-[r:RELATED {relation_type: $relation_type}]->(to)
            ON CREATE SET
                r.source_doc_id = $source_doc_id,
                r.chunk_id = $chunk_id,
                r.properties = $properties,
                r.created_at = datetime()
            RETURN r
        """
        
        try:
            with self.session() as session:
                session.run(
                    query,
                    from_entity=relationship.from_entity,
                    to_entity=relationship.to_entity,
                    relation_type=relationship.relation_type,
                    source_doc_id=relationship.source_doc_id,
                    chunk_id=relationship.chunk_id,
                    properties=str(relationship.properties)
                )
            return True
        except Exception as e:
            print(f"Error creating relationship: {e}")
            return False
    
    def create_relationships_batch(self, relationships: List[RelationshipEdge]) -> int:
        """
        Create multiple relationships in batch.
        
        Args:
            relationships: List of RelationshipEdge objects
            
        Returns:
            Number of relationships created
        """
        if not relationships:
            return 0
            
        # Group by relationship type
        from collections import defaultdict
        rels_by_type = defaultdict(list)
        for r in relationships:
            rels_by_type[r.relation_type].append(r)
            
        total_created = 0
        
        try:
            with self.session() as session:
                for rel_type, type_rels in rels_by_type.items():
                    # Sanitize (uppercase, alpha + underscore)
                    safe_type = "".join([c for c in rel_type.upper() if c.isalnum() or c == '_'])
                    if not safe_type:
                        safe_type = "RELATED"
                        
                    query = f"""
                        UNWIND $rels AS rel
                        MATCH (from:Entity {{name: rel.from_entity}})
                        MATCH (to:Entity {{name: rel.to_entity}})
                        MERGE (from)-[r:{safe_type} {{relation_type: rel.relation_type}}]->(to)
                        ON CREATE SET
                            r.source_doc_id = rel.source_doc_id,
                            r.chunk_id = rel.chunk_id,
                            r.properties = rel.properties,
                            r.created_at = datetime()
                        RETURN count(r) as count
                    """
                    
                    rel_data = [
                        {
                            "from_entity": r.from_entity,
                            "to_entity": r.to_entity,
                            "relation_type": r.relation_type,
                            "source_doc_id": r.source_doc_id,
                            "chunk_id": r.chunk_id,
                            "properties": str(r.properties)
                        }
                        for r in type_rels
                    ]
                    
                    result = session.run(query, rels=rel_data)
                    record = result.single()
                    count = record["count"] if record else 0
                    total_created += count
                    
                print(f"Created {total_created} relationships with specific types in Neo4j")
                return total_created
        except Exception as e:
            print(f"Error creating relationships batch: {e}")
            return 0
    
    def find_entities_by_name(self, name: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """
        Find entities by name.
        
        Args:
            name: Entity name to search
            fuzzy: Whether to use fuzzy matching
            
        Returns:
            List of matching entities
        """
        if fuzzy:
            query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($name)
                RETURN e.id as id, e.name as name, e.entity_type as type,
                       e.source_doc_id as doc_id, e.chunk_id as chunk_id
                LIMIT 10
            """
        else:
            query = """
                MATCH (e:Entity {name: $name})
                RETURN e.id as id, e.name as name, e.entity_type as type,
                       e.source_doc_id as doc_id, e.chunk_id as chunk_id
            """
        
        try:
            with self.session() as session:
                result = session.run(query, name=name)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error finding entities: {e}")
            return []
    
    def find_related_entities(
        self,
        entity_name: str,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity.
        
        Args:
            entity_name: Starting entity name
            max_depth: Maximum traversal depth
            
        Returns:
            List of related entities with relationship info
        """
        query = """
            MATCH (start:Entity)
            WHERE toLower(start.name) CONTAINS toLower($name)
            MATCH path = (start)-[r:RELATED*1..%d]-(related:Entity)
            RETURN DISTINCT
                related.name as name,
                related.entity_type as type,
                related.source_doc_id as doc_id,
                related.chunk_id as chunk_id,
                [rel in r | rel.relation_type] as relations,
                length(path) as distance
            ORDER BY distance
            LIMIT 20
        """ % max_depth
        
        try:
            with self.session() as session:
                result = session.run(query, name=entity_name)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error finding related entities: {e}")
            return []
    
    def get_chunks_for_entities(self, entity_names: List[str]) -> List[RetrievalResult]:
        """
        Get chunk references for given entities.
        
        Args:
            entity_names: List of entity names
            
        Returns:
            List of RetrievalResult with chunk references
        """
        query = """
            UNWIND $names AS name
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower(name)
            RETURN DISTINCT 
                e.source_doc_id as doc_id,
                e.chunk_id as chunk_id,
                e.name as entity_name,
                e.entity_type as entity_type
        """
        
        try:
            with self.session() as session:
                result = session.run(query, names=entity_names)
                results = []
                for record in result:
                    results.append(RetrievalResult(
                        doc_id=record["doc_id"],
                        chunk_id=record["chunk_id"],
                        text=f"Entity: {record['entity_name']} ({record['entity_type']})",
                        score=1.0,  # Graph matches are considered highly relevant
                        source="neo4j",
                        metadata={
                            "entity_name": record["entity_name"],
                            "entity_type": record["entity_type"]
                        }
                    ))
                return results
        except Exception as e:
            print(f"Error getting chunks for entities: {e}")
            return []
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        try:
            with self.session() as session:
                # Count entities
                entity_result = session.run("MATCH (e:Entity) RETURN count(e) as count")
                entity_count = entity_result.single()["count"]
                
                # Count relationships
                rel_result = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]
                
                # Count by entity type
                type_result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.entity_type as type, count(e) as count
                    ORDER BY count DESC
                """)
                type_counts = {r["type"]: r["count"] for r in type_result}
                
                return {
                    "total_entities": entity_count,
                    "total_relationships": rel_count,
                    "entities_by_type": type_counts
                }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_graph(self) -> bool:
        """Delete all nodes and relationships."""
        try:
            with self.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("Cleared all data from Neo4j")
            return True
        except Exception as e:
            print(f"Error clearing graph: {e}")
            return False
    
    def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None


# Global instance
neo4j_manager = Neo4jManager()
