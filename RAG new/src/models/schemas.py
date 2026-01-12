"""
Pydantic data models for the Agentic RAG system.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    URDU = "ur"
    MIXED = "mixed"


class DocumentChunk(BaseModel):
    """Represents a chunk of a document with its embedding."""
    
    doc_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document identifier"
    )
    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique chunk identifier"
    )
    text: str = Field(
        description="Chunk text content"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding of the chunk"
    )
    language: Language = Field(
        default=Language.ENGLISH,
        description="Detected language of the chunk"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Source file path"
    )
    chunk_index: int = Field(
        default=0,
        description="Index of chunk within document"
    )
    temporal_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Temporal metadata (year, date, etc.)"
    )


class EntityNode(BaseModel):
    """Represents an entity node in Neo4j."""
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique entity identifier"
    )
    entity_type: str = Field(
        description="Entity type (Person, Organization, Location, etc.)"
    )
    name: str = Field(
        description="Entity name"
    )
    source_doc_id: str = Field(
        description="Source document ID"
    )
    chunk_id: str = Field(
        description="Source chunk ID"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional entity properties"
    )
    language: Language = Field(
        default=Language.ENGLISH,
        description="Language of the entity"
    )


class RelationshipEdge(BaseModel):
    """Represents a relationship edge in Neo4j."""
    
    from_entity: str = Field(
        description="Source entity ID or name"
    )
    to_entity: str = Field(
        description="Target entity ID or name"
    )
    relation_type: str = Field(
        description="Type of relationship"
    )
    source_doc_id: str = Field(
        description="Source document ID"
    )
    chunk_id: str = Field(
        description="Source chunk ID"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional relationship properties"
    )


class RetrievalResult(BaseModel):
    """Represents a retrieval result from Qdrant or Neo4j."""
    
    doc_id: str = Field(
        description="Document ID"
    )
    chunk_id: str = Field(
        description="Chunk ID"
    )
    text: str = Field(
        description="Retrieved text content"
    )
    score: float = Field(
        default=0.0,
        description="Relevance score"
    )
    source: str = Field(
        default="qdrant",
        description="Retrieval source: 'qdrant' or 'neo4j'"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class Citation(BaseModel):
    """Represents a citation reference."""
    
    doc_id: str = Field(
        description="Document ID"
    )
    chunk_id: str = Field(
        description="Chunk ID"
    )


class AnswerResponse(BaseModel):
    """Represents the final answer with citations."""
    
    answer: str = Field(
        description="Synthesized answer text"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of citations"
    )
    language: Language = Field(
        default=Language.ENGLISH,
        description="Response language"
    )
    verified: bool = Field(
        default=False,
        description="Whether the answer passed verification"
    )


class IngestionState(BaseModel):
    """State for document ingestion pipeline."""
    
    raw_documents: List[str] = Field(
        default_factory=list,
        description="Raw document paths to process"
    )
    chunks: List[DocumentChunk] = Field(
        default_factory=list,
        description="Processed document chunks"
    )
    entities: List[EntityNode] = Field(
        default_factory=list,
        description="Extracted entities"
    )
    relationships: List[RelationshipEdge] = Field(
        default_factory=list,
        description="Extracted relationships"
    )
    indexed_to_qdrant: bool = Field(
        default=False,
        description="Whether chunks are indexed to Qdrant"
    )
    indexed_to_neo4j: bool = Field(
        default=False,
        description="Whether graph is built in Neo4j"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors during processing"
    )


class AgenticRAGState(BaseModel):
    """State for the query/retrieval pipeline."""
    
    query: str = Field(
        default="",
        description="User query"
    )
    query_language: Language = Field(
        default=Language.ENGLISH,
        description="Detected query language"
    )
    query_entities: List[str] = Field(
        default_factory=list,
        description="Entities extracted from query"
    )
    qdrant_results: List[RetrievalResult] = Field(
        default_factory=list,
        description="Results from Qdrant semantic search"
    )
    neo4j_results: List[RetrievalResult] = Field(
        default_factory=list,
        description="Results from Neo4j graph traversal"
    )
    merged_evidence: List[RetrievalResult] = Field(
        default_factory=list,
        description="Merged and deduplicated evidence"
    )
    synthesized_answer: str = Field(
        default="",
        description="Generated answer"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Answer citations"
    )
    verified: bool = Field(
        default=False,
        description="Whether answer passed verification"
    )
    final_output: Optional[AnswerResponse] = Field(
        default=None,
        description="Final verified answer"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors during processing"
    )
