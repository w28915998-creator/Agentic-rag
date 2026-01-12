"""
Configuration settings for the Agentic RAG system.
Uses Pydantic Settings for environment-based configuration.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    ollama_model: str = Field(
        default="gpt-oss:20b",
        description="Ollama model to use for answer synthesis"
    )
    
    # Qdrant Configuration
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server host"
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant server port"
    )
    qdrant_collection: str = Field(
        default="rag_test",
        description="Qdrant collection name"
    )
    
    # Neo4j Configuration
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI"
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    neo4j_password: str = Field(
        default="testpassword",
        description="Neo4j password"
    )
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Sentence transformer model for embeddings"
    )
    embedding_dimension: int = Field(
        default=768,
        description="Embedding vector dimension"
    )
    
    # Chunking Configuration
    chunk_size: int = Field(
        default=500,
        description="Maximum chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks"
    )
    
    # Retrieval Configuration
    top_k_results: int = Field(
        default=5,
        description="Number of top results to retrieve"
    )
    
    # Language Settings
    supported_languages: list = Field(
        default=["en", "ur", "mixed"],
        description="Supported languages"
    )


# Global settings instance
settings = Settings()
