"""
Document Ingestion Agent
Handles document loading, chunking, and embedding generation.
Supports English, Urdu, and mixed language documents.
"""

import os
import uuid
import re
from pathlib import Path
from typing import List, Optional, Generator
import chardet

from config import settings
from src.models import DocumentChunk, Language, IngestionState
from src.utils.language import detect_language, normalize_text
from src.utils.embeddings import EmbeddingGenerator


class DocumentIngestionAgent:
    """
    Agent responsible for document ingestion pipeline.
    
    Tasks:
    1. Load documents from various formats (TXT, PDF, DOCX)
    2. Detect language
    3. Normalize text encoding
    4. Chunk documents intelligently
    5. Generate embeddings
    6. Assign doc_id and chunk_id
    """
    
    def __init__(self):
        """Initialize the ingestion agent."""
        self.embedding_generator = EmbeddingGenerator()
        self.supported_extensions = {'.txt', '.pdf', '.docx', '.md'}
    
    def process_directory(self, directory_path: str) -> IngestionState:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            IngestionState with processed chunks
        """
        state = IngestionState()
        path = Path(directory_path)
        
        if not path.exists():
            state.errors.append(f"Directory not found: {directory_path}")
            return state
        
        # Find all supported files
        files = []
        for ext in self.supported_extensions:
            files.extend(path.glob(f"**/*{ext}"))
        
        state.raw_documents = [str(f) for f in files]
        print(f"Found {len(files)} documents to process")
        
        # Process each file
        for file_path in files:
            try:
                chunks = self.process_file(str(file_path))
                state.chunks.extend(chunks)
            except Exception as e:
                state.errors.append(f"Error processing {file_path}: {e}")
        
        print(f"Generated {len(state.chunks)} chunks from {len(files)} documents")
        return state
    
    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of DocumentChunk objects
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Load document content based on file type
        if extension == '.txt' or extension == '.md':
            content = self._load_text_file(file_path)
        elif extension == '.pdf':
            content = self._load_pdf_file(file_path)
        elif extension == '.docx':
            content = self._load_docx_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        if not content or not content.strip():
            print(f"Warning: Empty content from {file_path}")
            return []
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Detect language
        language = detect_language(content)
        print(f"Document {path.name}: Language detected as {language.value}")
        
        # Normalize text
        content = normalize_text(content, language)
        
        # Chunk the document
        chunks = self._chunk_text(
            text=content,
            doc_id=doc_id,
            source_file=file_path,
            language=language
        )
        
        from src.utils.ner import NERExtractor
        ner = NERExtractor()
        
        # Generate embeddings and extract temporal context
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_generator.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
            # Extract temporal context
            entities = ner.extract_entities(
                text=chunk.text,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                language=chunk.language
            )
            
            dates = [e.name for e in entities if e.entity_type == "DATE"]
            if dates:
                chunk.temporal_context["dates"] = dates
                # Extract years specifically for easier filtering
                years = []
                for date in dates:
                    # Match 4-digit years
                    import re
                    year_matches = re.findall(r'\b\d{4}\b', date)
                    years.extend([int(y) for y in year_matches])
                
                if years:
                    chunk.temporal_context["years"] = list(set(years))
        
        print(f"Processed {path.name}: {len(chunks)} chunks")
        return chunks
    
    def _load_text_file(self, file_path: str) -> str:
        """Load content from a text file with encoding detection."""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # Detect encoding
        detected = chardet.detect(raw_data)
        encoding = detected.get('encoding', 'utf-8')
        
        try:
            content = raw_data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # Fallback to utf-8
            content = raw_data.decode('utf-8', errors='ignore')
        
        return content
    
    def _load_pdf_file(self, file_path: str) -> str:
        """Load content from a PDF file."""
        try:
            from PyPDF2 import PdfReader
            
            reader = PdfReader(file_path)
            content_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content_parts.append(text)
            
            return '\n'.join(content_parts)
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
    
    def _load_docx_file(self, file_path: str) -> str:
        """Load content from a DOCX file."""
        try:
            from docx import Document
            
            doc = Document(file_path)
            content_parts = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    content_parts.append(para.text)
            
            return '\n'.join(content_parts)
        except ImportError:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")
    
    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        source_file: str,
        language: Language
    ) -> List[DocumentChunk]:
        """
        Chunk text into smaller segments.
        
        Uses sentence-aware chunking to avoid breaking mid-sentence.
        
        Args:
            text: Full document text
            doc_id: Document ID
            source_file: Source file path
            language: Document language
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Split into sentences
        sentences = self._split_sentences(text, language)
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if current_length + sentence_length > settings.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    language=language,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    metadata={"sentence_count": len(current_chunk)}
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                doc_id=doc_id,
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                language=language,
                source_file=source_file,
                chunk_index=chunk_index,
                metadata={"sentence_count": len(current_chunk)}
            ))
        
        return chunks
    
    def _split_sentences(self, text: str, language: Language) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            language: Text language
            
        Returns:
            List of sentences
        """
        # Universal sentence delimiters
        if language == Language.URDU:
            # Urdu sentence endings: ۔ (Urdu period), ؟ (question mark), ！ (exclamation)
            pattern = r'(?<=[۔؟!])\s+'
        else:
            # English sentence endings
            pattern = r'(?<=[.!?])\s+'
        
        sentences = re.split(pattern, text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        Get sentences for overlap from the end of the previous chunk.
        
        Args:
            sentences: List of sentences from previous chunk
            
        Returns:
            Overlap sentences
        """
        overlap_chars = settings.chunk_overlap
        overlap_sentences = []
        current_length = 0
        
        for sentence in reversed(sentences):
            if current_length + len(sentence) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                current_length += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def process_text(self, text: str, source_name: str = "direct_input") -> List[DocumentChunk]:
        """
        Process raw text directly (not from file).
        
        Args:
            text: Raw text to process
            source_name: Name to use as source
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        doc_id = str(uuid.uuid4())
        language = detect_language(text)
        text = normalize_text(text, language)
        
        chunks = self._chunk_text(
            text=text,
            doc_id=doc_id,
            source_file=source_name,
            language=language
        )
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_generator.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks


# Agent instance for use in LangGraph
document_ingestion_agent = DocumentIngestionAgent()


def run_ingestion_agent(state: IngestionState) -> IngestionState:
    """
    LangGraph node function for document ingestion.
    
    Args:
        state: Current ingestion state
        
    Returns:
        Updated state with processed chunks
    """
    agent = DocumentIngestionAgent()
    
    for doc_path in state.raw_documents:
        try:
            if os.path.isdir(doc_path):
                result = agent.process_directory(doc_path)
                state.chunks.extend(result.chunks)
                state.errors.extend(result.errors)
            elif os.path.isfile(doc_path):
                chunks = agent.process_file(doc_path)
                state.chunks.extend(chunks)
        except Exception as e:
            state.errors.append(f"Error processing {doc_path}: {e}")
    
    return state
