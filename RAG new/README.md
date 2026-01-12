# Agentic RAG System

A multi-agent Retrieval Augmented Generation (RAG) system with hybrid retrieval using LangGraph, Qdrant, and Neo4j.

## Features

- **Multi-language Support**: English, Urdu, and mixed language documents
- **Hybrid Retrieval**: Combines semantic search (Qdrant) with knowledge graph (Neo4j)
- **Multi-agent Architecture**: 8 specialized agents orchestrated via LangGraph
- **Strict Grounding**: All answers include `(doc_id, chunk_id)` citations
- **Verification Layer**: Guardrails to prevent hallucination

## Architecture

```
Document Ingestion Pipeline:
┌─────────────┐     ┌──────────────────┐
│  Documents  │ ──> │ Ingestion Agent  │
└─────────────┘     └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────────┐    ┌─────────────────────┐
    │  Semantic Indexer   │    │  Knowledge Graph    │
    │      (Qdrant)       │    │     (Neo4j)         │
    └─────────────────────┘    └─────────────────────┘

Query Pipeline:
┌─────────┐
│  Query  │
└────┬────┘
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  │
┌──────────┐     ┌───────────┐            │
│ Semantic │     │   Graph   │            │
│ Retrieval│     │ Retrieval │            │
└────┬─────┘     └─────┬─────┘            │
     │                 │                  │
     └────────┬────────┘                  │
              ▼                           │
    ┌─────────────────┐                   │
    │ Hybrid Merge    │                   │
    └────────┬────────┘                   │
             ▼                            │
    ┌─────────────────┐                   │
    │ Answer Synthesis│ <─────────────────┘
    └────────┬────────┘   (Ollama qwen:30b)
             ▼
    ┌─────────────────┐
    │  Verification   │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │     Output      │
    └─────────────────┘
```

## Prerequisites

1. **Python 3.10+**
2. **Docker** (for Qdrant and Neo4j)
3. **Ollama** with `qwen:30b` model

## Setup

### 1. Start Docker Services

```bash
# Start Qdrant and Neo4j
docker-compose up -d
```

This starts:
- Qdrant on `localhost:6333`
- Neo4j on `localhost:7474` (HTTP) and `localhost:7687` (Bolt)

### 2. Install Ollama Model

```bash
# Pull the qwen:30b model
ollama pull qwen:30b
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 4. Configure Environment

```bash
# Copy example config
copy .env.example .env

# Edit .env with your Neo4j password
```

## Usage

### Ingest Documents

```bash
# Ingest a single file
python main.py ingest ./documents/sample.txt

# Ingest a directory
python main.py ingest ./documents/

# Clear existing data and ingest
python main.py ingest ./documents/ --clear
```

### Query the System

```bash
# Single query
python main.py query "What is the relationship between X and Y?"

# Urdu query
python main.py query "یہ دستاویز کس بارے میں ہے؟"
```

### Interactive Mode

```bash
python main.py interactive
```

### Check Status

```bash
python main.py status
```

### Clear All Data

```bash
python main.py clear
```

## Agents

| Agent | Purpose |
|-------|---------|
| Document Ingestion | Load, chunk, and embed documents |
| Semantic Indexer | Store embeddings in Qdrant |
| Knowledge Graph | Build entity graph in Neo4j |
| Semantic Retrieval | Vector similarity search |
| Graph Retrieval | Entity-based graph traversal |
| Hybrid Controller | Merge and rank results |
| Answer Synthesis | Generate grounded answers |
| Verification | Validate citations and quality |

## Configuration

Key settings in `.env`:

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen:30b

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Retrieval
TOP_K_RESULTS=5
CHUNK_SIZE=500
```

## Answer Format

### English

```
Answer:
<structured answer with inline citations>

Evidence:
- (doc_id: abc123, chunk_id: def456)
- (doc_id: xyz789, chunk_id: uvw012)
```

### Urdu

```
جواب:
<واضح اور مستند جواب>

حوالہ جات:
- (doc_id: abc123, chunk_id: def456)
```

## Project Structure

```
RAG new/
├── config/
│   └── settings.py          # Configuration
├── src/
│   ├── models/
│   │   └── schemas.py        # Pydantic models
│   ├── agents/
│   │   ├── ingestion.py      # Document processing
│   │   ├── semantic_indexer.py
│   │   ├── knowledge_graph.py
│   │   ├── semantic_retrieval.py
│   │   ├── graph_retrieval.py
│   │   ├── hybrid_controller.py
│   │   ├── answer_synthesis.py
│   │   └── verification.py
│   ├── utils/
│   │   ├── language.py       # Language detection
│   │   ├── embeddings.py     # Embedding generation
│   │   └── ner.py            # NER utilities
│   ├── db/
│   │   ├── qdrant_client.py
│   │   └── neo4j_client.py
│   └── orchestrator.py       # LangGraph workflows
├── main.py                   # CLI entry point
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## License

MIT
