"""
LangGraph Orchestrator
Defines the workflow graphs for document ingestion and query processing.
"""

from typing import Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.models import IngestionState, AgenticRAGState

# Import agent functions
from src.agents.ingestion import run_ingestion_agent
from src.agents.semantic_indexer import run_semantic_indexer_agent
from src.agents.knowledge_graph import run_knowledge_graph_agent
from src.agents.semantic_retrieval import run_semantic_retrieval_agent
from src.agents.graph_retrieval import run_graph_retrieval_agent
from src.agents.hybrid_controller import run_hybrid_controller
from src.agents.answer_synthesis import run_answer_synthesis_agent
from src.agents.verification import run_verification_agent


# ============================================================================
# INGESTION GRAPH
# ============================================================================

def create_ingestion_graph() -> StateGraph:
    """
    Create the document ingestion workflow graph.
    
    Flow:
    Document Ingestion → Semantic Indexer (Qdrant)
                      └→ Knowledge Graph (Neo4j)
    
    Returns:
        Compiled StateGraph for ingestion
    """
    # Create graph with IngestionState
    graph = StateGraph(IngestionState)
    
    # Add nodes
    graph.add_node("ingest", run_ingestion_agent)
    graph.add_node("index_qdrant", run_semantic_indexer_agent)
    graph.add_node("build_graph", run_knowledge_graph_agent)
    
    # Define edges - sequential flow to avoid state merge conflicts
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "index_qdrant")
    graph.add_edge("index_qdrant", "build_graph")
    graph.add_edge("build_graph", END)
    
    return graph.compile()


# ============================================================================
# QUERY GRAPH
# ============================================================================

def create_query_graph() -> StateGraph:
    """
    Create the query processing workflow graph.
    
    Flow:
    Query → Semantic Retrieval (Qdrant) ─┐
         └→ Graph Retrieval (Neo4j) ─────┤
                                         ↓
                               Hybrid Controller
                                         ↓
                               Answer Synthesis
                                         ↓
                                   Verification
                                         ↓
                                       Output
    
    Returns:
        Compiled StateGraph for query processing
    """
    # Create graph with AgenticRAGState
    graph = StateGraph(AgenticRAGState)
    
    # Add nodes
    graph.add_node("semantic_search", run_semantic_retrieval_agent)
    graph.add_node("graph_search", run_graph_retrieval_agent)
    graph.add_node("merge", run_hybrid_controller)
    graph.add_node("synthesize", run_answer_synthesis_agent)
    graph.add_node("verify", run_verification_agent)
    
    # Define entry points (parallel retrieval)
    graph.set_entry_point("semantic_search")
    
    # Semantic search goes to merge
    graph.add_edge("semantic_search", "graph_search")
    
    # Graph search goes to merge
    graph.add_edge("graph_search", "merge")
    
    # Merge goes to synthesize
    graph.add_edge("merge", "synthesize")
    
    # Synthesize goes to verify
    graph.add_edge("synthesize", "verify")
    
    # Verify is the end
    graph.add_edge("verify", END)
    
    return graph.compile()


# ============================================================================
# WORKFLOW RUNNERS
# ============================================================================

class RAGOrchestrator:
    """
    Main orchestrator for the Agentic RAG system.
    Manages both ingestion and query workflows.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        print("Initializing RAG Orchestrator...")
        self.ingestion_graph = create_ingestion_graph()
        self.query_graph = create_query_graph()
        print("RAG Orchestrator ready")
    
    def ingest_documents(self, document_paths: list) -> IngestionState:
        """
        Run the document ingestion pipeline.
        
        Args:
            document_paths: List of file or directory paths
            
        Returns:
            Final IngestionState with results
        """
        print(f"\n{'='*60}")
        print("STARTING DOCUMENT INGESTION")
        print(f"{'='*60}")
        print(f"Documents: {document_paths}")
        
        # Create initial state
        initial_state = IngestionState(raw_documents=document_paths)
        
        # Run the graph
        final_state = self.ingestion_graph.invoke(initial_state)
        
        # Convert to IngestionState if needed
        if isinstance(final_state, dict):
            final_state = IngestionState(**final_state)
        
        print(f"\n{'='*60}")
        print("INGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"Chunks processed: {len(final_state.chunks)}")
        print(f"Entities extracted: {len(final_state.entities)}")
        print(f"Relationships found: {len(final_state.relationships)}")
        print(f"Indexed to Qdrant: {final_state.indexed_to_qdrant}")
        print(f"Indexed to Neo4j: {final_state.indexed_to_neo4j}")
        if final_state.errors:
            print(f"Errors: {final_state.errors}")
        
        return final_state
    
    def query(self, query_text: str) -> AgenticRAGState:
        """
        Run the query pipeline.
        
        Args:
            query_text: User query
            
        Returns:
            Final AgenticRAGState with answer
        """
        print(f"\n{'='*60}")
        print("PROCESSING QUERY")
        print(f"{'='*60}")
        print(f"Query: {query_text}")
        
        # Create initial state
        initial_state = AgenticRAGState(query=query_text)
        
        # Run the graph
        final_state = self.query_graph.invoke(initial_state)
        
        # Convert to AgenticRAGState if needed
        if isinstance(final_state, dict):
            final_state = AgenticRAGState(**final_state)
        
        print(f"\n{'='*60}")
        print("QUERY COMPLETE")
        print(f"{'='*60}")
        print(f"Language: {final_state.query_language.value}")
        print(f"Qdrant results: {len(final_state.qdrant_results)}")
        print(f"Neo4j results: {len(final_state.neo4j_results)}")
        print(f"Merged evidence: {len(final_state.merged_evidence)}")
        print(f"Verified: {final_state.verified}")
        if final_state.errors:
            print(f"Errors: {final_state.errors}")
        
        return final_state
    
    def format_answer(self, state: AgenticRAGState) -> str:
        """
        Format the answer for display.
        
        Args:
            state: Final query state
            
        Returns:
            Formatted answer string
        """
        if state.final_output:
            output = state.final_output
        else:
            from src.models import AnswerResponse, Language
            output = AnswerResponse(
                answer=state.synthesized_answer,
                citations=state.citations,
                language=state.query_language,
                verified=state.verified
            )
        
        # Format based on language
        if output.language.value == "ur":
            result = f"""
جواب:
{output.answer}

حوالہ جات:
"""
            # Create lookup map for source info
            evidence_map = {r.chunk_id: r for r in state.merged_evidence}
            
            for c in output.citations:
                source_tag = ""
                if c.chunk_id in evidence_map:
                    ev = evidence_map[c.chunk_id]
                    if ev.metadata.get("graph_validated"):
                        source_tag = " [Graph]"
                    else:
                        source_tag = " [Semantic]"
                        
                result += f"- {source_tag} (doc_id: {c.doc_id}, chunk_id: {c.chunk_id})\n"
        else:
            result = f"""
Answer:
{output.answer}

Evidence:
"""
            # Create lookup map for source info
            evidence_map = {r.chunk_id: r for r in state.merged_evidence}
            
            for c in output.citations:
                source_tag = ""
                if c.chunk_id in evidence_map:
                    ev = evidence_map[c.chunk_id]
                    if ev.metadata.get("graph_validated"):
                        source_tag = " [Graph]"
                    else:
                        source_tag = " [Semantic]"
                        
                result += f"- {source_tag} (doc_id: {c.doc_id}, chunk_id: {c.chunk_id})\n"
        
        if not output.verified:
            result += "\n⚠️ Warning: This answer has not been fully verified."
        
        return result


# Global orchestrator instance (lazy initialization)
_orchestrator = None


def get_orchestrator() -> RAGOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator
