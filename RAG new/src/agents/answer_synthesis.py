"""
Answer Synthesis Agent
Generates grounded answers from retrieved evidence using Ollama LLM.
"""

from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.models import (
    RetrievalResult, 
    AnswerResponse, 
    Citation, 
    Language,
    AgenticRAGState
)


# Failure messages for different languages
FAILURE_MESSAGES = {
    Language.ENGLISH: "The requested information is not found in the provided documents.",
    Language.URDU: "فراہم کردہ دستاویزات میں مطلوبہ معلومات موجود نہیں ہیں۔",
    Language.MIXED: "The requested information is not found in the provided documents. / فراہم کردہ دستاویزات میں مطلوبہ معلومات موجود نہیں ہیں۔"
}


class AnswerSynthesisAgent:
    """
    Agent responsible for synthesizing answers from evidence.
    
    Tasks:
    1. Compose coherent answer from evidence chunks
    2. Include inline citations (doc_id, chunk_id)
    3. Maintain query language
    4. Return failure message if no evidence
    """
    
    def __init__(self):
        """Initialize the answer synthesis agent."""
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,  # Low temperature for factual responses
        )
    
    def synthesize(
        self,
        query: str,
        evidence: List[RetrievalResult],
        language: Language = Language.ENGLISH
    ) -> AnswerResponse:
        """
        Synthesize an answer from evidence.
        
        Args:
            query: User query
            evidence: Retrieved evidence chunks
            language: Response language
            
        Returns:
            AnswerResponse with answer and citations
        """
        # Check for empty evidence
        if not evidence:
            return AnswerResponse(
                answer=FAILURE_MESSAGES.get(language, FAILURE_MESSAGES[Language.ENGLISH]),
                citations=[],
                language=language,
                verified=False
            )
        
        # Build evidence context
        evidence_context = self._format_evidence(evidence)
        
        # Create system prompt
        system_prompt = self._get_system_prompt(language)
        
        # Create user prompt
        user_prompt = self._get_user_prompt(query, evidence_context, language)
        
        try:
            # Call LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            answer_text = response.content
            
            # Extract citations from evidence
            citations = [
                Citation(doc_id=e.doc_id, chunk_id=e.chunk_id)
                for e in evidence
            ]
            
            return AnswerResponse(
                answer=answer_text,
                citations=citations,
                language=language,
                verified=False  # Will be set by verification agent
            )
            
        except Exception as e:
            print(f"Error in answer synthesis: {e}")
            return AnswerResponse(
                answer=f"Error generating answer: {str(e)}",
                citations=[],
                language=language,
                verified=False
            )
    
    def _format_evidence(self, evidence: List[RetrievalResult]) -> str:
        """Format evidence chunks for the prompt."""
        formatted = []
        
        for i, e in enumerate(evidence, 1):
            formatted.append(f"""
Evidence {i}:
- doc_id: {e.doc_id}
- chunk_id: {e.chunk_id}
- score: {e.score:.3f}
- source: {e.source}
- text: {e.text}
""")
        
        return "\n".join(formatted)
    
    def _get_system_prompt(self, language: Language) -> str:
        """Get the system prompt for answer synthesis."""
        
        if language == Language.URDU:
            return """آپ ایک RAG سسٹم ہیں جو صرف فراہم کردہ دستاویزات سے جواب دیتا ہے۔

اصول:
1. صرف فراہم کردہ شواہد سے جواب دیں
2. ہر دعوے کے ساتھ (doc_id, chunk_id) کا حوالہ دیں
3. اگر معلومات موجود نہ ہوں تو واضح طور پر بتائیں
4. کبھی اپنے علم سے جواب نہ دیں
5. جواب اردو میں دیں"""

        elif language == Language.MIXED:
            return """You are a RAG system that answers ONLY from provided documents.
آپ ایک RAG سسٹم ہیں جو صرف فراہم کردہ دستاویزات سے جواب دیتا ہے۔

Rules:
1. Answer ONLY from provided evidence
2. Include (doc_id, chunk_id) citations for every claim
3. If information is not found, clearly state so
4. NEVER use internal knowledge
5. Match the language style of the query"""

        else:  # English
            return """You are a RAG (Retrieval Augmented Generation) system that answers ONLY from provided documents.

STRICT RULES:
1. Answer ONLY from the provided evidence chunks
2. Include (doc_id, chunk_id) citations for EVERY claim you make
3. If the information is not found in the evidence, clearly state: "The requested information is not found in the provided documents."
4. NEVER use your internal knowledge to answer
5. NEVER hallucinate or make up information
6. Combine information from multiple evidence chunks when relevant
7. Maintain a coherent, well-structured response

Citation format: (doc_id: X, chunk_id: Y)"""

    def _get_user_prompt(
        self, 
        query: str, 
        evidence_context: str,
        language: Language
    ) -> str:
        """Get the user prompt for answer synthesis."""
        
        if language == Language.URDU:
            return f"""سوال: {query}

شواہد:
{evidence_context}

مندرجہ بالا شواہد کی بنیاد پر جواب دیں۔ ہر دعوے کے ساتھ حوالہ شامل کریں۔"""

        elif language == Language.MIXED:
            return f"""Question/سوال: {query}

Evidence/شواہد:
{evidence_context}

Answer based on the evidence above. Include citations for each claim.
مندرجہ بالا شواہد کی بنیاد پر جواب دیں۔"""

        else:  # English
            return f"""Question: {query}

Evidence:
{evidence_context}

Based on the evidence above, provide a comprehensive answer to the question.
Include (doc_id, chunk_id) citations for each claim you make.
If the evidence does not contain information to answer the question, say so clearly."""


# Agent instance
answer_synthesis_agent = AnswerSynthesisAgent()


def run_answer_synthesis_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    LangGraph node function for answer synthesis.
    
    Args:
        state: Current RAG state with merged evidence
        
    Returns:
        Updated state with synthesized answer
    """
    agent = AnswerSynthesisAgent()
    
    try:
        response = agent.synthesize(
            query=state.query,
            evidence=state.merged_evidence,
            language=state.query_language
        )
        
        state.synthesized_answer = response.answer
        state.citations = response.citations
        
    except Exception as e:
        state.errors.append(f"Error in answer synthesis: {e}")
        state.synthesized_answer = FAILURE_MESSAGES.get(
            state.query_language, 
            FAILURE_MESSAGES[Language.ENGLISH]
        )
    
    return state
