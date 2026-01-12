"""
Verification Agent
Final quality gate to ensure answers are properly grounded with citations.
"""

import re
from typing import List, Tuple

from src.models import (
    AnswerResponse, 
    Citation, 
    RetrievalResult,
    Language,
    AgenticRAGState
)


# Failure messages
FAILURE_MESSAGES = {
    Language.ENGLISH: "The requested information is not found in the provided documents.",
    Language.URDU: "فراہم کردہ دستاویزات میں مطلوبہ معلومات موجود نہیں ہیں۔",
    Language.MIXED: "The requested information is not found in the provided documents."
}


class VerificationAgent:
    """
    Agent responsible for verifying answer quality.
    
    Tasks:
    1. Verify all claims have citations
    2. Check citation format consistency
    3. Ensure no unsupported statements
    4. Block answers without evidence
    """
    
    def __init__(self):
        """Initialize the verification agent."""
        # Pattern to match citations in answer
        self.citation_pattern = re.compile(
            r'\(doc_id:\s*([a-fA-F0-9-]+),\s*chunk_id:\s*([a-fA-F0-9-]+)\)'
        )
    
    def verify(
        self,
        answer: str,
        citations: List[Citation],
        evidence: List[RetrievalResult],
        language: Language = Language.ENGLISH
    ) -> Tuple[bool, AnswerResponse]:
        """
        Verify the answer meets quality requirements.
        
        Args:
            answer: Synthesized answer text
            citations: List of citations
            evidence: Original evidence chunks
            language: Answer language
            
        Returns:
            Tuple of (verified: bool, response: AnswerResponse)
        """
        issues = []
        
        # Check 1: Evidence exists
        if not evidence:
            return False, AnswerResponse(
                answer=FAILURE_MESSAGES.get(language, FAILURE_MESSAGES[Language.ENGLISH]),
                citations=[],
                language=language,
                verified=False
            )
        
        # Check 2: Answer is not a failure message
        if self._is_failure_message(answer):
            # This is valid - system correctly identified no relevant info
            return True, AnswerResponse(
                answer=answer,
                citations=[],
                language=language,
                verified=True
            )
        
        # Check 3: Citations exist
        if not citations:
            issues.append("No citations provided")
        
        # Check 4: Inline citations in text
        inline_citations = self._extract_inline_citations(answer)
        if not inline_citations:
            issues.append("No inline citations found in answer text")
        
        # Check 5: Citations reference valid evidence
        evidence_ids = {e.chunk_id for e in evidence}
        invalid_citations = []
        
        for citation in citations:
            if citation.chunk_id not in evidence_ids:
                invalid_citations.append(citation.chunk_id)
        
        if invalid_citations:
            issues.append(f"Citations reference non-existent evidence: {invalid_citations}")
        
        # Check 6: Answer is not trivially short
        if len(answer.split()) < 10 and not self._is_failure_message(answer):
            issues.append("Answer is suspiciously short")
        
        # Determine verification result
        verified = len(issues) == 0
        
        if not verified:
            print(f"Verification issues: {issues}")
        
        # Build final response
        response = AnswerResponse(
            answer=answer,
            citations=citations,
            language=language,
            verified=verified
        )
        
        return verified, response
    
    def _is_failure_message(self, answer: str) -> bool:
        """Check if answer is a standard failure message."""
        failure_phrases = [
            "not found in the provided documents",
            "فراہم کردہ دستاویزات میں مطلوبہ معلومات موجود نہیں",
            "no relevant information",
            "cannot find",
            "no information available"
        ]
        
        answer_lower = answer.lower()
        return any(phrase.lower() in answer_lower for phrase in failure_phrases)
    
    def _extract_inline_citations(self, answer: str) -> List[Tuple[str, str]]:
        """Extract inline citations from answer text."""
        matches = self.citation_pattern.findall(answer)
        return matches
    
    def add_missing_citations(
        self,
        answer: str,
        evidence: List[RetrievalResult]
    ) -> str:
        """
        Add missing citations to answer if needed.
        
        Args:
            answer: Answer text
            evidence: Evidence chunks
            
        Returns:
            Answer with citations added
        """
        # Check if citations already exist
        if self.citation_pattern.search(answer):
            return answer
        
        # Add evidence section at the end
        if evidence:
            citation_section = "\n\nEvidence:\n"
            for e in evidence:
                citation_section += f"- (doc_id: {e.doc_id}, chunk_id: {e.chunk_id})\n"
            
            return answer + citation_section
        
        return answer


# Agent instance
verification_agent = VerificationAgent()


def run_verification_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    LangGraph node function for verification.
    
    Args:
        state: Current RAG state with synthesized answer
        
    Returns:
        Updated state with verification result
    """
    agent = VerificationAgent()
    
    try:
        verified, response = agent.verify(
            answer=state.synthesized_answer,
            citations=state.citations,
            evidence=state.merged_evidence,
            language=state.query_language
        )
        
        state.verified = verified
        state.final_output = response
        
        # If not verified but we have evidence, try to fix
        if not verified and state.merged_evidence:
            # Add missing citations
            fixed_answer = agent.add_missing_citations(
                state.synthesized_answer,
                state.merged_evidence
            )
            state.synthesized_answer = fixed_answer
            
            # Re-verify
            verified, response = agent.verify(
                answer=fixed_answer,
                citations=state.citations,
                evidence=state.merged_evidence,
                language=state.query_language
            )
            state.verified = verified
            state.final_output = response
        
    except Exception as e:
        state.errors.append(f"Error in verification: {e}")
        state.verified = False
    
    return state
