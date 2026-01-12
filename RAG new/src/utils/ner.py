"""
Named Entity Recognition and Relationship Extraction utilities.
Supports English and Urdu text processing.
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from src.models import EntityNode, RelationshipEdge, Language
from src.utils.language import detect_language


@dataclass
class ExtractedEntity:
    """Represents an extracted entity before conversion to EntityNode."""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = 1.0


class NERExtractor:
    """
    Named Entity Recognition extractor supporting English and Urdu.
    Uses spaCy for NER and custom patterns for Urdu.
    """
    
    _nlp_en = None
    _nlp_ur = None
    _initialized = False
    
    # Common Urdu entity patterns
    URDU_PERSON_PATTERNS = [
        r'جناب\s+[\u0600-\u06FF]+',
        r'محترم\s+[\u0600-\u06FF]+',
        r'صاحب\s+[\u0600-\u06FF]+',
        r'[\u0600-\u06FF]+\s+خان',
        r'[\u0600-\u06FF]+\s+علی',
        r'[\u0600-\u06FF]+\s+احمد',
    ]
    
    URDU_LOCATION_PATTERNS = [
        r'شہر\s+[\u0600-\u06FF]+',
        r'[\u0600-\u06FF]+\s+صوبہ',
        r'پاکستان|بھارت|امریکہ|چین|برطانیہ',
        r'کراچی|لاہور|اسلام آباد|پشاور|کوئٹہ',
    ]
    
    URDU_ORG_PATTERNS = [
        r'[\u0600-\u06FF]+\s+یونیورسٹی',
        r'[\u0600-\u06FF]+\s+کمپنی',
        r'[\u0600-\u06FF]+\s+ادارہ',
        r'[\u0600-\u06FF]+\s+بینک',
    ]
    
    URDU_DATE_PATTERNS = [
        r'\d{4}\s*ء?',  # Year like 1999 or 1999ء
        r'سال\s+\d{4}', # Year like "Year 2024"
        r'[0-9]+\s+(جنوری|فروری|مارچ|اپریل|مئی|جون|جولائی|اگست|ستمبر|اکتوبر|نومبر|دسمبر)', # Date like "14 August"
    ]
    
    def __init__(self):
        """Initialize NER models."""
        self._initialize_models()
    
    def _initialize_models(self):
        """Load spaCy models for NER."""
        # Check class variable
        if NERExtractor._initialized:
            return
            
        try:
            import spacy
            
            # Try to load English model
            try:
                # Set class variable
                NERExtractor._nlp_en = spacy.load("en_core_web_sm")
                print("Loaded English NER model: en_core_web_sm")
            except OSError:
                print("Warning: en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
                # Use blank model as fallback
                NERExtractor._nlp_en = spacy.blank("en")
            
            NERExtractor._initialized = True
            
        except ImportError:
            print("Warning: spaCy not installed. NER will use pattern matching only.")
            NERExtractor._initialized = True
    
    def extract_entities_english(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from English text using spaCy.
        
        Args:
            text: Input English text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        if NERExtractor._nlp_en is None:
            return entities
        
        doc = NERExtractor._nlp_en(text)
        
        for ent in doc.ents:
            entities.append(ExtractedEntity(
                text=ent.text,
                entity_type=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.9
            ))
            
        # Fallback for lowercase queries (common in chat interfaces)
        if not entities and text.islower():
            doc_title = NERExtractor._nlp_en(text.title())
            for ent in doc_title.ents:
                # Map back to original indices if possible, or just use text
                # For queries, exact indices matter less than finding the entity name
                entities.append(ExtractedEntity(
                    text=ent.text,
                    entity_type=ent.label_,
                    start=0,  # Approximate
                    end=0,    # Approximate
                    confidence=0.8
                ))
        
        return entities
    
    def extract_entities_urdu(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from Urdu text using pattern matching.
        
        Args:
            text: Input Urdu text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract person names
        for pattern in self.URDU_PERSON_PATTERNS:
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type="PERSON",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7
                ))
        
        # Extract locations
        for pattern in self.URDU_LOCATION_PATTERNS:
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type="LOCATION",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7
                ))
        
        # Extract organizations
        for pattern in self.URDU_ORG_PATTERNS:
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type="ORGANIZATION",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7
                ))
                
        # Extract dates
        for pattern in self.URDU_DATE_PATTERNS:
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type="DATE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        return entities
    
    def extract_entities(
        self,
        text: str,
        doc_id: str,
        chunk_id: str,
        language: Language = None
    ) -> List[EntityNode]:
        """
        Extract entities from text and return EntityNode objects.
        
        Args:
            text: Input text
            doc_id: Document ID for provenance
            chunk_id: Chunk ID for provenance
            language: Optional language hint
            
        Returns:
            List of EntityNode objects
        """
        if language is None:
            language = detect_language(text)
        
        # Determine extraction strategy based on content
        # We run specific extractors if we detect relevant characters, 
        # largely ignoring the dominant language classification to be robust.
        extracted = []
        
        has_english = bool(re.search(r'[a-zA-Z]', text))
        has_urdu = bool(re.search(r'[\u0600-\u06FF]', text))
        
        if has_english:
            extracted.extend(self.extract_entities_english(text))
            
        if has_urdu:
            extracted.extend(self.extract_entities_urdu(text))
            
        # Fallback
        if not has_english and not has_urdu:
            extracted.extend(self.extract_entities_english(text))
        
        # Convert to EntityNode objects
        entities = []
        seen_names = set()
        
        for ext in extracted:
            # Deduplicate within chunk
            name_key = (ext.text.lower(), ext.entity_type)
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            
            entity = EntityNode(
                entity_type=ext.entity_type,
                name=ext.text,
                source_doc_id=doc_id,
                chunk_id=chunk_id,
                properties={
                    "confidence": ext.confidence,
                    "start": ext.start,
                    "end": ext.end
                },
                language=language
            )
            entities.append(entity)
        
        return entities
    
    def extract_relationships(
        self,
        text: str,
        entities: List[EntityNode],
        doc_id: str,
        chunk_id: str
    ) -> List[RelationshipEdge]:
        """
        Extract relationships between entities in text.
        
        Args:
            text: Input text
            entities: Entities found in the text
            doc_id: Document ID
            chunk_id: Chunk ID
            
        Returns:
            List of RelationshipEdge objects
        """
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Simple co-occurrence based relationship extraction
        # Entities appearing in the same sentence are considered related
        
        # Common relationship verbs (English)
        relation_patterns = [
            (r'(\w+)\s+works?\s+(?:at|for|with)\s+(\w+)', 'WORKS_AT'),
            (r'(\w+)\s+(?:is|was)\s+(?:a|the)\s+\w+\s+of\s+(\w+)', 'MEMBER_OF'),
            (r'(\w+)\s+(?:founded|created|started)\s+(\w+)', 'FOUNDED'),
            (r'(\w+)\s+(?:located|based)\s+in\s+(\w+)', 'LOCATED_IN'),
            (r'(\w+)\s+(?:married|wed)\s+(\w+)', 'MARRIED_TO'),
        ]
        
        # Track existing pairs to avoid duplicates
        existing_pairs = set()
        
        for pattern, rel_type in relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Find matching entities
                subj = match.group(1)
                obj = match.group(2)
                
                subj_entity = self._find_matching_entity(subj, entities)
                obj_entity = self._find_matching_entity(obj, entities)
                
                if subj_entity and obj_entity and subj_entity != obj_entity:
                    pair_key = tuple(sorted([subj_entity.name, obj_entity.name]))
                    if pair_key not in existing_pairs:
                        relationships.append(RelationshipEdge(
                            from_entity=subj_entity.name,
                            to_entity=obj_entity.name,
                            relation_type=rel_type,
                            source_doc_id=doc_id,
                            chunk_id=chunk_id,
                            properties={"pattern": pattern}
                        ))
                        existing_pairs.add(pair_key)
        
        # Add co-occurrence relationships for remaining pairs
        # Limit to reasonable number to avoid explosion
        max_edges = 50
        if len(relationships) < max_edges and len(entities) >= 2:
            for i, ent1 in enumerate(entities):
                for ent2 in entities[i+1:]:
                    if len(relationships) >= max_edges:
                        break
                        
                    pair_key = tuple(sorted([ent1.name, ent2.name]))
                    if pair_key in existing_pairs:
                        continue
                    
                    # Determine relationship type
                    rel_type = "RELATED"
                    
                    # Temporal relationship?
                    if ent1.entity_type == "DATE" or ent2.entity_type == "DATE":
                        rel_type = "TEMPORAL"
                    
                    relationships.append(RelationshipEdge(
                        from_entity=ent1.name,
                        to_entity=ent2.name,
                        relation_type=rel_type,
                        source_doc_id=doc_id,
                        chunk_id=chunk_id,
                        properties={}
                    ))
                    existing_pairs.add(pair_key)
        
        return relationships
    
    def _find_matching_entity(
        self,
        text: str,
        entities: List[EntityNode]
    ) -> Optional[EntityNode]:
        """Find an entity that matches the given text."""
        text_lower = text.lower()
        
        for entity in entities:
            if text_lower in entity.name.lower():
                return entity
            if entity.name.lower() in text_lower:
                return entity
        
        return None


# Global instance
ner_extractor = NERExtractor()
