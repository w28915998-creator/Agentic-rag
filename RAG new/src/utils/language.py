"""
Language detection and text normalization utilities.
Supports English, Urdu, and mixed language detection.
"""

import re
import unicodedata
from typing import Tuple
from langdetect import detect, detect_langs, LangDetectException

from src.models import Language


# Urdu Unicode range
URDU_RANGE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')

# English letters
ENGLISH_RANGE = re.compile(r'[a-zA-Z]')


def detect_language(text: str) -> Language:
    """
    Detect the language of the given text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language enum value (ENGLISH, URDU, or MIXED)
    """
    if not text or not text.strip():
        return Language.ENGLISH
    
    # Count Urdu and English characters
    urdu_chars = len(URDU_RANGE.findall(text))
    english_chars = len(ENGLISH_RANGE.findall(text))
    total_alpha = urdu_chars + english_chars
    
    if total_alpha == 0:
        # No alphabetic characters, try langdetect
        try:
            detected = detect(text)
            if detected == 'ur':
                return Language.URDU
            return Language.ENGLISH
        except LangDetectException:
            return Language.ENGLISH
    
    urdu_ratio = urdu_chars / total_alpha
    english_ratio = english_chars / total_alpha
    
    # Determine language based on ratios
    if urdu_ratio > 0.7:
        return Language.URDU
    elif english_ratio > 0.7:
        return Language.ENGLISH
    else:
        return Language.MIXED


def get_language_breakdown(text: str) -> Tuple[float, float]:
    """
    Get the percentage breakdown of Urdu vs English characters.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (urdu_percentage, english_percentage)
    """
    urdu_chars = len(URDU_RANGE.findall(text))
    english_chars = len(ENGLISH_RANGE.findall(text))
    total = urdu_chars + english_chars
    
    if total == 0:
        return (0.0, 0.0)
    
    return (urdu_chars / total, english_chars / total)


def normalize_text(text: str, language: Language = None) -> str:
    """
    Normalize text for processing.
    
    Args:
        text: Input text to normalize
        language: Optional language hint
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Unicode normalization (NFC for composed characters)
    text = unicodedata.normalize('NFC', text)
    
    # Remove zero-width characters
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_urdu_text(text: str) -> str:
    """
    Specific normalization for Urdu text.
    
    Args:
        text: Input Urdu text
        
    Returns:
        Normalized Urdu text
    """
    # Apply general normalization first
    text = normalize_text(text)
    
    # Normalize Urdu-specific characters
    # Replace Arabic Yeh with Urdu Yeh
    text = text.replace('\u064a', '\u06cc')  # ي -> ی
    
    # Replace Arabic Kaf with Urdu Kaf
    text = text.replace('\u0643', '\u06a9')  # ك -> ک
    
    # Normalize Hamza
    text = text.replace('\u0623', '\u0627')  # أ -> ا
    text = text.replace('\u0625', '\u0627')  # إ -> ا
    
    return text


def is_urdu_dominant(text: str) -> bool:
    """
    Check if text is predominantly Urdu.
    
    Args:
        text: Input text
        
    Returns:
        True if Urdu is dominant
    """
    return detect_language(text) == Language.URDU


def is_english_dominant(text: str) -> bool:
    """
    Check if text is predominantly English.
    
    Args:
        text: Input text
        
    Returns:
        True if English is dominant
    """
    return detect_language(text) == Language.ENGLISH
