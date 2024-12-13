from typing import List
import re
from camel_tools.utils.normalize import normalize_unicode

class ArabicUtils:
    """Utilities for Arabic text processing."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize Arabic text.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Normalized text
        """
        # Apply CAMeL Tools normalization
        text = normalize_unicode(text)
        
        # Additional normalization
        text = re.sub("[إأٱآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("ؤ", "و", text)
        text = re.sub("ئ", "ي", text)
        
        return text
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text."""
        # Split on Arabic sentence markers
        markers = r'[.!؟\n]'
        sentences = re.split(markers, text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def contains_arabic(text: str) -> bool:
        """Check if text contains Arabic."""
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        return bool(arabic_pattern.search(text))
