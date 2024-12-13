from typing import List, Dict, Optional
import numpy as np
from camel_tools.utils.normalize import normalize_unicode
import random
import logging

logger = logging.getLogger(__name__)

class DataAugmenter:
    """Data augmentation utilities for Arabic text."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize data augmenter.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Load augmentation patterns
        self.patterns = {
            "synonyms": {
                "ذهب": ["مشى", "راح", "توجه"],
                "قال": ["تحدث", "تكلم", "حكى"],
                "كثيراً": ["جداً", "للغاية", "بشدة"]
            },
            "narrative_variations": {
                "opening": [
                    "كان يا ما كان",
                    "في قديم الزمان",
                    "يحكى أنه"
                ],
                "transition": [
                    "وفي يوم من الأيام",
                    "وبينما هو كذلك",
                    "وبعد ذلك"
                ],
                "closing": [
                    "وعاشوا في سعادة",
                    "وهكذا انتهت القصة",
                    "وتمت القصة بخير"
                ]
            }
        }
    
    def augment_text(
        self,
        text: str,
        techniques: Optional[List[str]] = None
    ) -> List[str]:
        """Apply augmentation techniques to text.
        
        Args:
            text: Input text
            techniques: List of techniques to apply
            
        Returns:
            List of augmented texts
        """
        if techniques is None:
            techniques = ['synonym', 'narrative', 'combination']
            
        augmented = []
        
        if 'synonym' in techniques:
            augmented.extend(self._synonym_replacement(text))
            
        if 'narrative' in techniques:
            augmented.extend(self._narrative_variation(text))
            
        if 'combination' in techniques:
            # Apply both techniques
            for syn_aug in self._synonym_replacement(text):
                augmented.extend(self._narrative_variation(syn_aug))
                
        return list(set(augmented))  # Remove duplicates
    
    def _synonym_replacement(self, text: str) -> List[str]:
        """Replace words with synonyms."""
        augmented = []
        words = text.split()
        
        for i, word in enumerate(words):
            if word in self.patterns['synonyms']:
                for synonym in self.patterns['synonyms'][word]:
                    new_words = words.copy()
                    new_words[i] = synonym
                    augmented.append(' '.join(new_words))
                    
        return augmented
    
    def _narrative_variation(self, text: str) -> List[str]:
        """Create variations using narrative patterns."""
        augmented = []
        
        # Identify narrative parts
        parts = text.split('.')
        if len(parts) < 2:
            return augmented
            
        # Vary opening
        for opening in self.patterns['narrative_variations']['opening']:
            if any(p in parts[0] for p in self.patterns['narrative_variations']['opening']):
                new_parts = parts.copy()
                new_parts[0] = opening + parts[0].split(' ', 3)[-1]
                augmented.append('.'.join(new_parts))
                
        # Vary transitions
        for i in range(1, len(parts)-1):
            for transition in self.patterns['narrative_variations']['transition']:
                new_parts = parts.copy()
                new_parts[i] = transition + new_parts[i].strip()
                augmented.append('.'.join(new_parts))
                
        return augmented
