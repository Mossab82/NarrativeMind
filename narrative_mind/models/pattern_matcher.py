from typing import Dict, List, Tuple
import re
import logging
import numpy as np

logger = logging.getLogger(__name__)

class PatternMatcher:
    """Cultural pattern matching component."""
    
    def __init__(self):
        """Initialize pattern matcher with cultural patterns."""
        # Load narrative patterns
        self.narrative_patterns = {
            "opening": {
                "traditional": [
                    r"كان يا ما كان في قديم الزمان",
                    r"يحكى أنه في سالف العصر والأوان"
                ],
                "modern": [
                    r"في يوم من الأيام",
                    r"في زمن مضى"
                ]
            },
            "development": {
                "sequential": [
                    r"وفي يوم من الأيام",
                    r"وبينما هو كذلك"
                ],
                "temporal": [
                    r"وبعد مرور الزمان",
                    r"ومع مرور الوقت"
                ]
            },
            "resolution": {
                "moral": [
                    r"وهكذا تعلم الجميع",
                    r"وكان في ذلك عبرة"
                ],
                "traditional": [
                    r"وعاشوا في سعادة ونعيم",
                    r"وتمت القصة بخير"
                ]
            }
        }
        
        # Load cultural elements
        self.cultural_elements = {
            "values": [
                "الشرف", "الكرامة", "الأمانة", "الشجاعة"
            ],
            "wisdom": [
                "الحكمة", "الأمثال", "العبر", "الدروس"
            ],
            "rhetoric": [
                "السجع", "الجناس", "التشبيه", "الاستعارة"
            ]
        }
        
        # Compile patterns
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.compiled_patterns = {}
        for category, subcats in self.narrative_patterns.items():
            self.compiled_patterns[category] = {}
            for subcat, patterns in subcats.items():
                self.compiled_patterns[category][subcat] = [
                    re.compile(p, re.UNICODE) for p in patterns
                ]
    
    def match_patterns(self, text: str) -> Dict[str, List[str]]:
        """Find all narrative patterns in text.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Dictionary mapping pattern categories to matched patterns
        """
        matches = {}
        for category, subcats in self.compiled_patterns.items():
            matches[category] = []
            for subcat, patterns in subcats.items():
                for pattern in patterns:
                    found = pattern.findall(text)
                    matches[category].extend(found)
        return matches
    
    def enhance_text(self, text: str) -> Tuple[str, float]:
        """Enhance text with cultural patterns.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (enhanced text, enhancement score)
        """
        original_score = self.compute_cultural_score(text)
        enhanced = text
        
        # Add opening if missing
        if not self.has_pattern(text, "opening", position="start"):
            enhanced = f"{self.narrative_patterns['opening']['traditional'][0]} {enhanced}"
            
        # Add development markers if needed
        sentences = enhanced.split('.')
        if len(sentences) > 2:
            enhanced_sents = []
            for i, sent in enumerate(sentences):
                if i > 0 and len(sent) > 20 and not self.has_pattern(sent, "development"):
                    enhanced_sents.append(
                        f"{self.narrative_patterns['development']['sequential'][0]} {sent}"
                    )
                else:
                    enhanced_sents.append(sent)
            enhanced = '. '.join(enhanced_sents)
            
        # Add resolution if missing
        if not self.has_pattern(text, "resolution", position="end"):
            enhanced = f"{enhanced} {self.narrative_patterns['resolution']['moral'][0]}"
            
        enhanced_score = self.compute_cultural_score(enhanced)
        improvement = enhanced_score - original_score
        
        return enhanced, improvement
    
    def compute_cultural_score(self, text: str) -> float:
        """Compute cultural preservation score.
        
        Args:
            text: Input text
            
        Returns:
            Score between 0-1 indicating cultural preservation
        """
        # Pattern matching score
        pattern_matches = self.match_patterns(text)
        pattern_score = sum(
            len(matches) for matches in pattern_matches.values()
        ) / len(self.narrative_patterns)
        
        # Cultural elements score
        element_count = 0
        for category, elements in self.cultural_elements.items():
            element_count += sum(1 for elem in elements if elem in text)
        element_score = element_count / sum(len(elems) for elems in self.cultural_elements.values())
        
        # Combine scores
        return (pattern_score + element_score) / 2
    
    def has_pattern(
        self, 
        text: str, 
        category: str, 
        position: str = "any"
    ) -> bool:
        """Check if text contains pattern from category.
        
        Args:
            text: Input text
            category: Pattern category to check
            position: Where to look for pattern ("start", "end", or "any")
        """
        if category not in self.compiled_patterns:
            return False
            
        check_text = text
        if position == "start":
            check_text = text[:100]
        elif position == "end":
            check_text = text[-100:]
            
        for subcat, patterns in self.compiled_patterns[category].items():
            for pattern in patterns:
                if pattern.search(check_text):
                    return True
        return False
