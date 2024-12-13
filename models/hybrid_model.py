import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional

class NarrativeMind:
    def __init__(
        self, 
        model_name: str = "asafaya/bert-base-arabic",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize NarrativeMind model with cultural-aware narrative generation.
        
        Args:
            model_name: Base Arabic language model to use
            device: Device to run model on ("cuda" or "cpu")
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Cultural pattern matching weights
        self.lambda_cultural = 0.7  # Weight for cultural scoring
        
        # Load dialect mappings
        self.dialect_patterns = {
            "egyptian": self._load_dialect_patterns("egyptian"),
            "gulf": self._load_dialect_patterns("gulf"), 
            "levantine": self._load_dialect_patterns("levantine"),
            "moroccan": self._load_dialect_patterns("moroccan")
        }
        
        # Load cultural patterns
        self.cultural_patterns = {
            "opening": [
                "كان يا ما كان في قديم الزمان",
                "يحكى أنه في سالف العصر والأوان",
            ],
            "transition": [
                "وفي يوم من الأيام",
                "وبينما هو كذلك",
            ],
            "closing": [
                "وعاشوا في سعادة ونعيم",
                "وهكذا تعلم الجميع"
            ]
        }

    def generate_story(
        self,
        prompt: str,
        dialect: Optional[str] = None,
        max_length: int = 512,
        num_beams: int = 5,
        temperature: float = 0.7
    ) -> str:
        """Generate a culturally-aware Arabic story from a prompt.
        
        Args:
            prompt: Story prompt/beginning
            dialect: Target dialect (egyptian, gulf, levantine, moroccan)
            max_length: Maximum story length
            num_beams: Number of beams for beam search
            temperature: Temperature for generation sampling
            
        Returns:
            Generated story text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate initial story
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            early_stopping=True,
        )
        
        story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Apply cultural pattern matching
        story = self._apply_cultural_patterns(story)
        
        # Apply dialect adaptation if specified
        if dialect and dialect in self.dialect_patterns:
            story = self._adapt_dialect(story, dialect)
            
        return story
    
    def _apply_cultural_patterns(self, text: str) -> str:
        """Apply cultural narrative patterns to generated text."""
        # Check and insert opening pattern if missing
        if not any(p in text[:100] for p in self.cultural_patterns["opening"]):
            text = self.cultural_patterns["opening"][0] + " " + text
            
        # Add transition phrases
        paragraphs = text.split("\n")
        enhanced = []
        for i, para in enumerate(paragraphs):
            if i > 0 and len(para) > 50:  # Add transitions between major sections
                enhanced.append(self.cultural_patterns["transition"][0] + " " + para)
            else:
                enhanced.append(para)
                
        text = "\n".join(enhanced)
        
        # Add closing if missing
        if not any(p in text[-100:] for p in self.cultural_patterns["closing"]):
            text += " " + self.cultural_patterns["closing"][0]
            
        return text
    
    def _adapt_dialect(self, text: str, dialect: str) -> str:
        """Adapt text to target dialect using loaded dialect patterns."""
        adapted = text
        patterns = self.dialect_patterns[dialect]
        
        # Apply dialect-specific substitutions
        for msa, dialectal in patterns.items():
            adapted = adapted.replace(msa, dialectal)
            
        return adapted
    
    def _load_dialect_patterns(self, dialect: str) -> Dict[str, str]:
        """Load dialect adaptation patterns from resources."""
        # This would load from JSON/YAML files in practice
        # Simplified example patterns shown here
        patterns = {
            "egyptian": {
                "في قديم الزمان": "في يوم من الأيام",
                "ذهب": "راح",
            },
            "gulf": {
                "في قديم الزمان": "يحكى إنه من قديم",
                "ذهب": "راح",
            },
            "levantine": {
                "في قديم الزمان": "كان في زمان وكان",
                "ذهب": "راح",
            },
            "moroccan": {
                "في قديم الزمان": "واحد النهار كان",
                "ذهب": "مشى",
            }
        }
        return patterns.get(dialect, {})
    
    def evaluate_cultural_preservation(self, text: str) -> float:
        """Evaluate cultural preservation score of generated text.
        
        Returns:
            Score between 0-1 indicating cultural pattern preservation
        """
        score = 0.0
        checks = [
            # Check narrative structure
            any(p in text[:100] for p in self.cultural_patterns["opening"]),
            any(p in text[-100:] for p in self.cultural_patterns["closing"]),
            any(p in text for p in self.cultural_patterns["transition"]),
            
            # Check cultural elements (simplified)
            "الشرف" in text or "الكرامة" in text,  # Honor concepts
            "الحكمة" in text or "الأمثال" in text,  # Wisdom sayings
            "العبر" in text or "الدروس" in text,    # Moral lessons
        ]
        score = sum(1 for c in checks if c) / len(checks)
        return score

    def save(self, path: str):
        """Save model and configuration."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    @classmethod
    def load(cls, path: str):
        """Load saved model."""
        instance = cls(model_name=path)
        return instance
