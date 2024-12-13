import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
from camel_tools.dialectid import DialectIdentifier
import numpy as np
import logging

logger = logging.getLogger(__name__)

class NarrativeMind:
    """Hybrid neural-symbolic system for Arabic story generation."""
    
    def __init__(
        self,
        model_name: str = "asafaya/bert-base-arabic",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config_path: Optional[str] = None
    ):
        """Initialize NarrativeMind system.
        
        Args:
            model_name: Pre-trained Arabic language model name
            device: Computing device to use
            config_path: Optional path to configuration file
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Load configuration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize components
        self._init_pattern_matcher()
        self._init_dialect_processor()
        
        # Cultural integration parameters (Equation 1 from paper)
        self.lambda_cultural = self.config.get('lambda_cultural', 0.7)
        
        logger.info(f"Initialized NarrativeMind with model: {model_name}")
        
    def _init_pattern_matcher(self):
        """Initialize cultural pattern matching component."""
        # Traditional narrative patterns (Section 1.1 of paper)
        self.patterns = {
            "opening": {
                "traditional": [
                    "كان يا ما كان في قديم الزمان",
                    "يحكى أنه في سالف العصر والأوان"
                ],
                "modern": [
                    "في يوم من الأيام",
                    "في زمن مضى"
                ]
            },
            "development": {
                "sequential": [
                    "وفي يوم من الأيام",
                    "وبينما هو كذلك"
                ],
                "temporal": [
                    "وبعد مرور الزمان",
                    "ومع مرور الوقت"
                ]
            },
            "resolution": {
                "moral": [
                    "وهكذا تعلم الجميع",
                    "وكان في ذلك عبرة"
                ],
                "traditional": [
                    "وعاشوا في سعادة ونعيم",
                    "وتمت القصة بخير"
                ]
            }
        }
        
        # Cultural elements (Section 2.4 of paper)
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
        
    def _init_dialect_processor(self):
        """Initialize dialect processing component."""
        self.dialect_id = DialectIdentifier.pretrained()
        
        # Dialect adaptation patterns (Section 3.4 of paper)
        self.dialect_patterns = {
            "egyptian": {
                "في قديم الزمان": "في يوم من الأيام",
                "ذهب": "راح",
                "قال": "ءال"
            },
            "gulf": {
                "في قديم الزمان": "يحكى إنه من قديم",
                "ذهب": "راح",
                "قال": "گال"
            },
            "levantine": {
                "في قديم الزمان": "كان في زمان وكان",
                "ذهب": "راح",
                "قال": "حكى"
            }
        }

    def generate_story(
        self,
        prompt: str,
        target_dialect: Optional[str] = None,
        max_length: int = 512,
        num_beams: int = 5,
        temperature: float = 0.7,
        cultural_weight: Optional[float] = None
    ) -> Tuple[str, Dict[str, float]]:
        """Generate a culturally-aware Arabic story.
        
        Args:
            prompt: Story prompt
            target_dialect: Target dialect for generation
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            cultural_weight: Optional override for cultural integration weight
            
        Returns:
            Tuple of (generated story, generation metrics)
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
            return_dict_in_generate=True,
            output_scores=True
        )
        
        story = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Apply cultural pattern matching
        lambda_c = cultural_weight if cultural_weight is not None else self.lambda_cultural
        story, cultural_score = self._apply_cultural_patterns(story, lambda_c)
        
        # Apply dialect adaptation if specified
        if target_dialect:
            story = self._adapt_dialect(story, target_dialect)
            
        # Compute generation metrics
        metrics = {
            'cultural_score': cultural_score,
            'length': len(story.split()),
            'perplexity': self._compute_perplexity(story)
        }
        
        if target_dialect:
            metrics['dialect_score'] = self._evaluate_dialect(story, target_dialect)
            
        return story, metrics

    def _apply_cultural_patterns(
        self, 
        text: str, 
        lambda_cultural: float
    ) -> Tuple[str, float]:
        """Apply cultural pattern matching with hybrid scoring.
        
        Implementation of Equation 1 from paper:
        O(x) = λC(x) + (1-λ)N(x)
        
        Args:
            text: Input text
            lambda_cultural: Weight for cultural scoring
            
        Returns:
            Tuple of (enhanced text, cultural score)
        """
        # Calculate neural score N(x)
        neural_score = -self._compute_perplexity(text)  # Negative perplexity
        
        # Calculate cultural score C(x)
        cultural_score = self._compute_cultural_score(text)
        
        # Combine scores using equation 1
        combined_score = (
            lambda_cultural * cultural_score + 
            (1 - lambda_cultural) * neural_score
        )
        
        # Enhance text based on combined score
        enhanced_text = text
        if combined_score < self.config.get('enhancement_threshold', 0.5):
            enhanced_text = self._enhance_cultural_elements(text)
            
        return enhanced_text, cultural_score
