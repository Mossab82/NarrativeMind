import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class NeuralGenerator:
    """Neural generation component based on AraBERT."""
    
    def __init__(
        self,
        model_name: str = "asafaya/bert-base-arabic",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize neural generator.
        
        Args:
            model_name: Pre-trained Arabic language model name
            device: Computing device to use
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        logger.info(f"Initialized neural generator with model: {model_name}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        num_beams: int = 5,
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[str, Dict[str, float]]:
        """Generate Arabic text using neural model.
        
        Args:
            prompt: Text prompt for generation
            max_length: Maximum generation length 
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            Tuple of (generated text, generation metrics)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
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
            output_scores=True,
            **kwargs
        )
        
        # Decode generated text
        text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Calculate metrics
        metrics = {
            'length': len(text.split()),
            'perplexity': self._compute_perplexity(text)
        }
        
        return text, metrics
    
    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity of generated text."""
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs, labels=inputs)
            return torch.exp(outputs.loss).item()
