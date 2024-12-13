import numpy as np
from typing import Dict, List, Optional, Union
from nltk.translate.bleu_score import sentence_bleu
from camel_tools.dialectid import DialectIdentifier
import logging

logger = logging.getLogger(__name__)

class NarrativeEvaluator:
    """Main evaluation framework for NarrativeMind."""
    
    def __init__(self):
        """Initialize evaluator with all metrics."""
        self.cultural_metrics = CulturalMetrics()
        self.dialect_metrics = DialectMetrics()
        logger.info("Initialized NarrativeEvaluator")
        
    def evaluate(
        self,
        generated_text: str,
        reference_text: Optional[str] = None,
        target_dialect: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate generated text on all metrics.
        
        Args:
            generated_text: Generated Arabic text
            reference_text: Optional reference text
            target_dialect: Optional target dialect
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['length'] = len(generated_text.split())
        
        # Cultural preservation metrics
        cultural_scores = self.cultural_metrics.evaluate(generated_text)
        metrics.update(cultural_scores)
        
        # Dialect accuracy if specified
        if target_dialect:
            dialect_scores = self.dialect_metrics.evaluate(
                generated_text, target_dialect)
            metrics.update(dialect_scores)
        
        # Compare with reference if provided
        if reference_text:
            reference_scores = self._evaluate_with_reference(
                generated_text, reference_text)
            metrics.update(reference_scores)
        
        return metrics
    
    def _evaluate_with_reference(
        self,
        generated: str,
        reference: str
    ) -> Dict[str, float]:
        """Compare generated text with reference."""
        metrics = {}
        
        # BLEU score (reported 29.8 ± 0.4 in paper)
        metrics['bleu'] = sentence_bleu(
            [reference.split()],
            generated.split()
        )
        
        return metrics
