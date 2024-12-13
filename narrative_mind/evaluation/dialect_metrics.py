from typing import Dict, List
from camel_tools.dialectid import DialectIdentifier
import numpy as np

class DialectMetrics:
    """Dialect accuracy metrics."""
    
    def __init__(self):
        """Initialize dialect metrics."""
        self.dialect_id = DialectIdentifier.pretrained()
        
    def evaluate(
        self,
        text: str,
        target_dialect: str
    ) -> Dict[str, float]:
        """Evaluate dialect accuracy.
        
        Args:
            text: Input Arabic text
            target_dialect: Target dialect
            
        Returns:
            Dictionary of dialect metrics
        """
        metrics = {}
        
        # Get dialect predictions
        predictions = self.dialect_id.predict(text.split('\n'))
        
        # Calculate accuracy (κ = 0.72 reported in paper)
        correct = sum(1 for pred in predictions if pred.top == target_dialect)
        metrics['dialect_accuracy'] = correct / len(predictions)
        
        # Calculate confidence scores
        confidences = [pred.scores[target_dialect] for pred in predictions]
        metrics['dialect_confidence'] = np.mean(confidences)
        
        return metrics
