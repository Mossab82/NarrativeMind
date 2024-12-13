import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd

class Visualizer:
    """Visualization utilities for evaluation results."""
    
    @staticmethod
    def plot_metrics(
        metrics: Dict[str, List[float]],
        title: str = "Evaluation Metrics"
    ):
        """Plot evaluation metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        
        df = pd.DataFrame(metrics)
        sns.boxplot(data=df)
        
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_dialect_distribution(
        predictions: List[str],
        title: str = "Dialect Distribution"
    ):
        """Plot dialect distribution.
        
        Args:
            predictions: List of dialect predictions
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        
        df = pd.DataFrame({'dialect': predictions})
        sns.countplot(data=df, x='dialect')
        
        plt.title(title)
        plt.xlabel("Dialect")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
