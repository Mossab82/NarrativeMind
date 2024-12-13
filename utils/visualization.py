class ResultVisualizer:
    @staticmethod
    def plot_cultural_preservation(
        scores: Dict[str, float],
        title: str = "Cultural Element Preservation"
    ):
        """Plot cultural preservation scores."""
        plt.figure(figsize=(8, 6))
        
        categories = list(scores.keys())
        values = list(scores.values())
        
        sns.barplot(x=categories, y=values)
        plt.title(title)
        plt.xlabel("Category")
        plt.ylabel("Preservation Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_dialect_distribution(
        predictions: List[str],
        title: str = "Dialect Distribution"
    ):
        """Plot distribution of predicted dialects."""
        plt.figure(figsize=(10, 6))
        
        df = pd.DataFrame({'dialect': predictions})
        sns.countplot(data=df, x='dialect')
        
        plt.title(title)
        plt.xlabel("Dialect")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_generation_length_distribution(
        lengths: List[int],
        title: str = "Text Length Distribution"
    ):
        """Plot distribution of generated text lengths."""
        plt.figure(figsize=(10, 6))
        
        sns.histplot(lengths, bins=30)
        plt.title(title)
        plt.xlabel("Text Length (words)")
        plt.ylabel("Count")
        plt.tight_layout()
        
        return plt.gcf()
