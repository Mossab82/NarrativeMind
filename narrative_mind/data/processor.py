import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from camel_tools.utils.normalize import normalize_unicode
import json

logger = logging.getLogger(__name__)

class MADARProcessor:
    """Processor for MADAR parallel corpus and related datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize MADAR processor.
        
        Args:
            data_dir: Base directory for data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # MADAR corpus statistics from paper
        self.corpus_stats = {
            'num_parallel_sentences': 12000,
            'num_cities': 25,
            'total_tokens': 5900000
        }
        
    def process_parallel_corpus(
        self,
        corpus_path: str,
        save: bool = True
    ) -> pd.DataFrame:
        """Process MADAR parallel corpus.
        
        Args:
            corpus_path: Path to MADAR corpus file
            save: Whether to save processed data
            
        Returns:
            DataFrame with processed parallel text
        """
        logger.info(f"Processing MADAR corpus from: {corpus_path}")
        
        # Read corpus (tab-separated format)
        df = pd.read_csv(corpus_path, sep='\t')
        
        # Process each dialect
        processed_data = []
        for dialect in df.columns[1:]:  # Skip ID column
            texts = df[dialect].tolist()
            
            # Create samples with proper splits (80/10/10)
            for i, text in enumerate(texts):
                processed_data.append({
                    'text': self._preprocess_text(text),
                    'dialect': dialect,
                    'split': 'train' if i < len(texts) * 0.8 else
                            'val' if i < len(texts) * 0.9 else 'test'
                })
        
        processed_df = pd.DataFrame(processed_data)
        
        if save:
            output_path = self.processed_dir / "madar_processed.parquet"
            processed_df.to_parquet(output_path)
            logger.info(f"Saved processed data to: {output_path}")
        
        return processed_df
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess Arabic text."""
        # Normalize Unicode
        text = normalize_unicode(text)
        
        # Basic cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text
    
    def create_story_dataset(
        self,
        stories_path: str,
        save: bool = True
    ) -> pd.DataFrame:
        """Process custom story dataset.
        
        Args:
            stories_path: Path to story dataset
            save: Whether to save processed data
            
        Returns:
            DataFrame with processed stories
        """
        logger.info(f"Processing story dataset from: {stories_path}")
        
        with open(stories_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)
            
        processed_data = []
        for story in stories:
            processed_data.append({
                'text': self._preprocess_text(story['text']),
                'dialect': story.get('dialect', 'MSA'),
                'patterns': story.get('patterns', []),
                'cultural_elements': story.get('cultural_elements', []),
                'split': story.get('split', 'train')
            })
            
        processed_df = pd.DataFrame(processed_data)
        
        if save:
            output_path = self.processed_dir / "stories_processed.parquet"
            processed_df.to_parquet(output_path)
            logger.info(f"Saved processed stories to: {output_path}")
            
        return processed_df
