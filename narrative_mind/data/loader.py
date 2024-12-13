from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class NarrativeDataset(Dataset):
    """Dataset for narrative generation."""
    
    def __init__(
        self,
        texts: List[str],
        dialects: Optional[List[str]] = None,
        tokenizer = None,
        max_length: int = 512
    ):
        """Initialize dataset.
        
        Args:
            texts: List of text samples
            dialects: Optional list of dialect labels
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.dialects = dialects
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item.
        
        Returns:
            Dictionary with encoded text and optional dialect
        """
        text = self.texts[idx]
        
        # Encode text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        # Add dialect if available
        if self.dialects is not None:
            item['dialect'] = self.dialects[idx]
            
        return item

class DataLoader:
    """Data loading utilities."""
    
    def __init__(
        self,
        data_dir: str = "data",
        tokenizer = None,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """Initialize data loader.
        
        Args:
            data_dir: Base data directory
            tokenizer: Tokenizer for text encoding
            batch_size: Batch size for data loading
            max_length: Maximum sequence length
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        
    def load_madar(
        self,
        split: str = 'train'
    ) -> torch.utils.data.DataLoader:
        """Load MADAR corpus data.
        
        Args:
            split: Data split to load ('train', 'val', or 'test')
            
        Returns:
            PyTorch DataLoader
        """
        # Load processed data
        df = pd.read_parquet(self.data_dir / "processed/madar_processed.parquet")
        split_df = df[df['split'] == split]
        
        # Create dataset
        dataset = NarrativeDataset(
            texts=split_df['text'].tolist(),
            dialects=split_df['dialect'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train')
        )
        
    def load_stories(
        self,
        split: str = 'train'
    ) -> torch.utils.data.DataLoader:
        """Load story dataset.
        
        Args:
            split: Data split to load
            
        Returns:
            PyTorch DataLoader
        """
        # Load processed stories
        df = pd.read_parquet(self.data_dir / "processed/stories_processed.parquet")
        split_df = df[df['split'] == split]
        
        # Create dataset
        dataset = NarrativeDataset(
            texts=split_df['text'].tolist(),
            dialects=split_df['dialect'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train')
        )
