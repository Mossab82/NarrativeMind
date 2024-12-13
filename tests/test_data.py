import pytest
import pandas as pd
from narrative_mind.data import MADARProcessor, DataLoader
from pathlib import Path

class TestMADARProcessor:
    @pytest.fixture
    def processor(self):
        """Initialize MADAR processor."""
        return MADARProcessor()
    
    def test_preprocessing(self, processor):
        """Test text preprocessing."""
        text = "كان   يا ما   كان"
        processed = processor._preprocess_text(text)
        
        assert isinstance(processed, str)
        assert "  " not in processed  # No double spaces
        
    def test_parallel_corpus(self, processor, tmp_path):
        """Test MADAR corpus processing."""
        # Create sample data
        sample_data = pd.DataFrame({
            'id': range(10),
            'egyptian': ['text'] * 10,
            'gulf': ['text'] * 10
        })
        path = tmp_path / "sample.csv"
        sample_data.to_csv(path, sep='\t', index=False)
        
        df = processor.process_parallel_corpus(path)
        assert isinstance(df, pd.DataFrame)
        assert 'dialect' in df.columns
        assert 'split' in df.columns
