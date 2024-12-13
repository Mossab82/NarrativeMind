import pytest
import torch
from narrative_mind.models import NarrativeMind
from narrative_mind.models.pattern_matcher import PatternMatcher
from narrative_mind.models.dialect_adapter import DialectAdapter

class TestNarrativeMind:
    @pytest.fixture
    def model(self):
        """Initialize model for testing."""
        return NarrativeMind(device='cpu')
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.tokenizer is not None
        assert model.model is not None
        assert isinstance(model.lambda_cultural, float)
        
    def test_generate_story(self, model):
        """Test story generation."""
        prompt = "اكتب قصة عن الكرم"
        story = model.generate_story(prompt)
        
        assert isinstance(story, str)
        assert len(story) > 0
        assert "كان" in story
        
    def test_cultural_patterns(self, model):
        """Test cultural pattern matching."""
        text = "كان يا ما كان في قديم الزمان ملك عادل"
        score = model._compute_cultural_score(text)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
    def test_dialect_adaptation(self, model):
        """Test dialect adaptation."""
        text = "كان يا ما كان في قديم الزمان"
        dialect = "egyptian"
        adapted = model._adapt_dialect(text, dialect)
        
        assert isinstance(adapted, str)
        assert adapted != text
        assert "في يوم من الأيام" in adapted

class TestPatternMatcher:
    @pytest.fixture
    def matcher(self):
        """Initialize pattern matcher."""
        return PatternMatcher()
    
    def test_pattern_matching(self, matcher):
        """Test pattern recognition."""
        text = "كان يا ما كان في قديم الزمان"
        matches = matcher.match_patterns(text)
        
        assert isinstance(matches, dict)
        assert len(matches['opening']) > 0
        
    def test_cultural_scoring(self, matcher):
        """Test cultural scoring."""
        text = """
        كان يا ما كان في قديم الزمان ملك عادل.
        وكان يحب الخير والصدق والأمانة.
        وهكذا تعلم الجميع درساً في الحكمة.
        """
        score = matcher.compute_cultural_score(text)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1

class TestDialectAdapter:
    @pytest.fixture
    def adapter(self):
        """Initialize dialect adapter."""
        return DialectAdapter()
    
    def test_dialect_identification(self, adapter):
        """Test dialect identification."""
        text = "كان فيه راجل طيب"
        dialect = adapter.identify_dialect(text)
        
        assert isinstance(dialect, str)
        assert dialect in ['egyptian', 'gulf', 'levantine', 'moroccan']
        
    def test_dialect_adaptation(self, adapter):
        """Test dialect adaptation."""
        text = "في قديم الزمان"
        target = "egyptian"
        adapted, confidence = adapter.adapt_text(text, target)
        
        assert isinstance(adapted, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
