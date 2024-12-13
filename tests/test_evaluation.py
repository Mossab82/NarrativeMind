import pytest
from narrative_mind.evaluation import NarrativeEvaluator
from narrative_mind.evaluation.cultural_metrics import CulturalMetrics
from narrative_mind.evaluation.dialect_metrics import DialectMetrics

class TestNarrativeEvaluator:
    @pytest.fixture
    def evaluator(self):
        """Initialize evaluator."""
        return NarrativeEvaluator()
    
    def test_evaluation(self, evaluator):
        """Test overall evaluation."""
        text = """
        كان يا ما كان في قديم الزمان ملك عادل.
        وكان يحب الخير والصدق والأمانة.
        وهكذا تعلم الجميع درساً في الحكمة.
        """
        metrics = evaluator.evaluate(text)
        
        assert isinstance(metrics, dict)
        assert 'cultural_score' in metrics
        assert 'coherence' in metrics
        assert all(0 <= v <= 1 for v in metrics.values())
        
    def test_reference_comparison(self, evaluator):
        """Test comparison with reference."""
        generated = "كان يا ما كان في قديم الزمان"
        reference = "في قديم الزمان كان هناك"
        metrics = evaluator._compare_with_reference(generated, reference)
        
        assert 'bleu' in metrics
        assert 0 <= metrics['bleu'] <= 1

class TestCulturalMetrics:
    @pytest.fixture
    def metrics(self):
        """Initialize cultural metrics."""
        return CulturalMetrics()
    
    def test_pattern_evaluation(self, metrics):
        """Test pattern preservation metrics."""
        text = "كان يا ما كان في قديم الزمان"
        scores = metrics._evaluate_patterns(text)
        
        assert 'pattern_precision' in scores
        assert 'pattern_recall' in scores
        assert 'pattern_f1' in scores
