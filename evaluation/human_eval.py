class HumanEvaluator:
    """Facilitate human evaluation of generated narratives."""
    
    def __init__(self):
        self.evaluation_criteria = {
            'cultural_authenticity': {
                'description': 'How well does the text preserve cultural elements?',
                'scale': [1, 2, 3, 4, 5]
            },
            'narrative_coherence': {
                'description': 'How well does the story flow and maintain consistency?',
                'scale': [1, 2, 3, 4, 5]
            },
            'dialectal_accuracy': {
                'description': 'How accurate is the dialect usage?',
                'scale': [1, 2, 3, 4, 5]
            },
            'overall_quality': {
                'description': 'Overall quality of the generated narrative',
                'scale': [1, 2, 3, 4, 5]
            }
        }
        
    def create_evaluation_form(self, text: str) -> Dict:
        """Create evaluation form for human annotators."""
        return {
            'text': text,
            'criteria': self.evaluation_criteria,
            'annotations': {
                criterion: None for criterion in self.evaluation_criteria
            },
            'comments': ''
        }
        
    def calculate_agreement(self, evaluations: List[Dict]) -> Dict[str, float]:
        """Calculate inter-annotator agreement."""
        agreements = {}
        
        for criterion in self.evaluation_criteria:
            scores = [[eval['annotations'][criterion] for eval in evaluations]]
            agreements[criterion] = cohen_kappa_score(scores[0], scores[1])
            
        return agreements
