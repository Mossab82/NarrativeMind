  def _evaluate_cultural_elements(self, text: str) -> float:
        """Evaluate preservation of cultural elements.
        
        Returns:
            Score between 0-1 indicating cultural element preservation
        """
        # Cultural elements to check
        elements = {
            "rhetorical_devices": [
                "السجع",      # Rhymed prose
                "الجناس",     # Paronomasia
                "التشبيه",    # Simile
                "الاستعارة"   # Metaphor
            ],
            "value_concepts": [
                "الشرف",      # Honor
                "الكرامة",    # Dignity  
                "الأمانة",    # Honesty
                "الشجاعة"     # Bravery
            ],
            "wisdom_elements": [
                "الحكمة",     # Wisdom
                "الأمثال",    # Proverbs
                "العبر",      # Lessons
                "الدروس"      # Morals
            ]
        }
        
        scores = []
        for category, items in elements.items():
            matches = sum(1 for item in items if item in text.lower())
            category_score = matches / len(items)
            scores.append(category_score)
            
        return np.mean(scores)
    
    def _evaluate_dialect(self, text: str, target_dialect: str) -> float:
        """Evaluate dialect accuracy.
        
        Args:
            text: Input text
            target_dialect: Expected dialect
            
        Returns:
            Score between 0-1 indicating dialect accuracy
        """
        # Use CAMeL Tools for dialect identification
        predictions = self.dialect_identifier.predict(text.split('\n'))
        
        # Calculate proportion of sentences matching target dialect
        correct = sum(1 for pred in predictions if pred.top == target_dialect)
        return correct / len(predictions)
    
    def _evaluate_coherence(self, text: str) -> float:
        """Evaluate narrative coherence.
        
        Returns:
            Score between 0-1 indicating coherence
        """
        # Split into sentences
        sentences = text.split('.')
        
        # Check for narrative flow indicators
        indicators = {
            'temporal_flow': self._check_temporal_flow(sentences),
            'character_consistency': self._check_character_consistency(sentences),
            'theme_coherence': self._check_theme_coherence(sentences),
            'structural_completeness': self._check_structural_completeness(text)
        }
        
        return np.mean(list(indicators.values()))
    
    def _compare_with_reference(self, generated: str, reference: str) -> Dict[str, float]:
        """Compare generated text with reference.
        
        Returns:
            Dictionary of comparison metrics
        """
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.translate.meteor_score import meteor_score
        
        metrics = {}
        
        # BLEU score
        reference_tokens = reference.split()
        generated_tokens = generated.split()
        metrics['bleu'] = sentence_bleu([reference_tokens], generated_tokens)
        
        # Structural similarity
        metrics['structural_similarity'] = self._compute_structural_similarity(
            generated, reference)
            
        return metrics
    
    def _check_temporal_flow(self, sentences: List[str]) -> float:
        """Check for proper temporal progression."""
        temporal_markers = [
            'وبعد ذلك', 'ثم', 'فيما بعد', 'وفي يوم', 
            'لاحقاً', 'وعندما', 'حينها', 'في النهاية'
        ]
        
        matches = sum(1 for s in sentences 
                     if any(marker in s for marker in temporal_markers))
        return min(matches / len(sentences), 1.0)
    
    def _check_character_consistency(self, sentences: List[str]) -> float:
        """Check for character name and reference consistency."""
        # Extract character mentions using morphological analysis
        char_mentions = []
        for sent in sentences:
            analyses = self.morph.analyze(sent)
            for analysis in analyses:
                if analysis.get('pos') == 'NOUN' and analysis.get('person'):
                    char_mentions.append(analysis.get('lex'))
                    
        if not char_mentions:
            return 0.0
            
        # Check consistency of character references
        unique_chars = set(char_mentions)
        consistency = len(char_mentions) / (len(unique_chars) * len(sentences))
        return min(consistency, 1.0)
    
    def _check_theme_coherence(self, sentences: List[str]) -> float:
        """Check for thematic consistency."""
        # Simplified theme checking using key term repetition
        all_words = ' '.join(sentences).split()
        unique_words = set(all_words)
        
        # Calculate term frequency for theme significance
        theme_terms = {}
        for word in all_words:
            if len(word) > 3:  # Filter out short words
                theme_terms[word] = theme_terms.get(word, 0) + 1
                
        # Get top theme terms
        sorted_terms = sorted(theme_terms.items(), key=lambda x: x[1], reverse=True)
        top_themes = sorted_terms[:5]
        
        # Calculate theme coherence score
        total_terms = sum(freq for _, freq in top_themes)
        coherence = total_terms / len(all_words)
        return min(coherence, 1.0)
    
    def _check_structural_completeness(self, text: str) -> float:
        """Check for complete narrative structure."""
        required_elements = {
            'opening': r'كان يا ما كان|في قديم الزمان|يحكى أن',
            'conflict': r'وفجأة|ولكن|غير أن|حدث أن',
            'resolution': r'وهكذا|وفي النهاية|وأخيراً',
            'moral': r'وتعلم|والعبرة|والحكمة'
        }
        
        score = 0
        for element, pattern in required_elements.items():
            if re.search(pattern, text):
                score += 0.25
                
        return score
