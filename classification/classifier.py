"""
Consistency Classifier Module

Main classifier that integrates BDH components for narrative consistency evaluation.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from bdh_core.state import BDHState, BDHStateConfig
from data_processing.narrative import NarrativeProcessor, NarrativeSegment
from data_processing.backstory import BackstoryAnalyzer, CharacterBackstory
from representation.vectorizer import TextVectorizer, SegmentVectorizer

class ConsistencyClassifier:
    """Main classifier that uses BDH representations for consistency evaluation"""

    def __init__(self, config: Optional[BDHStateConfig] = None):
        """
        Initialize the consistency classifier

        Args:
            config: Optional BDH state configuration
        """
        self.bdh_state = BDHState(config or BDHStateConfig())
        self.narrative_processor = NarrativeProcessor()
        self.backstory_analyzer = BackstoryAnalyzer()
        # Initialize vectorizers with dimension matching BDH state
        vector_dim = config.state_dim if config else 128
        self.text_vectorizer = TextVectorizer(vector_dim=vector_dim)
        self.segment_vectorizer = SegmentVectorizer(TextVectorizer(vector_dim=vector_dim))

        # Tracking metrics
        self.processing_stats = {
            'total_segments': 0,
            'total_backstories': 0,
            'classifications': 0
        }

    def process_narrative_backstory_pair(
        self,
        narrative_text: str,
        backstory_text: str,
        character_name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Process a narrative-backstory pair and return consistency assessment

        Args:
            narrative_text: Full narrative text
            backstory_text: Character backstory text
            character_name: Name of the character
            metadata: Optional additional metadata

        Returns:
            Dictionary containing classification results and metrics
        """
        # Reset state for new pair
        self.bdh_state.reset_state()
        self.text_vectorizer.reset_stats()

        # Process narrative
        narrative_segments = self.narrative_processor.segment_narrative(narrative_text)
        self.processing_stats['total_segments'] += len(narrative_segments)

        # Process backstory
        backstory = self.backstory_analyzer.parse_backstory(backstory_text, character_name, metadata)
        self.processing_stats['total_backstories'] += 1

        # Create representations and update BDH state
        self._create_representations(narrative_segments, backstory)

        # Evaluate consistency
        consistency_score = self._evaluate_consistency()

        # Update classification count
        self.processing_stats['classifications'] += 1

        return {
            'consistency_score': consistency_score,
            'prediction': 1 if consistency_score > 0.5 else 0,
            'prediction_label': 'Consistent' if consistency_score > 0.5 else 'Inconsistent',
            'narrative_segments': len(narrative_segments),
            'backstory_elements': self.backstory_analyzer.get_analysis_stats(backstory),
            'state_stats': self.bdh_state.get_state_stats(),
            'vectorizer_stats': self.text_vectorizer.get_vectorizer_stats(),
            'processing_stats': self.processing_stats.copy(),
            'metadata': {
                'character_name': character_name,
                'narrative_length': len(narrative_text),
                'backstory_length': len(backstory_text),
                **backstory.metadata
            }
        }

    def _create_representations(
        self,
        segments: List[NarrativeSegment],
        backstory: CharacterBackstory
    ) -> None:
        """
        Create BDH-style representations from narrative and backstory

        Args:
            segments: List of narrative segments
            backstory: Parsed backstory object
        """
        # Process narrative segments
        for segment in segments:
            # Use segment vectorizer for enhanced representation
            segment_vector = self.segment_vectorizer.segment_to_vector(
                segment.text, segment.features
            )

            # Calculate importance based on segment features
            importance = self._calculate_segment_importance(segment)
            self.bdh_state.selective_update(segment_vector, importance)

            # Update segment stats
            self.segment_vectorizer.update_segment_stats(segment.features)

        # Process backstory elements with different confidence levels
        self._process_backstory_elements(backstory.early_events, confidence=0.7)
        self._process_backstory_elements(backstory.beliefs, confidence=0.6)
        self._process_backstory_elements(backstory.motivations, confidence=0.8)
        self._process_backstory_elements(backstory.assumptions, confidence=0.5)

    def _process_backstory_elements(self, elements: List[str], confidence: float) -> None:
        """Process a list of backstory elements with specified confidence"""
        for element in elements:
            if element.strip():
                element_vector = self.text_vectorizer.text_to_vector(element)
                self.bdh_state.incremental_belief_update(element_vector, confidence)

    def _calculate_segment_importance(self, segment: NarrativeSegment) -> float:
        """
        Calculate importance score for a narrative segment

        Args:
            segment: Narrative segment to evaluate

        Returns:
            Importance score between 0 and 1
        """
        # Base importance from segment features
        base_importance = min(segment.features['unique_words'] / max(segment.features['length'], 1), 1.0)

        # Adjust based on other features
        sentence_factor = min(segment.features['sentence_count'] / 10.0, 1.0)
        character_factor = min(segment.features['character_names'] / 5.0, 1.0)

        # Combine factors with weights
        importance = (base_importance * 0.6 +
                     sentence_factor * 0.2 +
                     character_factor * 0.2)

        return min(max(importance, 0.1), 1.0)  # Ensure within [0.1, 1.0] range

    def _evaluate_consistency(self) -> float:
        """
        Evaluate consistency based on BDH state

        Returns:
            Consistency score between 0 and 1
        """
        state = self.bdh_state.get_current_state()
        stats = self.bdh_state.get_state_stats()

        # Base consistency metric: inverse of state variance (lower variance = more consistent)
        variance = stats['variance']
        consistency = 1.0 - min(variance / 100.0, 1.0)  # Normalize variance

        # Adjust based on update frequency and distribution
        update_ratio = min(stats['updates'] / 50.0, 1.0)
        update_distribution = stats['update_positions'] / len(state)

        # Combine factors
        consistency = (consistency * 0.5 +  # Base variance score
                      update_ratio * 0.3 +   # Update frequency
                      update_distribution * 0.2)  # Update distribution

        # Additional adjustments based on state characteristics
        mean_factor = 1.0 - abs(stats['mean'] - 0.5)  # Prefer states centered around 0.5
        consistency = consistency * 0.7 + mean_factor * 0.3

        return max(0.0, min(1.0, consistency))

    def get_classifier_stats(self) -> dict:
        """Get overall classifier statistics"""
        return {
            **self.processing_stats,
            'current_state': self.bdh_state.get_state_stats(),
            'vectorizer': self.text_vectorizer.get_vectorizer_stats()
        }

    def reset_classifier(self) -> None:
        """Reset classifier state and statistics"""
        self.bdh_state.reset_state()
        self.text_vectorizer.reset_stats()
        self.narrative_processor.reset_stats()
        self.processing_stats = {
            'total_segments': 0,
            'total_backstories': 0,
            'classifications': 0
        }

class ConsistencyEvaluator:
    """Evaluates classifier performance and provides analysis"""

    def __init__(self):
        self.results_history = []
        self.metrics = {
            'total_evaluations': 0,
            'consistent_predictions': 0,
            'inconsistent_predictions': 0,
            'avg_consistency_score': 0.0
        }

    def record_result(self, result: Dict) -> None:
        """Record a classification result for evaluation"""
        self.results_history.append(result)
        self.metrics['total_evaluations'] += 1

        if result['prediction'] == 1:
            self.metrics['consistent_predictions'] += 1
        else:
            self.metrics['inconsistent_predictions'] += 1

        # Update average consistency score
        total_score = sum(r['consistency_score'] for r in self.results_history)
        self.metrics['avg_consistency_score'] = total_score / len(self.results_history)

    def get_evaluation_metrics(self) -> dict:
        """Get evaluation metrics"""
        if self.metrics['total_evaluations'] == 0:
            return {**self.metrics, 'consistency_rate': 0.0}

        consistency_rate = (self.metrics['consistent_predictions'] /
                           self.metrics['total_evaluations'])

        return {
            **self.metrics,
            'consistency_rate': consistency_rate,
            'inconsistency_rate': 1.0 - consistency_rate
        }

    def analyze_results(self) -> dict:
        """Analyze classification results"""
        if not self.results_history:
            return {'status': 'No results to analyze'}

        # Calculate statistics
        consistency_scores = [r['consistency_score'] for r in self.results_history]
        predictions = [r['prediction'] for r in self.results_history]

        return {
            'score_stats': {
                'mean': float(np.mean(consistency_scores)),
                'std': float(np.std(consistency_scores)),
                'min': float(np.min(consistency_scores)),
                'max': float(np.max(consistency_scores)),
                'median': float(np.median(consistency_scores))
            },
            'prediction_distribution': {
                'consistent': sum(predictions),
                'inconsistent': len(predictions) - sum(predictions)
            },
            'result_count': len(self.results_history)
        }

    def reset_evaluator(self) -> None:
        """Reset evaluator state"""
        self.results_history = []
        self.metrics = {
            'total_evaluations': 0,
            'consistent_predictions': 0,
            'inconsistent_predictions': 0,
            'avg_consistency_score': 0.0
        }
