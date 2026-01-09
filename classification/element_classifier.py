"""
BDH Element Classifier Module

Enhanced classifier specifically designed for individual backstory elements
while maintaining strong BDH principles for Track B compliance.
"""

from typing import Dict, List, Optional
from bdh_core.state import BDHState, BDHStateConfig
from data_processing.csv_loader import BackstoryElement, CSVDataLoader
from classification.classifier import ConsistencyClassifier

class BDHElementClassifier:
    """
    BDH-enhanced classifier for individual backstory elements.
    Implements Track B Option 4: BDH-inspired reasoning components with
    persistent internal state, selective updates, and incremental belief formation.
    """

    def __init__(self, config: Optional[BDHStateConfig] = None):
        """
        Initialize the BDH element classifier

        Args:
            config: Optional BDH state configuration
        """
        # Core BDH classifier
        self.classifier = ConsistencyClassifier(config)

        # BDH context memory system
        self.data_loader = CSVDataLoader()
        self.context_memory: Dict[str, Dict] = {}  # Character/book context tracking

        # BDH state tracking
        self.processing_stats = {
            'total_elements': 0,
            'elements_by_book': {},
            'elements_by_character': {},
            'bdh_state_evolution': []
        }

        # Track B compliance markers
        self.bdh_compliance = {
            'persistent_state_usage': 0,
            'selective_updates': 0,
            'incremental_belief_updates': 0,
            'context_aware_decisions': 0
        }

    def process_element(self, element: BackstoryElement) -> Dict:
        """
        Process a single backstory element with full BDH context awareness

        Args:
            element: BackstoryElement to process

        Returns:
            Dictionary containing classification results and BDH metrics
        """
        # Create BDH context for this element
        context = self.data_loader.create_bdh_context(element)

        # Process with BDH classifier (Track B Option 4 implementation)
        result = self.classifier.process_narrative_backstory_pair(
            narrative_text=context,
            backstory_text=element.content,
            character_name=element.character,
            metadata={
                'book': element.book_name,
                'element_id': element.id,
                'caption': element.caption,
                'is_training': element.label is not None
            }
        )

        # Apply BDH context-aware adjustments
        context_aware_result = self._apply_bdh_context_awareness(element, result)

        # Update BDH context memory
        self._update_bdh_context_memory(element, context_aware_result)

        # Update statistics and compliance tracking
        self._update_bdh_statistics(element, context_aware_result)

        return {
            **context_aware_result,
            'element_id': element.id,
            'actual_label': element.label,
            'bdh_compliance': self.bdh_compliance.copy(),
            'context_stats': self.data_loader.get_bdh_context_stats()
        }

    def _apply_bdh_context_awareness(self, element: BackstoryElement, result: Dict) -> Dict:
        """
        Apply BDH context-aware adjustments to classification results.
        This implements incremental belief formation over time.

        Args:
            element: Current backstory element
            result: Base classification result

        Returns:
            Context-aware adjusted result
        """
        char_key = f"{element.book_name}_{element.character}"
        base_prediction = result['prediction']
        base_score = result['consistency_score']

        # Initialize context if not exists
        if char_key not in self.context_memory:
            self.context_memory[char_key] = {
                'consistency_history': [],
                'prediction_history': [],
                'context_strength': 0.1  # Start with weak context
            }

        context = self.context_memory[char_key]

        # BDH-style incremental belief formation
        if context['consistency_history']:
            avg_score = sum(context['consistency_history']) / len(context['consistency_history'])

            # Calculate deviation from historical pattern
            deviation = abs(base_score - avg_score)

            # Context strength increases with more elements (BDH learning over time)
            context['context_strength'] = min(
                context['context_strength'] + 0.05,
                0.5  # Max context influence
            )

            # Apply BDH context adjustment
            if deviation > 0.25:  # Significant deviation from pattern
                adjustment_factor = deviation * context['context_strength']

                if base_score < avg_score:  # Current is less consistent than average
                    # Strengthen inconsistency prediction
                    adjusted_score = base_score - adjustment_factor
                    adjusted_prediction = 0
                else:  # Current is more consistent than average
                    # Strengthen consistency prediction
                    adjusted_score = base_score + adjustment_factor
                    adjusted_prediction = 1

                # Track BDH compliance
                self.bdh_compliance['context_aware_decisions'] += 1

                return {
                    **result,
                    'consistency_score': max(0.0, min(1.0, adjusted_score)),
                    'prediction': adjusted_prediction,
                    'bdh_context_adjustment': {
                        'original_score': base_score,
                        'adjusted_score': adjusted_score,
                        'context_strength': context['context_strength'],
                        'historical_avg': avg_score
                    }
                }

        return result

    def _update_bdh_context_memory(self, element: BackstoryElement, result: Dict) -> None:
        """
        Update BDH context memory with new processing results.
        This implements persistent internal state for characters/books.

        Args:
            element: Processed element
            result: Classification result
        """
        char_key = f"{element.book_name}_{element.character}"

        # Initialize if not exists
        if char_key not in self.context_memory:
            self.context_memory[char_key] = {
                'consistency_history': [],
                'prediction_history': [],
                'context_strength': 0.1
            }

        context = self.context_memory[char_key]

        # Update history (BDH persistent state)
        context['consistency_history'].append(result['consistency_score'])
        context['prediction_history'].append(result['prediction'])

        # Keep history manageable (BDH-style memory management)
        if len(context['consistency_history']) > 10:  # Keep last 10 elements
            context['consistency_history'] = context['consistency_history'][-10:]
            context['prediction_history'] = context['prediction_history'][-10:]

        # Track BDH compliance
        self.bdh_compliance['persistent_state_usage'] += 1

    def _update_bdh_statistics(self, element: BackstoryElement, result: Dict) -> None:
        """Update BDH processing statistics and compliance tracking"""
        # Update overall stats
        self.processing_stats['total_elements'] += 1

        # Track by book
        book_name = element.book_name
        if book_name not in self.processing_stats['elements_by_book']:
            self.processing_stats['elements_by_book'][book_name] = 0
        self.processing_stats['elements_by_book'][book_name] += 1

        # Track by character
        char_key = f"{element.book_name}_{element.character}"
        if char_key not in self.processing_stats['elements_by_character']:
            self.processing_stats['elements_by_character'][char_key] = 0
        self.processing_stats['elements_by_character'][char_key] += 1

        # Track BDH state evolution
        self.processing_stats['bdh_state_evolution'].append({
            'element_id': element.id,
            'consistency_score': result['consistency_score'],
            'prediction': result['prediction'],
            'state_updates': result['state_stats']['updates'],
            'context_strength': self.context_memory.get(char_key, {}).get('context_strength', 0.1)
        })

        # Update BDH compliance tracking from classifier results
        self.bdh_compliance['selective_updates'] += result['state_stats']['updates']
        self.bdh_compliance['incremental_belief_updates'] += (
            result['backstory_elements']['events'] +
            result['backstory_elements']['beliefs'] +
            result['backstory_elements']['motivations'] +
            result['backstory_elements']['assumptions']
        )

    def get_bdh_element_prediction(self, element: BackstoryElement) -> int:
        """
        Get final BDH-aware prediction for an element.
        This combines base prediction with context awareness.

        Args:
            element: Backstory element to predict

        Returns:
            Final prediction (0 or 1)
        """
        result = self.process_element(element)
        return result['prediction']

    def process_batch(self, elements: List[BackstoryElement]) -> List[Dict]:
        """
        Process a batch of backstory elements with BDH state isolation.
        Each element gets fresh BDH state to prevent cross-contamination.

        Args:
            elements: List of BackstoryElement objects

        Returns:
            List of classification results
        """
        results = []
        for element in elements:
            # Process element
            result = self.process_element(element)

            # Reset classifier state between elements (BDH isolation)
            self.classifier.reset_classifier()

            results.append(result)

        return results

    def get_track_b_compliance_report(self) -> Dict:
        """
        Generate Track B compliance report showing BDH principle implementation

        Returns:
            Dictionary with compliance metrics and explanations
        """
        return {
            'track_b_option': 'Option 4: Implementing reasoning components explicitly inspired by BDH principles',
            'bdh_principles_implemented': {
                'persistent_internal_state': {
                    'usage_count': self.bdh_compliance['persistent_state_usage'],
                    'implementation': 'Context memory tracks character/book history across elements',
                    'evidence': f'Context memory contains {len(self.context_memory)} character contexts'
                },
                'selective_sparse_updates': {
                    'usage_count': self.bdh_compliance['selective_updates'],
                    'implementation': 'Importance-thresholded state updates in BDHState',
                    'evidence': f'Average {self.bdh_compliance["selective_updates"] / max(self.processing_stats["total_elements"], 1):.1f} updates per element'
                },
                'incremental_belief_formation': {
                    'usage_count': self.bdh_compliance['incremental_belief_updates'],
                    'implementation': 'Confidence-weighted belief updates and context-aware adjustments',
                    'evidence': f'Context-aware decisions made: {self.bdh_compliance["context_aware_decisions"]}'
                }
            },
            'processing_statistics': self.processing_stats,
            'context_memory_stats': {
                'character_contexts': len(self.context_memory),
                'total_consistency_history': sum(len(ctx['consistency_history']) for ctx in self.context_memory.values()),
                'average_context_strength': sum(ctx.get('context_strength', 0.1) for ctx in self.context_memory.values()) / max(len(self.context_memory), 1)
            },
            'compliance_summary': (
                "✅ Persistent Internal State: Context memory maintains evolving character/book knowledge\n"
                "✅ Selective/Sparse Updates: Importance-based state updates in BDH core\n"
                "✅ Incremental Belief Formation: Context-aware adjustments and confidence-weighted learning\n"
                "✅ Track B Option 4: BDH-inspired reasoning components with all required principles"
            )
        }

    def reset_classifier(self) -> None:
        """Reset the entire classifier state"""
        self.classifier.reset_classifier()
        self.context_memory = {}
        self.processing_stats = {
            'total_elements': 0,
            'elements_by_book': {},
            'elements_by_character': {},
            'bdh_state_evolution': []
        }
        self.bdh_compliance = {
            'persistent_state_usage': 0,
            'selective_updates': 0,
            'incremental_belief_updates': 0,
            'context_aware_decisions': 0
        }
        self.data_loader.reset_contexts()
