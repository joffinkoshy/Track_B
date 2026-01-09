"""
Classification Module Tests

Tests for classification components including BDH element classifier.
"""

import pytest
import numpy as np
from bdh_core.state import BDHStateConfig
from data_processing.csv_loader import BackstoryElement
from classification.classifier import ContextBasedClassifier
from classification.element_classifier import ContextElementClassifier

class TestContextBasedClassifier:
    """Test the main context-based classifier"""

    def test_initialization(self):
        """Test classifier initialization"""
        config = BDHStateConfig(state_dim=128)
        classifier = ContextBasedClassifier(config)

        assert classifier.bdh_state is not None
        assert classifier.narrative_processor is not None
        assert classifier.backstory_analyzer is not None
        assert classifier.text_vectorizer is not None

    def test_processing_pipeline(self):
        """Test complete processing pipeline"""
        classifier = ContextBasedClassifier()

        # Test data
        narrative = "This is a test narrative about a character's journey."
        backstory = "The character grew up in a small village and dreamed of adventure."
        character_name = "TestCharacter"

        # Process
        result = classifier.process_context_pair(
            narrative, backstory, character_name
        )

        # Verify result structure
        assert 'context_score' in result
        assert 'prediction' in result
        assert 'prediction_label' in result
        assert 'narrative_segments' in result
        assert 'backstory_elements' in result
        assert 'state_stats' in result

        # Verify prediction is binary
        assert result['prediction'] in [0, 1]

    def test_state_reset(self):
        """Test classifier state reset"""
        classifier = ContextBasedClassifier()

        # Process something to change state
        classifier.process_context_pair(
            "Test narrative", "Test backstory", "TestChar"
        )

        # Reset and verify
        classifier.reset_classifier()
        stats = classifier.get_classifier_stats()
        assert stats['classifications'] == 0

class TestContextElementClassifier:
    """Test context-based element classifier"""

    def test_initialization(self):
        """Test element classifier initialization"""
        config = BDHStateConfig(state_dim=256)
        classifier = ContextElementClassifier(config)

        assert classifier.classifier is not None
        assert classifier.data_loader is not None
        assert classifier.context_memory == {}

    def test_element_processing(self):
        """Test individual element processing"""
        classifier = ContextElementClassifier()

        # Create test element
        element = BackstoryElement(
            id="test1",
            book_name="TestBook",
            character="TestChar",
            caption="Test caption",
            content="This character believed in justice and wanted to help others.",
            label="consistent"
        )

        # Process element
        result = classifier.process_element(element)

        # Verify result
        assert 'context_score' in result
        assert 'prediction' in result
        assert 'element_id' in result
        assert result['element_id'] == "test1"
        assert 'actual_label' in result
        assert result['actual_label'] == "consistent"

    def test_context_awareness(self):
        """Test context awareness functionality"""
        classifier = ContextElementClassifier()

        # Create related elements
        element1 = BackstoryElement(
            id="1", book_name="Book1", character="Char1",
            caption="Childhood", content="Character had a happy childhood.", label="consistent"
        )

        element2 = BackstoryElement(
            id="2", book_name="Book1", character="Char1",
            caption="Adulthood", content="Character became a hero later in life.", label="consistent"
        )

        # Process first element
        result1 = classifier.process_element(element1)

        # Process second element (should use context from first)
        result2 = classifier.process_element(element2)

        # Verify context memory was used
        assert len(classifier.context_memory) > 0
        char_key = f"Book1_Char1"
        assert char_key in classifier.context_memory

        # Verify context statistics
        context = classifier.context_memory[char_key]
        assert len(context['consistency_history']) == 2
        assert len(context['prediction_history']) == 2

    def test_batch_processing(self):
        """Test batch processing of elements"""
        classifier = ContextElementClassifier()

        # Create test elements
        elements = [
            BackstoryElement(
                id=f"test{i}",
                book_name="TestBook",
                character="TestChar",
                caption=f"Caption {i}",
                content=f"Content about character's life stage {i}.",
                label="consistent"
            ) for i in range(3)
        ]

        # Process batch
        results = classifier.process_batch(elements)

        assert len(results) == 3
        assert all('context_score' in result for result in results)
        assert all('prediction' in result for result in results)

    def test_compliance_reporting(self):
        """Test Track B compliance reporting"""
        classifier = ContextElementClassifier()

        # Process some elements to generate compliance data
        element = BackstoryElement(
            id="1", book_name="Book1", character="Char1",
            caption="Test", content="Test content.", label="consistent"
        )
        classifier.process_element(element)

        # Get compliance report
        report = classifier.get_track_b_compliance_report()

        # Verify report structure
        assert 'track_b_option' in report
        assert 'context_principles_implemented' in report
        assert 'processing_statistics' in report
        assert 'context_memory_stats' in report
        assert 'compliance_summary' in report

        # Verify compliance claims
        assert "Option 3" in report['track_b_option']
        assert "Persistent Internal State" in report['compliance_summary']
        assert "Importance-thresholded Updates" in report['compliance_summary']
        assert "Context-based Scoring" in report['compliance_summary']

class TestClassificationEdgeCases:
    """Test edge cases in classification"""

    def test_empty_content(self):
        """Test handling of empty content"""
        classifier = ContextBasedClassifier()

        result = classifier.process_context_pair(
            "Narrative text", "", "TestChar"
        )

        assert 'context_score' in result
        assert result['prediction'] in [0, 1]  # Should still produce valid prediction

    def test_very_short_texts(self):
        """Test handling of very short texts"""
        classifier = ContextBasedClassifier()

        result = classifier.process_context_pair(
            "Short", "Text", "Char"
        )

        assert 'context_score' in result
        assert result['prediction'] in [0, 1]

    def test_special_characters(self):
        """Test handling of special characters"""
        classifier = ContextBasedClassifier()

        result = classifier.process_context_pair(
            "Text with !@#$%^&*() characters",
            "More special chars: {}[]|\\;:'\",./<>?",
            "Char"
        )

        assert 'context_score' in result
        assert result['prediction'] in [0, 1]
