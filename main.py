"""
KDSH 2026 Track B: BDH-inspired Narrative Consistency Classifier

Implementation of Option 3: Context/state-based scoring using BDH-inspired
representation learning with persistent internal state and importance-thresholded updates.
"""

import csv
from typing import Dict, List, Tuple
from bdh_core.state import BDHStateConfig
from data_processing.csv_loader import CSVDataLoader, BackstoryElement
from classification.element_classifier import ContextElementClassifier

def demonstrate_track_b_compliance() -> None:
    """Demonstrate Track B compliance with CSV data processing"""
    print("🚀 KDSH 2026 Track B: BDH-inspired Narrative Consistency Classifier")
    print("=" * 70)
    print("🎯 Implementing Option 3: Context/state-based scoring using BDH-inspired representation learning")
    print("📚 Processing structured backstory elements with BDH principles")
    print()

    # Initialize BDH element classifier
    config = BDHStateConfig(
        state_dim=256,
        importance_threshold=0.6,
        confidence_threshold=0.4,
        learning_rate_scale=0.15
    )

    element_classifier = ContextElementClassifier(config)

    # Process training data if available
    train_results = []
    try:
        print("🔍 Processing training data...")
        train_elements = CSVDataLoader().load_csv_data('train.csv', is_training=True)
        train_results = element_classifier.process_batch(train_elements)

        # Evaluate training performance
        train_metrics = evaluate_performance(train_results)
        print(f"📊 Training Results: {len(train_results)} elements processed")
        print(f"   Accuracy: {train_metrics['accuracy']:.3f}")
        print(f"   Precision: {train_metrics['precision']:.3f}")
        print(f"   Recall: {train_metrics['recall']:.3f}")
        print(f"   F1 Score: {train_metrics['f1_score']:.3f}")
        print()

    except FileNotFoundError:
        print("⚠️  Training data not found, skipping training evaluation")
        print()

    # Process test data
    print("🧪 Processing test data...")
    test_elements = CSVDataLoader().load_csv_data('test.csv', is_training=False)
    test_results = element_classifier.process_batch(test_elements)

    print(f"📊 Test Results: {len(test_results)} elements processed")
    print("Sample predictions:")
    for i, result in enumerate(test_results[:5]):
        actual_label = result.get('actual_label', 'N/A')
        print(f"   ID {result['metadata']['element_id']}: "
              f"Prediction={result['prediction']} "
              f"(Score: {result['context_score']:.3f}) "
              f"Actual: {actual_label}")

    # Export results
    export_results_to_csv(test_results, 'results.csv')
    print(f"\n✅ Results exported to results.csv")

    # Display compliance report
    compliance_report = element_classifier.get_track_b_compliance_report()
    print("\n🎯 TRACK B COMPLIANCE REPORT:")
    print("-" * 50)
    print(f"📋 Option: {compliance_report['track_b_option']}")
    print()

    for principle, details in compliance_report['context_principles_implemented'].items():
        print(f"🔹 {principle.replace('_', ' ').title()}:")
        print(f"   Usage: {details['usage_count']} instances")
        print(f"   Implementation: {details['implementation']}")
        print(f"   Evidence: {details['evidence']}")
        print()

    print("📊 CONTEXT MEMORY STATISTICS:")
    ctx_stats = compliance_report['context_memory_stats']
    print(f"   Character contexts: {ctx_stats['character_contexts']}")
    print(f"   Total consistency history: {ctx_stats['total_consistency_history']}")
    print(f"   Average context strength: {ctx_stats['average_context_strength']:.3f}")
    print()

    print("✅ COMPLIANCE SUMMARY:")
    print(compliance_report['compliance_summary'])

    # Show sample analysis
    if test_results:
        print("\n🔍 SAMPLE DETAILED ANALYSIS:")
        sample = test_results[0]
        print(f"Element ID: {sample['metadata']['element_id']}")
        print(f"Book: {sample['metadata']['book']}")
        print(f"Character: {sample['metadata']['character_name']}")
        print(f"Context Score: {sample['context_score']:.3f}")
        print(f"Prediction: {sample['prediction']} ({'Consistent' if sample['prediction'] == 1 else 'Inconsistent'})")
        print(f"Narrative Segments: {sample['narrative_segments']}")
        print(f"Backstory Elements: {sample['backstory_elements']['total_elements']}")
        print(f"BDH State Updates: {sample['state_stats']['updates']}")

        if 'bdh_context_adjustment' in sample:
            adj = sample['bdh_context_adjustment']
            print(f"BDH Context Adjustment: {adj['original_score']:.3f} → {adj['adjusted_score']:.3f}")
            print(f"Context Strength: {adj['context_strength']:.3f}")

def evaluate_performance(results: List[Dict]) -> Dict:
    """
    Evaluate classifier performance on training data with labels

    Args:
        results: List of classification results with actual labels

    Returns:
        Dictionary with performance metrics
    """
    if not results or 'actual_label' not in results[0]:
        return {"status": "No training data provided"}

    correct = 0
    total = len(results)
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for result in results:
        prediction = result['prediction']
        actual = 1 if result['actual_label'] == 'consistent' else 0

        if prediction == actual:
            correct += 1

        if actual == 1:
            if prediction == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if prediction == 0:
                true_negatives += 1
            else:
                false_positives += 1

    accuracy = correct / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_samples': total,
        'correct_predictions': correct
    }

def export_results_to_csv(results: List[Dict], output_path: str) -> None:
    """
    Export results to CSV format required for submission

    Args:
        results: List of classification results
        output_path: Path to output CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'Prediction', 'Rationale'])

        for result in results:
            rationale = f"Context-based consistency score: {result['context_score']:.3f}. "
            rationale += f"Analysis based on {result['narrative_segments']} narrative segments "
            rationale += f"and {result['backstory_elements']['total_elements']} backstory elements. "

            if 'bdh_context_adjustment' in result:
                adj = result['bdh_context_adjustment']
                rationale += f"Context-aware state adjustment applied (strength: {adj['context_strength']:.2f}). "

            rationale += f"Deterministic state updates: {result['state_stats']['updates']}. "

            confidence = "high" if result['context_score'] > 0.7 else "medium" if result['context_score'] > 0.4 else "low"
            rationale += f"Prediction confidence: {confidence}. "

            rationale += "Decision derived from context-based scoring using BDH-inspired representation learning with persistent internal state and importance-thresholded updates."

            writer.writerow([
                result['metadata']['element_id'],
                result['prediction'],
                rationale
            ])


def show_track_b_implementation() -> None:
    """Explain Track B implementation"""
    print("\n🎯 TRACK B IMPLEMENTATION:")
    print("=" * 50)
    print("""
📋 CHALLENGE REQUIREMENTS:
- Determine if character backstories are consistent with long narratives
- Handle 100k+ word novels and structured backstory data
- Provide binary classification (1=consistent, 0=inconsistent)

🎯 OUR APPROACH (Option 3):
"Context/state-based scoring using BDH-inspired representation learning
with persistent internal state and importance-thresholded updates."

🧠 BDH-INSPIRED PRINCIPLES IMPLEMENTED:

1. PERSISTENT INTERNAL STATE:
   - BDHState class maintains evolving memory across elements
   - Context memory tracks character/book history
   - State history preserves complete evolution

2. SELECTIVE UPDATES:
   - Importance thresholds determine what gets remembered
   - Only significant information triggers state changes

3. CONTEXT-BASED SCORING:
   - Deterministic scoring from state representations
   - Context-aware adjustments based on historical patterns
   - Gradual state evolution

🔧 TECHNICAL IMPLEMENTATION:
- CSV Data → BackstoryElements → BDH Context → Vectorization
- BDH State Processing → Consistency Scoring → Prediction
- Context Memory → Historical Pattern Analysis → Score Adjustments

✅ TRACK B COMPLIANCE:
- ✅ Uses BDH-inspired representation learning (not transformers)
- ✅ Implements context/state-based scoring
- ✅ Provides binary classification output
- ✅ Handles structured backstory data
- ✅ Generates required CSV output format
- ✅ Includes evaluation metrics and rationale
    """)

if __name__ == "__main__":
    demonstrate_track_b_compliance()
    show_track_b_implementation()
