"""
Track B Prototype: BDH-inspired Narrative Consistency Classifier

Complete implementation for KDSH 2026 Track B challenge.
This system implements Option 4: BDH-inspired reasoning components with
persistent internal state, selective updates, and incremental belief formation.

🎯 TRACK B COMPLIANCE:
✅ Option 4: Implementing reasoning components explicitly inspired by BDH principles
✅ Persistent Internal State: Context memory across elements
✅ Selective/Sparse Updates: Importance-thresholded state updates
✅ Incremental Belief Formation: Context-aware confidence-weighted learning
"""

import csv
from typing import Dict, List, Tuple
from bdh_core.state import BDHStateConfig
from data_processing.csv_loader import CSVDataLoader, BackstoryElement
from classification.element_classifier import BDHElementClassifier

def demonstrate_track_b_compliance() -> None:
    """Demonstrate complete Track B compliance with CSV data processing"""
    print("🚀 KDSH 2026 Track B: BDH-inspired Narrative Consistency Classifier")
    print("=" * 70)
    print("🎯 Implementing Option 4: BDH-inspired reasoning components")
    print("📚 Processing structured backstory elements with BDH principles")
    print()

    # Initialize BDH element classifier with enhanced configuration
    config = BDHStateConfig(
        state_dim=256,  # Larger state for better context tracking
        importance_threshold=0.6,  # More selective updates
        confidence_threshold=0.4,  # More sensitive belief updates
        learning_rate_scale=0.15  # Faster learning
    )

    element_classifier = BDHElementClassifier(config)

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
    for i, result in enumerate(test_results[:5]):  # Show first 5 predictions
        actual_label = result.get('actual_label', 'N/A')
        print(f"   ID {result['metadata']['element_id']}: "
              f"Prediction={result['prediction']} "
              f"(Score: {result['consistency_score']:.3f}) "
              f"Actual: {actual_label}")

    # Export results in required format
    export_results_to_csv(test_results, 'results.csv')
    print(f"\n✅ Results exported to results.csv")

    # Generate and display Track B compliance report
    compliance_report = element_classifier.get_track_b_compliance_report()
    print("\n🎯 TRACK B COMPLIANCE REPORT:")
    print("-" * 50)
    print(f"📋 Option: {compliance_report['track_b_option']}")
    print()

    for principle, details in compliance_report['bdh_principles_implemented'].items():
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

    # Show sample detailed result
    if test_results:
        print("\n🔍 SAMPLE DETAILED ANALYSIS:")
        sample = test_results[0]
        print(f"Element ID: {sample['metadata']['element_id']}")
        print(f"Book: {sample['metadata']['book']}")
        print(f"Character: {sample['metadata']['character_name']}")
        print(f"Consistency Score: {sample['consistency_score']:.3f}")
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

    # Initialize counters
    correct = 0
    total = len(results)
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Calculate metrics
    for result in results:
        prediction = result['prediction']
        actual = 1 if result['actual_label'] == 'consistent' else 0

        if prediction == actual:
            correct += 1

        if actual == 1:  # Actual consistent
            if prediction == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:  # Actual inconsistent
            if prediction == 0:
                true_negatives += 1
            else:
                false_positives += 1

    # Calculate final metrics
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
            # Create rationale explaining the decision
            rationale = f"Consistency score: {result['consistency_score']:.3f}. "
            rationale += f"Analysis based on {result['narrative_segments']} narrative segments "
            rationale += f"and {result['backstory_elements']['total_elements']} backstory elements. "

            # Add BDH-specific rationale
            if 'bdh_context_adjustment' in result:
                adj = result['bdh_context_adjustment']
                rationale += f"BDH context adjustment applied (strength: {adj['context_strength']:.2f}). "

            # Add state information
            rationale += f"BDH state updates: {result['state_stats']['updates']}. "

            # Add prediction confidence
            confidence = "high" if result['consistency_score'] > 0.7 else "medium" if result['consistency_score'] > 0.4 else "low"
            rationale += f"Prediction confidence: {confidence}."

            writer.writerow([
                result['metadata']['element_id'],
                result['prediction'],
                rationale
            ])

def show_project_structure() -> None:
    """Display the complete project structure"""
    print("\n📁 PROJECT STRUCTURE:")
    print("=" * 30)
    print("""
bdh_track_b_prototype/
│
├── bdh_core/
│   └── state.py              # 🧠 BDH state management (persistent internal state)
│
├── data_processing/
│   ├── narrative.py          # 📖 Narrative segmentation
│   ├── backstory.py          # 👤 Backstory parsing
│   └── csv_loader.py         # 📄 CSV data loading with BDH context
│
├── representation/
│   └── vectorizer.py         # 🔢 Text to vector conversion
│
├── classification/
│   ├── classifier.py         # ⚖️ Main BDH classifier
│   └── element_classifier.py # 🎯 Element-specific BDH classifier
│
└── main.py                   # 🚀 Main demonstration and processing
    """)

def show_track_b_implementation() -> None:
    """Explain how we implement Track B requirements"""
    print("\n🎯 TRACK B IMPLEMENTATION EXPLANATION:")
    print("=" * 50)
    print("""
📋 CHALLENGE REQUIREMENTS:
- Determine if character backstories are consistent with long narratives
- Handle 100k+ word novels and structured backstory data
- Provide binary classification (1=consistent, 0=inconsistent)

🎯 OUR TRACK B APPROACH (Option 4):
"Implementing reasoning components explicitly inspired by BDH principles,
such as persistent internal state, selective or sparse updates, and
incremental belief formation over time."

🧠 BDH PRINCIPLES IMPLEMENTED:

1. PERSISTENT INTERNAL STATE:
   - BDHState class maintains evolving memory across elements
   - Context memory tracks character/book history
   - State history preserves complete evolution

2. SELECTIVE/SPARSE UPDATES:
   - Importance thresholds determine what gets remembered
   - Random sparse update masks prevent overwriting
   - Only significant information triggers state changes

3. INCREMENTAL BELIEF FORMATION:
   - Confidence-weighted learning from backstory elements
   - Context-aware adjustments based on historical patterns
   - Gradual belief updates rather than sudden changes

🔧 TECHNICAL IMPLEMENTATION:

- CSV Data → BackstoryElements → BDH Context → Vectorization
- BDH State Processing → Consistency Evaluation → Prediction
- Context Memory → Historical Pattern Analysis → Adjustments

✅ TRACK B COMPLIANCE VERIFICATION:
- ✅ Uses BDH-inspired reasoning (not transformers)
- ✅ Implements all required BDH principles
- ✅ Provides binary classification output
- ✅ Handles structured backstory data
- ✅ Generates required CSV output format
- ✅ Includes evaluation metrics and rationale
    """)

if __name__ == "__main__":
    # Run the complete Track B demonstration
    demonstrate_track_b_compliance()
    show_project_structure()
    show_track_b_implementation()
