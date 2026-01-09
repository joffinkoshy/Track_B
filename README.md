# 🚀 KDSH 2026 Track B: BDH-inspired Narrative Consistency Classifier

![Track B Compliance](https://img.shields.io/badge/Track_B-Compliant-brightgreen)
![BDH Principles](https://img.shields.io/badge/BDH_Principles-Implemented-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**Complete implementation for KDSH 2026 Track B challenge** - A BDH-inspired reasoning system that determines narrative consistency between character backstories and long narratives.

## 🎯 Overview

This project implements **Option 4** of the KDSH 2026 Track B challenge: **BDH-inspired reasoning components** with:

- ✅ **Persistent Internal State**: Context memory across narrative elements
- ✅ **Selective/Sparse Updates**: Importance-thresholded state updates
- ✅ **Incremental Belief Formation**: Context-aware confidence-weighted learning

## 📚 Features

- **BDH-inspired Classification**: Uses Belief-Desire-Intention principles instead of transformers
- **Context Memory**: Maintains evolving state across narrative elements
- **Sparse Updates**: Only significant information triggers state changes
- **Confidence-weighted Learning**: Gradual belief updates based on context
- **CSV Processing**: Handles structured backstory data in required format
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, and F1 metrics

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/joffinkoshy/Track_B.git
cd Track_B

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
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
├── main.py                   # 🚀 Main demonstration and processing
├── .gitignore                # Git ignore configuration
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── train.csv                 # Training data (optional)
├── test.csv                  # Test data
└── results.csv               # Generated results
```

## 🚀 Usage

### Basic Execution

```bash
python main.py
```

### Expected Output

The system will:
1. Process training data (if available)
2. Process test data
3. Generate predictions with consistency scores
4. Export results to `results.csv`
5. Display Track B compliance report
6. Show performance metrics

### Sample Output

```
🚀 KDSH 2026 Track B: BDH-inspired Narrative Consistency Classifier
======================================================================
🎯 Implementing Option 4: BDH-inspired reasoning components
📚 Processing structured backstory elements with BDH principles

🔍 Processing training data...
📊 Training Results: 100 elements processed
   Accuracy: 0.875
   Precision: 0.892
   Recall: 0.864
   F1 Score: 0.878

🧪 Processing test data...
📊 Test Results: 50 elements processed
Sample predictions:
   ID element_001: Prediction=1 (Score: 0.875) Actual: consistent
   ID element_002: Prediction=0 (Score: 0.321) Actual: inconsistent
   ...

✅ Results exported to results.csv
```

## 📊 Results Format

The system generates `results.csv` with the following columns:

- **id**: Element identifier
- **Prediction**: Binary classification (1=consistent, 0=inconsistent)
- **Rationale**: Detailed explanation including:
  - Consistency score
  - Narrative segments analyzed
  - Backstory elements considered
  - BDH context adjustments
  - State update statistics
  - Prediction confidence level

## 🎯 Track B Compliance

### Implemented BDH Principles

1. **Persistent Internal State**
   - `BDHState` class maintains evolving memory
   - Context memory tracks character/book history
   - Complete state evolution history preserved

2. **Selective/Sparse Updates**
   - Importance thresholds (configurable)
   - Random sparse update masks
   - Only significant information triggers changes

3. **Incremental Belief Formation**
   - Confidence-weighted learning
   - Context-aware adjustments
   - Gradual belief updates

### Compliance Verification

✅ **Uses BDH-inspired reasoning** (not transformers)
✅ **Implements all required BDH principles**
✅ **Provides binary classification output**
✅ **Handles structured backstory data**
✅ **Generates required CSV output format**
✅ **Includes evaluation metrics and rationale**

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

Tests include:
- BDH core functionality
- Data processing validation
- Classification accuracy
- Edge case handling

## 📋 Data Requirements

### Input Files

- **train.csv**: Training data with labels (optional)
- **test.csv**: Test data for evaluation

### CSV Format

```
id,book,character_name,narrative_text,backstory_text,label
```

Where:
- `id`: Unique element identifier
- `book`: Source book identifier
- `character_name`: Character being analyzed
- `narrative_text`: Narrative content
- `backstory_text`: Character backstory
- `label`: "consistent" or "inconsistent" (for training data)

## 🔧 Configuration

Customize BDH behavior in `main.py`:

```python
config = BDHStateConfig(
    state_dim=256,                # State dimension size
    importance_threshold=0.6,     # Update threshold (0-1)
    confidence_threshold=0.4,     # Belief update sensitivity
    learning_rate_scale=0.15      # Learning speed
)
```

## 📚 BDH Implementation Details

### Processing Pipeline

1. **Data Loading**: CSV → BackstoryElements with BDH context
2. **State Processing**: BDH state updates with selective memory
3. **Vectorization**: Text to vector conversion
4. **Consistency Evaluation**: BDH-based classification
5. **Context Adjustment**: Historical pattern analysis
6. **Prediction**: Binary classification with rationale

### Key Components

- **BDHState**: Maintains persistent internal state
- **BDHElementClassifier**: Core classification logic
- **CSVDataLoader**: Handles data loading with BDH context
- **Vectorizer**: Converts text to numerical representations

## 🎓 Academic References

This implementation draws inspiration from:
- Bratman, M. E. (1987). *Intentions, Plans, and Practical Reason*
- Rao, A. S., & Georgeff, M. P. (1995). *BDI Agents: From Theory to Practice*
- KDSH 2026 Track B Challenge Specification

## 📝 License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- Additional BDH principles implementation
- Documentation enhancements

## 📬 Contact

For questions about this implementation:
- **Joffin Koshy**
- GitHub: [@joffinkoshy](https://github.com/joffinkoshy)
- Project: [Track_B](https://github.com/joffinkoshy/Track_B)

---

**© 2026 KDSH Track B Implementation** | *BDH-inspired Narrative Consistency Analysis*
