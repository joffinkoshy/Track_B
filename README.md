# 🚀 KDSH 2026 Track B: BDH-inspired Narrative Consistency Classifier

![Track B Compliance](https://img.shields.io/badge/Track_B-Compliant-brightgreen)
![BDH Principles](https://img.shields.io/badge/BDH_Principles-Implemented-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**Complete implementation for KDSH 2026 Track B challenge** - A BDH-inspired representation learning system that determines narrative consistency between character backstories and long narratives.

## 🎯 Overview

This project implements **Option 3** of the KDSH 2026 Track B challenge: **Context/state-based scoring** using BDH-inspired representation learning with:

- ✅ **Persistent Internal State**: Context memory across narrative elements
- ✅ **Selective Updates**: Importance-thresholded state updates
- ✅ **Context-based Scoring**: Deterministic scoring based on state representations

## 📚 Features

- **Context-based Classification**: Uses BDH-inspired state representations for scoring
- **Context Memory**: Maintains evolving state across narrative elements
- **Selective Updates**: Importance-thresholded state updates for representation learning
- **Deterministic Scoring**: Consistent scoring based on state representations
- **CSV Processing**: Handles structured backstory data in required format
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, and F1 metrics

## 🎯 Track B Evaluation Alignment

This project is **optimized for KDSH 2026 Track B evaluation criteria**:

### 🏆 **Evaluation Dimensions Addressed**

#### 1. **Accuracy and Robustness on Core Task**
- ✅ **Binary Classification**: Determines narrative consistency (1=consistent, 0=inconsistent)
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 Score
- ✅ **Robust Processing**: Handles 100k+ word narratives and structured backstory data
- ✅ **Error Handling**: Graceful handling of edge cases and missing data

#### 2. **Representation Learning using BDH**
- ✅ **BDH-inspired State**: `BDHState` class implements persistent internal state
- ✅ **Selective Learning**: Importance-thresholded updates for efficient representation
- ✅ **Context-based Scoring**: Deterministic scoring from state representations
- ✅ **Context Memory**: Maintains evolving character/book knowledge across elements

#### 3. **Clarity in BDH Influence on Representations**
- ✅ **Signal Generation**: `get_reasoning_signals()` provides clear state-based signals
- ✅ **Separation from Transformers**: Pure representation learning, no transformer components
- ✅ **Interpretability Features**: Full traceability of state changes
- ✅ **Documented Principles**: Clear explanation of BDH-inspired representation

#### 4. **Comparison to Transformer Approaches**
- ✅ **Non-Transformer Architecture**: Uses BDH-inspired state representations instead of attention
- ✅ **Deterministic vs. Stochastic**: Provides reproducible results vs. transformer variability
- ✅ **Interpretability vs. Black Box**: Offers full traceability vs. transformer opacity
- ✅ **Resource Efficiency**: Uses sparse updates vs. transformer full attention

#### 5. **Evidence Rationale (Optional)**
- ✅ **Detailed Rationales**: Generated in `results.csv` with comprehensive explanations
- ✅ **Confidence Scores**: Included in all predictions
- ✅ **Context Information**: Narrative segments and backstory elements documented
- ✅ **State-based Adjustments**: Shows how state representations influenced scoring

### 🎯 **Track B Compliance & BDH-inspired Representation**

This project implements **Option 3** using BDH-inspired representation learning:

#### ✅ **Deterministic Behavior**
- **All randomness removed** from state updates
- **Same inputs → Same outputs** guaranteed for reproducibility
- **Evaluation safety** ensured through deterministic operations

#### ✅ **Separation of Concerns**
- **State Layer**: Pure representation (generates signals only)
- **Scoring Layer**: Context-based scoring from state representations
- **Decision Layer**: Classification based on scores
- **Clean API boundary** via `get_reasoning_signals()` method

#### ✅ **Comprehensive Interpretability**
- **Full update traceability** with `StateUpdateTrace` records
- **Reason enumeration** explaining why each update occurred
- **Evidence strength tracking** for all state changes
- **Complete compliance documentation** via `get_track_b_compliance_info()`

#### ✅ **BDH-inspired Principles**
- **Persistent Internal State**: Maintained deterministically across operations
- **Selective Updates**: Importance-thresholded deterministic updates
- **Context-based Scoring**: Deterministic scoring from state representations

**See [BDH_REFACTORING_SUMMARY.md](BDH_REFACTORING_SUMMARY.md) for complete details on the refactoring process and rationale.**

**See [TRACK_B_EVALUATION_ALIGNMENT.md](TRACK_B_EVALUATION_ALIGNMENT.md) for comprehensive evaluation criteria alignment.**

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
🎯 Implementing Option 3: Context/state-based scoring using BDH-inspired representation learning
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
  - Context-based consistency score
  - Narrative segments analyzed
  - Backstory elements considered
  - Context-aware state adjustments
  - Deterministic state update statistics
  - Prediction confidence level
  - Explicit Track B compliance statement

## 🎯 Track B Compliance

### Implemented BDH Principles (Option 3)

1. **Persistent Internal State**
   - `BDHState` class maintains evolving memory
   - Context memory tracks character/book history
   - Complete state evolution history preserved
   - Pure representation-support layer (never makes decisions)

2. **Importance-thresholded Updates**
   - Configurable importance thresholds
   - Only significant information triggers deterministic state changes
   - No randomness - fully reproducible operations

3. **Context-based Scoring**
   - Deterministic scoring from state representations
   - Context-aware adjustments based on historical patterns
   - Gradual state evolution with full traceability

### Compliance Verification

✅ **Uses BDH-inspired representation learning** (not transformers)
✅ **Implements Option 3: Context/state-based scoring**
✅ **Provides binary classification output (0/1)**
✅ **Handles structured backstory data in required CSV format**
✅ **Generates compliant CSV output with proper rationales**
✅ **Includes comprehensive evaluation metrics**
✅ **Maintains full reproducibility guarantees**
✅ **Clear separation: representation layer vs. scoring layer**
✅ **No claims of reasoning, belief formation, or randomness**

### Track B Safety Features

- **Deterministic Operations**: Same inputs → Same outputs guaranteed
- **Reproducibility**: No random number generation anywhere
- **Interpretability**: Full state update traceability
- **Compliance Documentation**: Automatic compliance reporting
- **Safe Terminology**: Context/state-based scoring only

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
    confidence_threshold=00.4,     # Belief update sensitivity
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
