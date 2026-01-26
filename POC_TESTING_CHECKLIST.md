# POC Testing Checklist

**Reproduce the Entity Matching POC in 5 Minutes**

Use this checklist to validate your local setup and reproduce the POC results.

---

## ✅ Pre-Flight Check

Before starting:
- [ ] Python 3.9+ installed: `python3 --version`
- [ ] Git installed: `git --version`
- [ ] 2GB free disk space
- [ ] Internet connection for dependencies

---

## 🚀 Installation (3 minutes)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd MET_CapitalIQ_identityReco
```

### Step 2: Create Virtual Environment

```bash
# Create environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate  # Windows

# Verify (should show .venv in path)
which python3
```

**✅ Checkpoint:** Path should include `.venv`

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Verify installation
python3 -c "import torch, sentence_transformers; print('✅ Ready!')"
```

**✅ Checkpoint:** Should print "✅ Ready!" with no errors

---

## 🧪 Run POC Test (30 seconds)

```bash
python3 example.py
```

---

## ✅ Validation Checklist

After running `example.py`, verify these results:

### Console Output Validation

- [ ] Script completed without errors
- [ ] Shows "Loading data... Reference entities: 500"
- [ ] Shows "Source entities: 50"
- [ ] Shows match statistics at the end
- [ ] Shows "Example completed successfully!"

### Performance Metrics

Mark your actual results:

| Metric | Expected | Your Result |
|--------|----------|-------------|
| **Total Entities** | 50 | _____ |
| **Matched** | 47 (94.0%) | _____ |
| **Average Confidence** | ~93.2% | _____ |
| **Exact Matches (Stage 1)** | 18 (36%) | _____ |
| **Vector Search (Stage 2)** | 24 (48%) | _____ |
| **Training Pairs Generated** | 200 | _____ |

### File Generation

- [ ] Directory created: `data/`
- [ ] File created: `data/ditto_training_sample.csv`
- [ ] File size: ~26KB

```bash
# Verify files
ls -la data/
ls -lh data/ditto_training_sample.csv
head -5 data/ditto_training_sample.csv
```

---

## 📊 Expected Full Output

Your output should look like this:

```
================================================================================
Entity Matching for S&P Capital IQ - Quick Example
================================================================================

1. Loading data...
   - Reference entities: 500
   - Source entities: 50

2. Initializing pipeline...

3. Matching single entity...
   Source Entity:
   - Name: Apple Inc.
   - Ticker: AAPL

   Match Result:
   - CIQ ID: IQ24937
   - Confidence: 100.00%
   - Method: exact_match
   - Stage: Stage 1: Exact Match

4. Pipeline Statistics:
   - Total Entities: 50
   - Matched: 47 (94.0%)
   - Avg Confidence: 93.2%

   Matches by Stage:
     exact_match: 18 (36.0%)
     vector_search: 24 (48.0%)
     ditto_matcher: 5 (10.0%)

6. Generating training data for Ditto...
   - Generated 200 training pairs
   - Positive pairs: 100
   - Negative pairs: 100
   - Saved to: data/ditto_training_sample.csv

================================================================================
Example completed successfully!
================================================================================
```

---

## 🎯 What Was Tested?

### ✅ Stage 1: Exact Matching ($0 cost)
- LEI identifier matching
- CUSIP identifier matching
- ISIN identifier matching
- Result: 36% of entities matched instantly

### ✅ Stage 2: Vector Search ($0.0001/entity)
- BGE embedding model loaded
- Semantic similarity matching
- Top-10 candidate retrieval
- Result: 48% of remaining entities matched

### ✅ Training Data Generation
- Positive pairs (matching entities)
- Negative pairs (non-matching entities)
- CSV export for Ditto model training
- Result: 200 training pairs created

---

## 🔧 Troubleshooting

### ❌ Error: "ModuleNotFoundError: No module named 'torch'"

**Fix:**
```bash
# Ensure venv is activated
source .venv/bin/activate

# Reinstall
pip install -r requirements.txt
```

### ❌ Error: "ImportError: No module named 'src.data'"

**Fix:**
```bash
# Must run from project root
cd MET_CapitalIQ_identityReco
pwd  # Should end with MET_CapitalIQ_identityReco

python3 example.py
```

### ❌ Error: "command not found: python3"

**Fix:**
```bash
# Use python instead
python --version  # Check version is 3.9+
python example.py
```

### ⚠️ Warning: Installation takes >10 minutes

**Fix for Mac (Apple Silicon):**
```bash
brew install miniforge
conda create -n entity-match python=3.10
conda activate entity-match
conda install pytorch -c pytorch
pip install -r requirements.txt
```

### ⚠️ Warning: Low memory

**Symptom:** System runs out of RAM during execution

**Fix:**
- Close other applications
- BGE model needs ~1GB RAM
- Consider using smaller model (see documentation)

---

## 🎓 Next Steps After Successful Test

### Option 1: Test with Your Own Data

Modify `example.py`:
```python
my_entities = [
    {"company_name": "Your Company", "ticker": "TICK"},
    {"company_name": "Another Corp", "lei": "YOUR_LEI"}
]

results = pipeline.batch_match(my_entities)
```

### Option 2: Generate More Training Data

```bash
python3 -c "
from src.data.training_generator import TrainingDataGenerator
from src.data.loader import DataLoader

loader = DataLoader()
generator = TrainingDataGenerator()
training_df = generator.generate_from_sp500(
    reference_df=loader.load_reference_data(),
    num_positive_pairs=1000,
    num_negative_pairs=1000
)
training_df.to_csv('data/ditto_training_full.csv', index=False)
print('Generated 2000 training pairs')
"
```

### Option 3: Test with Databricks (Requires Access)

```bash
# Configure Databricks
databricks configure --profile LPT_FREE_EDITION

# Copy environment template
cp .env.example .env
# Edit .env: Add SPARK_CONNECT_CLUSTER_ID

# Test connection
python3 test_spark_connect.py

# Run Spark example
python3 example_spark_connect.py
```

See [TESTING_GUIDE.md](documentation/TESTING_GUIDE.md) for details.

---

## ✅ Success Criteria

Your POC setup is validated when:

- ✅ Installation completed without errors
- ✅ `python3 example.py` runs successfully
- ✅ Match rate achieved: **~94%** (47/50)
- ✅ Training data generated: **200 pairs**
- ✅ File created: `data/ditto_training_sample.csv`
- ✅ No import or module errors

---

## 📚 Documentation Links

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Main project overview |
| [TESTING_GUIDE.md](documentation/TESTING_GUIDE.md) | Comprehensive testing guide |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Production deployment |
| [executive-summary.md](documentation/executive-summary.md) | Business case & ROI |

---

## 📞 Need Help?

1. Check Python version: `python3 --version` (must be 3.9+)
2. Verify venv active: `which python3` (should show .venv)
3. Check installed packages: `pip list | grep -E "(torch|sentence-transformers|pandas)"`
4. Review error logs carefully
5. Consult [TESTING_GUIDE.md](documentation/TESTING_GUIDE.md)

---

**Ready? Run `python3 example.py` and check off the validation items above!**

**Target Performance: 94% Match Rate | 93% Confidence | $0.01/entity**
