# Open-Source Models for Entity Matching & Reconciliation
## Quick Reference Guide for S&P Capital IQ Identity Reconciliation

**Last Updated:** 2026-01-22

---

## Executive Summary

Based on recent research (2024-2025), **specialized entity matching models significantly outperform general-purpose LLMs** for entity reconciliation tasks. Key findings:

- **Ditto** achieves 96.5% F1 on company matching (vs 85-90% for GPT-4 zero-shot)
- Fine-tuned **Llama 3.1** exceeds GPT-4 by 1-10% F1 with minimal training data
- **GLiNER** outperforms ChatGPT for zero-shot entity extraction at 80x lower cost
- Specialized models are **70-90% cheaper** than commercial LLM APIs

**Recommendation:** Use a hybrid approach combining Ditto (fine-tuned matcher) + BGE embeddings (retrieval) + DBRX (edge cases) for optimal accuracy and cost.

---

## Model Comparison Matrix

| Model | Type | F1 Score | Training Required | Cost/Entity | Best For | Hugging Face / GitHub |
|-------|------|----------|-------------------|-------------|----------|----------------------|
| **Ditto** | Entity Matcher | 96.5% | Yes (500-1K pairs) | $0.001 | Structured entity pair matching | [megagonlabs/ditto](https://github.com/megagonlabs/ditto) |
| **Llama 3.1 Fine-Tuned** | LLM Matcher | 90-95% | Yes (few examples) | $0.01 | Instruction-based matching | Databricks Foundation Models |
| **GLiNER** | Zero-Shot NER | 80-85% | No | $0.0001 | Extract company names from text | [urchade/gliner_medium-v2.1](https://huggingface.co/urchade/gliner_medium-v2.1) |
| **ReLiK** | Entity Linking | 85-90% | Optional | $0.001 | Link entities to knowledge bases | [sapienzanlp/relik-entity-linking-base](https://huggingface.co/sapienzanlp/relik-entity-linking-base) |
| **BGE-Large-EN** | Embeddings | N/A | No | $0 (OSS) | Semantic similarity, candidate retrieval | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| **DBRX Instruct** | Foundation Model | 85-90% | No | $0.05 | Complex reasoning, edge cases | Databricks Foundation Models |
| **GPT-4** | Foundation Model | 88-92% | No | $0.30 | High accuracy, no setup | OpenAI API |
| **DeepMatcher** | Deep Matcher | 80-85% | Yes (large dataset) | $0.001 | Legacy entity matching | [anhaidgroup/deepmatcher](https://github.com/anhaidgroup/deepmatcher) |

---

## Top Models for S&P Capital IQ Reconciliation

### 1. Ditto - Deep Entity Matching (RECOMMENDED)

**Use Case:** Primary matching engine for entity pair classification

**Performance:**
- 96.5% F1 score on company dataset matching (789K vs 412K records)
- Outperforms prior SOTA by up to 29% F1 score
- Superior to GPT-4 zero-shot by 4-8% on structured data

**Technical Details:**
- Based on: BERT, RoBERTa, or DistilBERT
- Approach: Sequence-pair classification
- Training: Fine-tune on 500-1000 labeled entity pairs
- Inference: <10ms per pair

**Key Features:**
- Domain knowledge injection (highlight LEI, ticker, etc.)
- String summarization for long fields
- Data augmentation for small training sets
- Interpretable predictions

**Repository:** [https://github.com/megagonlabs/ditto](https://github.com/megagonlabs/ditto)

**When to Use:**
- You have 500+ labeled entity pairs (or can create them from S&P 500)
- Need highest accuracy (96%+)
- Want cheapest inference cost ($0.001/entity)
- Structured entity records with consistent fields

**Databricks Deployment:**
```python
# Train Ditto locally or on Databricks cluster
# Deploy via MLflow + Model Serving

import mlflow

# Log trained Ditto model
with mlflow.start_run():
    mlflow.pytorch.log_model(ditto_model, "ditto-entity-matcher")

# Deploy to Model Serving
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
w.serving_endpoints.create(
    name="ditto-matcher",
    config={
        "served_models": [{
            "model_name": "ditto-entity-matcher",
            "model_version": "1",
            "scale_to_zero_enabled": True
        }]
    }
)
```

**Training Data Format:**
```csv
left_company,right_company,label
"Apple Inc.","Apple Computer Inc.",1
"Apple Inc.","Microsoft Corp",0
"AAPL","Apple Inc.",1
```

---

### 2. GLiNER - Zero-Shot Entity Recognition

**Use Case:** Extract company names from unstructured text (descriptions, news, documents)

**Performance:**
- Outperforms ChatGPT in zero-shot NER benchmarks
- 80x cheaper than large LLMs
- No training data required

**Technical Details:**
- Based on: Bidirectional transformer encoder
- Approach: Entity-span matching with runtime labels
- Inference: ~100ms per document

**Key Features:**
- True zero-shot (no training needed)
- Specify entity types at runtime: `["company", "financial institution", "organization"]`
- Handles multiple entity types simultaneously
- Lightweight (200-500MB models)

**Models on Hugging Face:**
- `urchade/gliner_medium-v2.1` - General purpose (250M params)
- `numind/NuNER_Zero` - Alternative GLiNER architecture
- `knowledgator/gliner-decoder-large-v1.0` - Larger variant

**Repository:** [https://github.com/urchade/GLiNER](https://github.com/urchade/GLiNER)

**When to Use:**
- Entity names are in unstructured text or descriptions
- No labeled training data available
- Need fast extraction across many documents
- Want to avoid LLM API costs

**Example Usage:**
```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

text = """
Apple Inc. (NASDAQ: AAPL) announced partnership with
Microsoft Corporation. The deal involves Goldman Sachs
as financial advisor.
"""

labels = ["company", "financial institution", "stock exchange"]
entities = model.predict_entities(text, labels)

# Output:
# [
#   {"text": "Apple Inc.", "label": "company", "score": 0.96},
#   {"text": "Microsoft Corporation", "label": "company", "score": 0.94},
#   {"text": "Goldman Sachs", "label": "financial institution", "score": 0.92},
#   {"text": "NASDAQ", "label": "stock exchange", "score": 0.89}
# ]
```

---

### 3. BGE-Large-EN - Semantic Embeddings

**Use Case:** Generate embeddings for semantic similarity search and candidate retrieval

**Performance:**
- State-of-the-art open-source embedding model
- 1024-dimensional vectors
- Superior to OpenAI text-embedding-ada-002 on many benchmarks

**Technical Details:**
- Based on: BERT-large
- Dimensions: 1024
- Max sequence length: 512 tokens
- Cost: Free (open-source)

**Repository:** [https://huggingface.co/BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

**When to Use:**
- Generate embeddings for vector search
- Retrieve top-K candidate matches
- Semantic similarity scoring
- Replace costly OpenAI embeddings API

**Databricks Deployment:**
```python
from sentence_transformers import SentenceTransformer
import mlflow

# Load BGE model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Generate embeddings
entities = ["Apple Inc.", "Microsoft Corp", "Amazon.com Inc."]
embeddings = model.encode(entities)

# Deploy to Model Serving
with mlflow.start_run():
    mlflow.sentence_transformers.log_model(
        model,
        "bge-embeddings",
        signature=mlflow.models.infer_signature(entities, embeddings)
    )
```

---

### 4. Fine-Tuned Llama 3.1 - Instruction-Based Matching

**Use Case:** Entity matching with natural language reasoning, minimal training data

**Performance:**
- Exceeds GPT-4 zero-shot by 1-10% F1 on 4/6 datasets
- Requires only a few training examples (vs thousands for traditional models)
- Competitive with fully fine-tuned BERT models

**Technical Details:**
- Based on: Llama 3.1 70B
- Approach: Instruction tuning for matching tasks
- Training: Few-shot or full fine-tuning
- Inference: Available via Databricks Foundation Models

**When to Use:**
- Limited labeled data (10-100 examples)
- Need natural language explanations for matches
- Want to avoid external APIs (OpenAI/Anthropic)
- Require flexible prompt-based matching

**Example Prompt:**
```python
prompt = f"""
You are an entity matching expert. Determine if these two entities
refer to the same company:

Entity A: {source_entity}
  - Name: Apple Inc.
  - Ticker: AAPL
  - Location: Cupertino, CA
  - Industry: Technology Hardware

Entity B: {candidate_entity}
  - Name: Apple Computer Inc.
  - ID: CIQ12345
  - Country: United States
  - Sector: Consumer Electronics

Return JSON: {{"match": true/false, "confidence": 0-100, "reasoning": "..."}}
"""

response = llama_model(prompt)
```

---

### 5. ReLiK - Entity Linking

**Use Case:** Link extracted entities to S&P Capital IQ identifiers

**Performance:**
- Fast and lightweight retrieval + linking
- Competitive with heavier models

**Technical Details:**
- Approach: Retrieval + ranking for entity disambiguation
- Handles entity variations and aliases

**Repository:** [https://huggingface.co/sapienzanlp/relik-entity-linking-base](https://huggingface.co/sapienzanlp/relik-entity-linking-base)

**When to Use:**
- Map entities to knowledge base IDs
- Disambiguate between multiple candidate matches
- Need fast entity resolution pipeline

---

## Recommended Pipeline Architecture

### Option 1: Ditto-First (Highest Accuracy)

```
Input Entity
    ↓
[1] Exact Match (LEI, CUSIP, ISIN) → 30-40% ✓
    ↓
[2] BGE Embeddings + Vector Search → Top-10 candidates
    ↓
[3] Ditto Fine-Tuned Matcher → Binary classification → 96%+ F1
    ↓ (low confidence <80%)
[4] DBRX Foundation Model → Edge cases only
    ↓
Output: Matched CIQ ID + Confidence
```

**Pros:**
- Highest accuracy (96%+ F1)
- Cheapest inference ($0.01/entity)
- Explainable predictions

**Cons:**
- Requires 500-1000 labeled pairs
- Initial fine-tuning effort (+3 days)

---

### Option 2: Foundation Model-First (Fastest Setup)

```
Input Entity
    ↓
[1] Exact Match → 30-40% ✓
    ↓
[2] BGE Embeddings + Vector Search → Top-10 candidates
    ↓
[3] DBRX Foundation Model → LLM reasoning
    ↓
Output: Matched CIQ ID + Confidence
```

**Pros:**
- No training data required
- Fast setup (no fine-tuning)
- Flexible prompting

**Cons:**
- Lower accuracy (85-90% F1)
- Higher cost ($0.05/entity)
- Less interpretable

---

### Option 3: Hybrid (Recommended)

Combines the best of both:
- Ditto for high-confidence matches (96% of cases)
- Foundation Model for edge cases (4% of cases)
- **Result: 93-95% F1 at $0.01/entity**

---

## Training Data Requirements

| Model | Minimum Data | Optimal Data | Data Type |
|-------|--------------|--------------|-----------|
| Ditto | 500 pairs | 1000-5000 pairs | Labeled entity pairs (match/no-match) |
| Llama 3.1 Fine-Tuned | 10-50 examples | 100-500 examples | Entity pairs with reasoning |
| GLiNER | 0 (zero-shot) | N/A | None required |
| BGE Embeddings | 0 (pre-trained) | N/A | None required |
| DBRX/Foundation | 0 (zero-shot) | 5-10 examples | Few-shot prompt examples |

### Creating Training Data for Ditto

**Approach 1: Use S&P 500 as Gold Standard**
```python
# Extract S&P 500 companies from S&P Capital IQ
spglobal_sp500 = spark.table("reference.spglobal_entities") \
    .filter(col("index") == "S&P 500")

# Create positive pairs (variations of same company)
positive_pairs = []
for company in spglobal_sp500:
    # Pair official name with aliases
    for alias in company.aliases:
        positive_pairs.append((company.name, alias, 1))

    # Pair with ticker
    positive_pairs.append((company.name, company.ticker, 1))

# Create negative pairs (different companies)
negative_pairs = []
companies = list(spglobal_sp500)
for i in range(1000):
    c1, c2 = random.sample(companies, 2)
    negative_pairs.append((c1.name, c2.name, 0))

# Combine and save
training_data = pd.DataFrame(
    positive_pairs + negative_pairs,
    columns=["left_entity", "right_entity", "label"]
)
training_data.to_csv("ditto_training_data.csv", index=False)
```

**Approach 2: Manual Labeling of Difficult Cases**
- Export 500 entity pairs with low embedding similarity
- Manually label match/no-match
- Focus on edge cases (M&A, name changes, subsidiaries)

---

## Cost Comparison

**10,000 Entity Matching Job:**

| Approach | Setup Cost | Inference Cost | Total | Accuracy |
|----------|------------|----------------|-------|----------|
| GPT-4 API | $0 | $3,000 | $3,000 | 88-92% |
| DBRX Foundation Model | $0 | $500 | $500 | 85-90% |
| Ditto Fine-Tuned | $500 | $10 | $510 | 96%+ |
| Hybrid (Ditto + DBRX) | $500 | $100 | $600 | 93-95% |
| GLiNER + BGE + DBRX | $0 | $200 | $200 | 85-90% |

**For 1M entities/year:**
- GPT-4: $300,000/year
- DBRX only: $50,000/year
- Ditto + DBRX: $10,000/year (98% cost reduction vs GPT-4)

---

## Implementation Timeline

### With Ditto Fine-Tuning (6 weeks)

| Week | Tasks |
|------|-------|
| 1 | Setup, data extraction, create training dataset (500-1000 pairs) |
| 2 | Fine-tune Ditto, deploy BGE embeddings, setup Vector Search |
| 3 | Integrate pipeline, test on S&P 500 gold standard |
| 4 | Add Foundation Model fallback, optimize thresholds |
| 5 | Validation, MLflow tracking, performance analysis |
| 6 | Documentation, executive presentation |

### Without Fine-Tuning (5 weeks)

| Week | Tasks |
|------|-------|
| 1 | Setup, data extraction, S&P reference data load |
| 2-3 | Deploy BGE + Vector Search + Foundation Model pipeline |
| 3.5-5 | Testing, tuning, validation |
| 5 | Analysis, documentation |

---

## Key Takeaways

1. **Specialized models >> General LLMs** for entity matching tasks
2. **Ditto achieves 96.5% F1** with minimal training data (500-1000 pairs)
3. **Hybrid approach** (Ditto + Foundation Model) provides best accuracy/cost tradeoff
4. **GLiNER enables zero-shot** company name extraction from unstructured text
5. **Open-source models reduce costs by 70-98%** vs commercial LLM APIs

---

## Sources

- [Entity Matching using Large Language Models (ArXiv 2023, updated 2025)](https://arxiv.org/abs/2310.11244)
- [Deep Entity Matching with Pre-Trained Language Models - Ditto (ArXiv 2020)](https://arxiv.org/abs/2004.00584)
- [Effective entity matching with transformers (VLDB Journal 2023)](https://link.springer.com/article/10.1007/s00778-023-00779-z)
- [Match, Compare, or Select? LLMs for Entity Matching (COLING 2025)](https://aclanthology.org/2025.coling-main.8/)
- [GLiNER: Generalist Model for NER (NAACL 2024)](https://aclanthology.org/2024.naacl-long.300.pdf)
- [CFM Financial Entity Recognition Case Study (Hugging Face 2025)](https://huggingface.co/blog/cfm-case-study)
- [Multi-Agent RAG Framework for Entity Resolution (MDPI Dec 2025)](https://www.mdpi.com/2073-431X/14/12/525)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-22
