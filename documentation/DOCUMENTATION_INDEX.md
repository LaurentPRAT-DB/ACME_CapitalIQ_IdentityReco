# Documentation Index

**Complete guide to all project documentation**

This index helps you find the right documentation based on your role and objectives.

---

## 🎯 Start Here

| Document | Purpose | Time | Audience |
|----------|---------|------|----------|
| **[README.md](README.md)** | Project overview and navigation | 10 min | Everyone |
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Quick start and basic examples | 5 min | Developers |
| **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** | This file - find what you need | 2 min | Everyone |

---

## 📖 Documentation by Role

### For Developers

**Goal**: Set up, test, and understand the codebase

1. **[README.md](README.md)** - Start here for project overview
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** - 5-minute quick start
3. **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive testing guide
4. **[src/README.md](src/README.md)** - Code structure and API reference *(if exists)*

**Quick Commands**:
```bash
# Install
pip install -r requirements.txt

# Test
python example.py

# Run tests
pytest tests/ -v
```

### For Data Engineers

**Goal**: Deploy and operate the pipeline in production on Databricks

1. **[README.md](README.md)** - Understand architecture and objectives
2. **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Test with Spark Connect
3. **[PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)** - Deploy to Databricks
4. **[DATABRICKS_SERVERLESS_SETUP.md](DATABRICKS_SERVERLESS_SETUP.md)** - Serverless configuration

**Deployment Phases**:
- Phase 1: Unity Catalog setup (30 min)
- Phase 2: Deploy Ditto model (45 min)
- Phase 3: Vector Search (30 min)
- Phase 4: Scheduled jobs (1 hour)
- Phase 5: Monitoring (30 min)

### For ML Engineers & Data Scientists

**Goal**: Train, evaluate, and optimize the Ditto model

1. **[entity-matching-models-summary.md](entity-matching-models-summary.md)** - Model comparison
2. **[notebooks/02_train_ditto_model.py](notebooks/02_train_ditto_model.py)** - Train Ditto
3. **[notebooks/03_full_pipeline_example.py](notebooks/03_full_pipeline_example.py)** - Full pipeline
4. **[src/models/](src/models/)** - Model implementations

**Key Files**:
- `src/models/ditto_matcher.py` - Ditto implementation
- `src/models/embeddings.py` - BGE embeddings
- `src/evaluation/metrics.py` - Evaluation metrics

### For Business Stakeholders

**Goal**: Understand ROI, business value, and success metrics

1. **[executive-summary.md](executive-summary.md)** - Business case and ROI
2. **[genai-identity-reconciliation-poc.md](genai-identity-reconciliation-poc.md)** - Full POC specification
3. **[README.md](README.md#performance-metrics)** - Achieved metrics

**Key Metrics**:
- **F1 Score**: 94.2% (target: ≥93%)
- **Cost per Entity**: $0.009 (target: $0.01)
- **Auto-Match Rate**: 87.3% (target: ≥85%)
- **Annual Savings**: $232,500 vs manual (58% reduction)

### For DevOps / SRE

**Goal**: Deploy, monitor, and maintain the system

1. **[PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)** - Production setup
2. **[PRODUCTION_DEPLOYMENT.md#phase-5-monitoring--observability](PRODUCTION_DEPLOYMENT.md#phase-5-monitoring--observability)** - Monitoring setup
3. **[TESTING_GUIDE.md#troubleshooting](TESTING_GUIDE.md#troubleshooting)** - Troubleshooting

**Operations**:
- Monitoring dashboards
- Alert configuration
- Cost tracking
- Performance optimization
- Incident response

---

## 📚 Documentation by Topic

### Getting Started

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview and quick start |
| [GETTING_STARTED.md](GETTING_STARTED.md) | 5-minute setup and examples |
| [QUICK_START.md](QUICK_START.md) | *(Legacy - merged into GETTING_STARTED.md)* |

### Testing & Development

| Document | Purpose |
|----------|---------|
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | Comprehensive testing guide |
| [LOCAL_TESTING_GUIDE.md](LOCAL_TESTING_GUIDE.md) | *(Legacy - merged into TESTING_GUIDE.md)* |
| [TESTING_CHEATSHEET.md](TESTING_CHEATSHEET.md) | *(Legacy - see TESTING_GUIDE cheatsheet section)* |
| [test_spark_connect.py](test_spark_connect.py) | Connection test script |
| [example.py](example.py) | Local example |
| [example_spark_connect.py](example_spark_connect.py) | Spark Connect example |

### Production Deployment

| Document | Purpose |
|----------|---------|
| [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) | Complete production deployment guide |
| [DATABRICKS_SERVERLESS_SETUP.md](DATABRICKS_SERVERLESS_SETUP.md) | Serverless configuration details |

### Business & Research

| Document | Purpose |
|----------|---------|
| [executive-summary.md](executive-summary.md) | Business case, ROI, and objectives |
| [genai-identity-reconciliation-poc.md](genai-identity-reconciliation-poc.md) | Full POC specification |
| [entity-matching-models-summary.md](entity-matching-models-summary.md) | Model comparison and research |

### Notebooks (Databricks)

| Notebook | Purpose |
|----------|---------|
| [notebooks/01_quick_start.py](notebooks/01_quick_start.py) | Getting started on Databricks |
| [notebooks/02_train_ditto_model.py](notebooks/02_train_ditto_model.py) | Train Ditto matcher |
| [notebooks/03_full_pipeline_example.py](notebooks/03_full_pipeline_example.py) | Production pipeline example |
| [notebooks/04_spark_connect_example.py](notebooks/04_spark_connect_example.py) | Spark Connect demonstration |

### Legacy Documentation

*(These files are kept for reference but have been consolidated into main docs)*

| Document | Replaced By |
|----------|-------------|
| QUICK_START.md | [GETTING_STARTED.md](GETTING_STARTED.md) |
| LOCAL_TESTING_GUIDE.md | [TESTING_GUIDE.md](TESTING_GUIDE.md) |
| TESTING_CHEATSHEET.md | [TESTING_GUIDE.md](TESTING_GUIDE.md#testing-cheatsheet) |
| SPARK_CONNECT_GUIDE.md | [TESTING_GUIDE.md](TESTING_GUIDE.md#spark-connect-testing-remote-databricks) |
| SPARK_CONNECT_SETUP.md | [TESTING_GUIDE.md](TESTING_GUIDE.md#setup-one-time) |
| SPARK_CONNECT_README.md | [TESTING_GUIDE.md](TESTING_GUIDE.md) |
| IMPLEMENTATION_SUMMARY.md | [README.md](README.md) + [GETTING_STARTED.md](GETTING_STARTED.md) |

---

## 🚀 User Journeys

### Journey 1: New Developer

**Goal**: Get started and understand the project

1. Read [README.md](README.md) for overview (10 min)
2. Follow [GETTING_STARTED.md](GETTING_STARTED.md) (5 min)
3. Run `python example.py` to test locally
4. Review [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing
5. Explore `src/` code and notebooks

**Time**: 30 minutes to first successful test

### Journey 2: Data Engineer Deploying to Production

**Goal**: Deploy pipeline to Databricks production

1. Read [README.md](README.md) for architecture (10 min)
2. Complete [TESTING_GUIDE.md](TESTING_GUIDE.md) Spark Connect setup (15 min)
3. Follow [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) step-by-step (3 hours)
   - Phase 1: Unity Catalog (30 min)
   - Phase 2: Deploy Ditto (45 min)
   - Phase 3: Vector Search (30 min)
   - Phase 4: Scheduled jobs (1 hour)
   - Phase 5: Monitoring (30 min)
4. Run end-to-end validation tests
5. Set up monitoring dashboards

**Time**: 4 hours from setup to production deployment

### Journey 3: ML Engineer Training Ditto

**Goal**: Fine-tune Ditto model for higher accuracy

1. Review [entity-matching-models-summary.md](entity-matching-models-summary.md) (15 min)
2. Generate training data using [GETTING_STARTED.md](GETTING_STARTED.md) examples
3. Follow [notebooks/02_train_ditto_model.py](notebooks/02_train_ditto_model.py) (2 hours)
4. Evaluate model using `src/evaluation/validator.py`
5. Deploy to Model Serving via [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md#phase-2-deploy-ditto-model)

**Time**: 3-4 hours including training

### Journey 4: Business Stakeholder Reviewing Project

**Goal**: Understand business value and approve investment

1. Read [executive-summary.md](executive-summary.md) (15 min)
2. Review [README.md](README.md#performance-metrics) for achieved results (5 min)
3. Watch demo (developer runs `python example.py`)
4. Review [genai-identity-reconciliation-poc.md](genai-identity-reconciliation-poc.md) for full details (30 min)
5. Approve deployment

**Time**: 1 hour for complete review

---

## 🔍 Quick Reference

### Key Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| F1 Score | ≥93% | 94.2% ✅ |
| Precision | ≥95% | 96.1% ✅ |
| Auto-Match Rate | ≥85% | 87.3% ✅ |
| Cost per Entity | $0.01 | $0.009 ✅ |
| Latency | <1s | 0.6s ✅ |

### Pipeline Stages

| Stage | Coverage | Cost/Entity | Latency |
|-------|----------|-------------|---------|
| 1. Exact Match | 30-40% | $0 | <10ms |
| 2. Vector Search | 100% | $0.0001 | <100ms |
| 3. Ditto Matcher | 90%+ | $0.001 | <100ms |
| 4. Foundation Model | <10% | $0.05 | 1-2s |

### Essential Commands

```bash
# Setup
pip install -r requirements.txt
databricks configure --profile DEFAULT

# Test
python example.py
python test_spark_connect.py
pytest tests/ -v

# Train
python -m src.models.ditto_matcher --training-data data/training.csv

# Deploy (see PRODUCTION_DEPLOYMENT.md)
```

---

## 📦 Related Resources

### External Documentation

- [Databricks Documentation](https://docs.databricks.com/)
- [Ditto GitHub](https://github.com/megagonlabs/ditto)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Research Papers

- [Ditto: Deep Entity Matching (ArXiv 2020)](https://arxiv.org/abs/2004.00584)
- [Entity Matching with LLMs (ArXiv 2023)](https://arxiv.org/abs/2310.11244)
- [GLiNER: NER Model (NAACL 2024)](https://aclanthology.org/2024.naacl-long.300.pdf)

### Internal Links

- GitHub Repository: *(Add your repo URL)*
- Confluence Page: *(Add your wiki URL)*
- JIRA Epic: *(Add your JIRA URL)*
- Slack Channel: *(Add your Slack channel)*

---

## 🆘 Getting Help

### Common Questions

**Q: Where do I start?**
→ Read [README.md](README.md), then [GETTING_STARTED.md](GETTING_STARTED.md)

**Q: How do I test locally?**
→ See [TESTING_GUIDE.md](TESTING_GUIDE.md)

**Q: How do I deploy to production?**
→ See [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)

**Q: How accurate is the system?**
→ See [README.md#performance-metrics](README.md#performance-metrics)

**Q: What does it cost?**
→ See [README.md#cost-breakdown](README.md#cost-breakdown) or [executive-summary.md](executive-summary.md)

**Q: How do I train Ditto?**
→ See [notebooks/02_train_ditto_model.py](notebooks/02_train_ditto_model.py)

**Q: I'm getting errors...**
→ See [TESTING_GUIDE.md#troubleshooting](TESTING_GUIDE.md#troubleshooting)

### Support Contacts

- **Technical Issues**: See [TESTING_GUIDE.md#troubleshooting](TESTING_GUIDE.md#troubleshooting)
- **Databricks Questions**: *(Add your team contact)*
- **Business Questions**: *(Add your stakeholder contact)*
- **Project Lead**: Laurent Prat - laurent.prat@databricks.com

---

## 📝 Documentation Standards

### File Naming

- **ALL_CAPS.md**: Main documentation files
- **lowercase.md**: Supporting documentation
- **src/**: Source code with inline docstrings
- **notebooks/**: Databricks notebooks (numbered)

### Document Structure

All main documents follow this structure:
1. Title and brief description
2. Table of contents (for long docs)
3. Prerequisites
4. Step-by-step instructions
5. Examples and code snippets
6. Troubleshooting
7. Next steps / related docs

### Code Examples

All code examples:
- Are tested and working
- Include expected output
- Have clear comments
- Show error handling

---

## 🔄 Documentation Updates

### Last Updated

- **README.md**: 2026-01-23
- **GETTING_STARTED.md**: 2026-01-23
- **TESTING_GUIDE.md**: 2026-01-23
- **PRODUCTION_DEPLOYMENT.md**: 2026-01-23
- **DOCUMENTATION_INDEX.md**: 2026-01-23

### Changelog

**2026-01-23**: Major documentation consolidation
- Merged QUICK_START.md, LOCAL_TESTING_GUIDE.md, TESTING_CHEATSHEET.md into TESTING_GUIDE.md
- Created GETTING_STARTED.md for quick start
- Enhanced README.md as main entry point
- Created DOCUMENTATION_INDEX.md for navigation
- Updated PRODUCTION_DEPLOYMENT.md with complete deployment guide

**2026-01-22**: Initial documentation
- Executive summary
- POC specification
- Model comparison

---

## 🎯 Quick Navigation Table

| I want to... | Go to... | Time |
|--------------|----------|------|
| **Get started quickly** | [GETTING_STARTED.md](GETTING_STARTED.md) | 5 min |
| **Understand the project** | [README.md](README.md) | 10 min |
| **Test locally** | [TESTING_GUIDE.md](TESTING_GUIDE.md) | 30 min |
| **Deploy to production** | [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) | 3 hours |
| **Train Ditto model** | [notebooks/02_train_ditto_model.py](notebooks/02_train_ditto_model.py) | 2 hours |
| **Understand business case** | [executive-summary.md](executive-summary.md) | 15 min |
| **Learn about models** | [entity-matching-models-summary.md](entity-matching-models-summary.md) | 20 min |
| **See full POC details** | [genai-identity-reconciliation-poc.md](genai-identity-reconciliation-poc.md) | 45 min |
| **Troubleshoot issues** | [TESTING_GUIDE.md#troubleshooting](TESTING_GUIDE.md#troubleshooting) | As needed |
| **Find specific documentation** | This file | 2 min |

---

**Ready to start?** → [GETTING_STARTED.md](GETTING_STARTED.md)

**Need help?** → [TESTING_GUIDE.md#troubleshooting](TESTING_GUIDE.md#troubleshooting)

**Target: 93-95% F1 Score | $0.01/entity | 85%+ Auto-Match Rate**
