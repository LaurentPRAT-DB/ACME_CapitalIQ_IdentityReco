"""
Unit tests for entity matching pipeline
"""
import pytest
from src.data.loader import DataLoader
from src.data.preprocessor import EntityPreprocessor
from src.pipeline.exact_match import ExactMatcher
from src.pipeline.hybrid_pipeline import HybridMatchingPipeline


@pytest.fixture
def reference_data():
    """Load sample reference data"""
    loader = DataLoader()
    return loader.load_reference_data()


@pytest.fixture
def source_entities():
    """Load sample source entities"""
    loader = DataLoader()
    return loader.load_sample_entities()


def test_preprocessor():
    """Test entity preprocessor"""
    preprocessor = EntityPreprocessor()

    # Test company name normalization
    assert preprocessor.normalize_company_name("Apple Inc.") == "APPLE"
    assert preprocessor.normalize_company_name("Microsoft Corporation") == "MICROSOFT"
    assert preprocessor.normalize_company_name("Amazon.com, Inc") == "AMAZONCOM"

    # Test ticker normalization
    assert preprocessor.normalize_ticker("aapl") == "AAPL"
    assert preprocessor.normalize_ticker(" msft ") == "MSFT"


def test_exact_matcher(reference_data, source_entities):
    """Test exact matching"""
    matcher = ExactMatcher(reference_data)

    # Test LEI match
    entity = {
        "company_name": "Apple Inc.",
        "lei": "HWUPKR0MPOU8FGXBT394"
    }
    result = matcher.match(entity)
    assert result is not None
    assert result["ciq_id"] == "IQ24937"
    assert result["confidence"] == 1.0
    assert result["match_method"] == "exact_lei"


def test_hybrid_pipeline(reference_data, source_entities):
    """Test hybrid pipeline"""
    pipeline = HybridMatchingPipeline(
        reference_df=reference_data,
        enable_foundation_model=False
    )

    # Test single entity match
    entity = source_entities[0]
    result = pipeline.match(entity)

    assert result is not None
    assert "ciq_id" in result
    assert "confidence" in result
    assert "match_method" in result
    assert "stage" in result


def test_batch_matching(reference_data, source_entities):
    """Test batch matching"""
    pipeline = HybridMatchingPipeline(
        reference_df=reference_data,
        enable_foundation_model=False
    )

    results = pipeline.batch_match(source_entities, show_progress=False)

    assert len(results) == len(source_entities)
    assert all("ciq_id" in r for r in results)
    assert all("confidence" in r for r in results)


def test_pipeline_stats(reference_data, source_entities):
    """Test pipeline statistics calculation"""
    pipeline = HybridMatchingPipeline(
        reference_df=reference_data,
        enable_foundation_model=False
    )

    results = pipeline.batch_match(source_entities, show_progress=False)
    stats = pipeline.get_pipeline_stats(results)

    assert "total_entities" in stats
    assert "matched" in stats
    assert "match_rate" in stats
    assert "stages" in stats
    assert "methods" in stats
