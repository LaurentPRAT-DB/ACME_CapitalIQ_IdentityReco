"""
Configuration management for entity matching pipeline
"""
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for ML models"""

    # BGE Embeddings
    bge_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    # Ditto Configuration
    ditto_model_path: Optional[str] = None
    ditto_base_model: str = "distilbert-base-uncased"
    ditto_max_length: int = 256
    ditto_batch_size: int = 64

    # Foundation Models
    foundation_model_name: str = "databricks-dbrx-instruct"
    fallback_model_name: str = "databricks-llama-3-1-70b-instruct"
    max_tokens: int = 500
    temperature: float = 0.1

    # Vector Search
    vector_search_top_k: int = 10
    vector_search_endpoint: str = "entity-matching-endpoint"
    vector_search_index: str = "spglobal_embeddings_index"


@dataclass
class PipelineConfig:
    """Configuration for matching pipeline"""

    # Confidence thresholds
    exact_match_threshold: float = 1.0
    ditto_high_confidence_threshold: float = 0.90
    ditto_low_confidence_threshold: float = 0.70
    foundation_model_threshold: float = 0.80

    # Stage enablement
    enable_exact_match: bool = True
    enable_vector_search: bool = True
    enable_ditto: bool = True
    enable_foundation_model: bool = True

    # Performance settings
    batch_size: int = 100
    max_workers: int = 4


@dataclass
class DataConfig:
    """Configuration for data sources"""

    # Paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    # Database tables
    bronze_table: str = "bronze.entities"
    silver_table: str = "silver.entities_normalized"
    gold_table: str = "gold.matched_entities"
    reference_table: str = "reference.spglobal_entities"

    # Training data
    training_data_path: Optional[str] = None
    gold_standard_path: Optional[str] = None

    def __post_init__(self):
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)


@dataclass
class Config:
    """Main configuration object"""

    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()
    data: DataConfig = DataConfig()

    # Environment
    databricks_host: Optional[str] = None
    databricks_token: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config = cls()

        # Load from environment
        config.databricks_host = os.getenv("DATABRICKS_HOST")
        config.databricks_token = os.getenv("DATABRICKS_TOKEN")
        config.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")

        # Model paths
        if model_path := os.getenv("DITTO_MODEL_PATH"):
            config.model.ditto_model_path = model_path

        return config


# Global config instance
config = Config.from_env()
