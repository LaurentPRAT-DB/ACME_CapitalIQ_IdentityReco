"""
BGE embeddings model for semantic similarity
"""
from __future__ import annotations

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch


class BGEEmbeddings:
    """Generate embeddings using BGE-Large-EN model"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = None
    ):
        """
        Initialize BGE embeddings model

        Args:
            model_name: Hugging Face model name
            device: Device to run model on (cuda/cpu)
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading BGE model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Show progress bar
            normalize: Normalize embeddings to unit length

        Returns:
            Numpy array of embeddings (shape: [n, embedding_dim])
        """
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings

    def encode_entity(self, entity: dict) -> np.ndarray:
        """
        Encode entity dictionary to embedding

        Args:
            entity: Entity dictionary with attributes

        Returns:
            Embedding vector
        """
        # Create text representation
        text_parts = []

        if name := entity.get("company_name"):
            text_parts.append(name)

        if ticker := entity.get("ticker"):
            text_parts.append(ticker)

        if industry := entity.get("industry"):
            text_parts.append(industry)

        if country := entity.get("country"):
            text_parts.append(country)

        text = " ".join(text_parts)
        return self.encode(text)[0]

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarities between query and multiple candidates

        Args:
            query_embedding: Query embedding (1D array)
            candidate_embeddings: Candidate embeddings (2D array)

        Returns:
            Array of similarity scores
        """
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Normalize candidates
        candidate_norms = candidate_embeddings / np.linalg.norm(
            candidate_embeddings,
            axis=1,
            keepdims=True
        )

        # Compute dot products (cosine similarity)
        similarities = np.dot(candidate_norms, query_norm)
        return similarities


def create_embeddings_udf(model_name: str = "BAAI/bge-large-en-v1.5"):
    """
    Create a Pandas UDF for PySpark to generate embeddings

    Args:
        model_name: BGE model name

    Returns:
        Pandas UDF function
    """
    try:
        from pyspark.sql.functions import pandas_udf
        import pandas as pd

        # Initialize model (will be done once per executor)
        embeddings_model = None

        @pandas_udf("array<float>")
        def embed_udf(texts: pd.Series) -> pd.Series:
            nonlocal embeddings_model

            # Lazy initialization
            if embeddings_model is None:
                embeddings_model = BGEEmbeddings(model_name=model_name)

            # Generate embeddings
            embeddings = embeddings_model.encode(
                texts.tolist(),
                show_progress_bar=False
            )

            # Convert to list of lists
            return pd.Series([emb.tolist() for emb in embeddings])

        return embed_udf

    except ImportError:
        raise ImportError("PySpark is required to create UDF")
