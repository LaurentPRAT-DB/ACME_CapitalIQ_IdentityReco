"""
Databricks Foundation Model embeddings via API
Uses native Databricks embedding endpoints for semantic similarity
"""
from __future__ import annotations

import numpy as np
from typing import List, Union, Optional
import time


class DatabricksEmbeddings:
    """Generate embeddings using Databricks Foundation Model API"""

    def __init__(
        self,
        model_name: str = "databricks-gte-large-en",
        client=None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Databricks embeddings model

        Args:
            model_name: Databricks embedding model name
            client: Databricks WorkspaceClient (if None, will create one)
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize Databricks client
        if client is None:
            from databricks.sdk import WorkspaceClient
            self.client = WorkspaceClient()
        else:
            self.client = client

        # Databricks GTE models have 1024 dimensions
        # Map model names to their embedding dimensions
        self.dimension_map = {
            "databricks-gte-large-en": 1024,
            "databricks-bge-large-en": 1024,
        }

        self.embedding_dim = self.dimension_map.get(model_name, 1024)

        print(f"Using Databricks embedding model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call Databricks Foundation Model API for embeddings

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        for attempt in range(self.max_retries):
            try:
                # Call Foundation Model API using serving endpoints
                response = self.client.serving_endpoints.query(
                    name=self.model_name,
                    inputs=texts
                )

                # Extract embeddings from response
                # Response format: {"predictions": [[emb1], [emb2], ...]}
                if hasattr(response, 'predictions'):
                    embeddings = response.predictions
                elif isinstance(response, dict) and 'predictions' in response:
                    embeddings = response['predictions']
                else:
                    raise ValueError(f"Unexpected response format: {response}")

                return embeddings

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"⚠ API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed to get embeddings after {self.max_retries} attempts: {e}")

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
            batch_size: Batch size for API calls (Databricks may have limits)
            show_progress_bar: Show progress bar (for compatibility, uses print)
            normalize: Normalize embeddings to unit length

        Returns:
            Numpy array of embeddings (shape: [n, embedding_dim])
        """
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            if show_progress_bar:
                batch_num = i // batch_size + 1
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

            # Call API for batch
            batch_embeddings = self._call_api(batch)
            all_embeddings.extend(batch_embeddings)

        # Convert to numpy array
        embeddings = np.array(all_embeddings, dtype=np.float32)

        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

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
