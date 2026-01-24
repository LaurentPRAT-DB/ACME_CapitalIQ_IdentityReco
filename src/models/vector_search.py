"""
Vector search index for candidate retrieval
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import faiss


class VectorSearchIndex:
    """Vector search index using FAISS for fast similarity search"""

    def __init__(self, embedding_dim: int = 1024):
        """
        Initialize vector search index

        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.id_map = []  # Maps index position to entity ID
        self.metadata = []  # Stores entity metadata

    def build_index(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: List[Dict]
    ):
        """
        Build FAISS index from embeddings

        Args:
            embeddings: Numpy array of embeddings (n x embedding_dim)
            ids: List of entity IDs
            metadata: List of entity metadata dictionaries
        """
        print(f"Building index with {len(embeddings)} vectors...")

        # Validate dimensions
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)

        # Create FAISS index (inner product = cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        # Store metadata
        self.id_map = ids
        self.metadata = metadata

        print(f"Index built successfully. Total vectors: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search for nearest neighbors

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of candidate entities with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Normalize query
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.id_map):  # Valid index
                results.append({
                    "rank": i + 1,
                    "ciq_id": self.id_map[idx],
                    "similarity": float(distance),  # Cosine similarity
                    "metadata": self.metadata[idx]
                })

        return results

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[List[Dict]]:
        """
        Search for nearest neighbors for multiple queries

        Args:
            query_embeddings: Query embeddings (n x embedding_dim)
            top_k: Number of results per query

        Returns:
            List of result lists (one per query)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Normalize queries
        query_embeddings = query_embeddings.astype('float32')
        faiss.normalize_L2(query_embeddings)

        # Search
        distances, indices = self.index.search(query_embeddings, top_k)

        # Format results
        all_results = []
        for query_idx in range(len(query_embeddings)):
            query_results = []
            for i, (distance, idx) in enumerate(zip(distances[query_idx], indices[query_idx])):
                if idx < len(self.id_map):
                    query_results.append({
                        "rank": i + 1,
                        "ciq_id": self.id_map[idx],
                        "similarity": float(distance),
                        "metadata": self.metadata[idx]
                    })
            all_results.append(query_results)

        return all_results

    def save_index(self, filepath: str):
        """Save FAISS index to disk"""
        if self.index is None:
            raise ValueError("No index to save")

        faiss.write_index(self.index, filepath)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(filepath)
        print(f"Index loaded from {filepath}. Total vectors: {self.index.ntotal}")


def build_reference_index(
    reference_df: pd.DataFrame,
    embeddings_model,
    embedding_column: str = "embedding"
) -> VectorSearchIndex:
    """
    Build vector search index from reference data

    Args:
        reference_df: DataFrame with S&P Capital IQ reference data
        embeddings_model: BGEEmbeddings model instance
        embedding_column: Column name for embeddings

    Returns:
        VectorSearchIndex instance
    """
    print("Building reference index...")

    # Extract embeddings
    if embedding_column in reference_df.columns:
        # Use pre-computed embeddings
        embeddings = np.vstack(reference_df[embedding_column].values)
    else:
        # Generate embeddings
        print("Generating embeddings for reference data...")
        texts = []
        for _, row in reference_df.iterrows():
            text_parts = [
                row.get("company_name", ""),
                row.get("primary_ticker", ""),
                row.get("industry", "")
            ]
            texts.append(" ".join(filter(None, text_parts)))

        embeddings = embeddings_model.encode(texts, show_progress_bar=True)

    # Extract IDs and metadata
    ids = reference_df["ciq_id"].tolist()
    metadata = reference_df.to_dict('records')

    # Build index
    index = VectorSearchIndex(embedding_dim=embeddings.shape[1])
    index.build_index(embeddings, ids, metadata)

    return index
