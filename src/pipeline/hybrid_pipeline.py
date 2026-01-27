"""
Hybrid entity matching pipeline orchestrator
Combines exact matching, vector search, Ditto, and Foundation Models
"""
from __future__ import annotations

from typing import Dict, List, Optional
from ..data.preprocessor import EntityPreprocessor, create_entity_features
from ..models.embeddings import BGEEmbeddings, create_embeddings_model
from ..models.ditto_matcher import DittoMatcher
from ..models.vector_search import VectorSearchIndex
from ..models.foundation_model import FoundationModelMatcher
from .exact_match import ExactMatcher


class HybridMatchingPipeline:
    """
    Multi-stage entity matching pipeline:
    1. Exact matching (identifiers)
    2. Vector search (candidate retrieval)
    3. Ditto matcher (fine-tuned classification)
    4. Foundation Model (edge cases)
    """

    def __init__(
        self,
        reference_df,
        ditto_model_path: Optional[str] = None,
        embeddings_provider: str = "huggingface",
        embeddings_model_name: str = None,
        foundation_model_name: str = "databricks-dbrx-instruct",
        ditto_high_confidence: float = 0.90,
        ditto_low_confidence: float = 0.70,
        enable_foundation_model: bool = True,
        databricks_client=None
    ):
        """
        Initialize hybrid pipeline

        Args:
            reference_df: S&P Capital IQ reference data
            ditto_model_path: Path to trained Ditto model
            embeddings_provider: "huggingface" or "databricks" (default: "huggingface")
            embeddings_model_name: Model name (provider-specific default if None)
                - Hugging Face: "BAAI/bge-large-en-v1.5" (default)
                - Databricks: "databricks-gte-large-en" (default)
            foundation_model_name: Foundation model name
            ditto_high_confidence: High confidence threshold for auto-accept
            ditto_low_confidence: Low confidence threshold for Foundation Model
            enable_foundation_model: Enable Foundation Model fallback
            databricks_client: Databricks WorkspaceClient
        """
        self.reference_df = reference_df
        self.preprocessor = EntityPreprocessor()

        # Thresholds
        self.ditto_high_confidence = ditto_high_confidence
        self.ditto_low_confidence = ditto_low_confidence
        self.enable_foundation_model = enable_foundation_model

        # Stage 1: Exact matching
        print("Initializing Exact Matcher...")
        self.exact_matcher = ExactMatcher(reference_df)

        # Stage 2: Embeddings + Vector Search
        print(f"Initializing Embeddings (provider: {embeddings_provider})...")
        self.embeddings_model = create_embeddings_model(
            provider=embeddings_provider,
            model_name=embeddings_model_name,
            databricks_client=databricks_client
        )

        print("Building Vector Search Index...")
        self.vector_index = self._build_vector_index()

        # Stage 3: Ditto matcher
        self.ditto_matcher = None
        if ditto_model_path:
            print("Loading Ditto Matcher...")
            self.ditto_matcher = DittoMatcher()
            self.ditto_matcher.load_model(ditto_model_path)

        # Stage 4: Foundation Model
        self.foundation_model = None
        if enable_foundation_model:
            print("Initializing Foundation Model...")
            self.foundation_model = FoundationModelMatcher(
                model_name=foundation_model_name,
                databricks_client=databricks_client
            )

        print("Pipeline initialized successfully!")

    def _build_vector_index(self) -> VectorSearchIndex:
        """Build vector search index from reference data"""
        from ..models.vector_search import build_reference_index
        return build_reference_index(self.reference_df, self.embeddings_model)

    def match(self, entity: Dict, return_candidates: bool = False) -> Dict:
        """
        Match a single entity through the pipeline

        Args:
            entity: Source entity dictionary
            return_candidates: Return candidate information

        Returns:
            Match result with ciq_id, confidence, method, and reasoning
        """
        # Preprocess entity
        entity = self.preprocessor.preprocess_entity(entity)

        # Stage 1: Exact matching
        if result := self.exact_matcher.match(entity):
            result["stage"] = 1
            result["stage_name"] = "exact_match"
            return result

        # Stage 2: Vector search for candidates
        query_embedding = self.embeddings_model.encode_entity(entity)
        candidates = self.vector_index.search(query_embedding, top_k=10)

        if not candidates:
            return {
                "ciq_id": None,
                "confidence": 0.0,
                "match_method": "no_candidates",
                "reasoning": "No candidates found in vector search",
                "stage": 2
            }

        # Stage 3: Ditto matcher
        if self.ditto_matcher:
            best_match = None
            best_confidence = 0.0

            # Create entity features for Ditto
            source_features = create_entity_features(entity)

            for candidate in candidates:
                candidate_features = create_entity_features(candidate["metadata"])

                # Predict match
                prediction, confidence = self.ditto_matcher.predict(
                    source_features,
                    candidate_features
                )

                if prediction == 1 and confidence > best_confidence:
                    best_confidence = confidence
                    best_match = candidate

            # High confidence match
            if best_match and best_confidence >= self.ditto_high_confidence:
                result = {
                    "ciq_id": best_match["ciq_id"],
                    "confidence": best_confidence,
                    "match_method": "ditto_high_confidence",
                    "reasoning": f"Ditto prediction with {best_confidence:.1%} confidence",
                    "stage": 3,
                    "stage_name": "ditto"
                }
                if return_candidates:
                    result["candidates"] = candidates[:3]
                return result

            # Medium confidence - review queue
            if best_match and best_confidence >= self.ditto_low_confidence:
                result = {
                    "ciq_id": best_match["ciq_id"],
                    "confidence": best_confidence,
                    "match_method": "ditto_medium_confidence",
                    "reasoning": f"Ditto prediction with {best_confidence:.1%} confidence (needs review)",
                    "stage": 3,
                    "stage_name": "ditto",
                    "needs_review": True
                }
                if return_candidates:
                    result["candidates"] = candidates[:3]
                return result

        # Stage 4: Foundation Model fallback
        if self.enable_foundation_model and self.foundation_model:
            fm_result = self.foundation_model.match(entity, candidates, top_k=3)
            fm_result["stage"] = 4
            fm_result["stage_name"] = "foundation_model"

            if return_candidates:
                fm_result["candidates"] = candidates[:3]

            return fm_result

        # No match found
        result = {
            "ciq_id": None,
            "confidence": 0.0,
            "match_method": "no_match",
            "reasoning": "No confident match found across all stages",
            "stage": 4
        }

        if return_candidates:
            result["candidates"] = candidates[:3]

        return result

    def batch_match(
        self,
        entities: List[Dict],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Match multiple entities

        Args:
            entities: List of entity dictionaries
            show_progress: Show progress bar

        Returns:
            List of match results
        """
        results = []

        if show_progress:
            from tqdm import tqdm
            entities = tqdm(entities, desc="Matching entities")

        for entity in entities:
            result = self.match(entity)
            result["source_id"] = entity.get("source_id")
            results.append(result)

        return results

    def get_pipeline_stats(self, results: List[Dict]) -> Dict:
        """
        Calculate pipeline statistics

        Args:
            results: List of match results

        Returns:
            Dictionary with pipeline statistics
        """
        total = len(results)
        matched = len([r for r in results if r["ciq_id"] is not None])

        # Count by stage
        stage_counts = {}
        for result in results:
            stage = result.get("stage_name", "unknown")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        # Count by method
        method_counts = {}
        for result in results:
            method = result.get("match_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        # Confidence distribution
        confidences = [r["confidence"] for r in results if r["confidence"] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        stats = {
            "total_entities": total,
            "matched": matched,
            "match_rate": matched / total if total > 0 else 0,
            "avg_confidence": avg_confidence,
            "stages": stage_counts,
            "methods": method_counts,
            "needs_review": len([r for r in results if r.get("needs_review", False)])
        }

        return stats

    def print_pipeline_stats(self, results: List[Dict]):
        """Print formatted pipeline statistics"""
        stats = self.get_pipeline_stats(results)

        print("\n" + "=" * 60)
        print("PIPELINE STATISTICS")
        print("=" * 60)
        print(f"Total Entities: {stats['total_entities']}")
        print(f"Matched: {stats['matched']} ({stats['match_rate']:.1%})")
        print(f"Average Confidence: {stats['avg_confidence']:.1%}")
        print(f"Needs Review: {stats['needs_review']}")

        print("\nMatches by Stage:")
        for stage, count in stats['stages'].items():
            percentage = count / stats['total_entities'] * 100
            print(f"  {stage}: {count} ({percentage:.1f}%)")

        print("\nMatches by Method:")
        for method, count in sorted(stats['methods'].items(), key=lambda x: -x[1]):
            percentage = count / stats['total_entities'] * 100
            print(f"  {method}: {count} ({percentage:.1f}%)")

        print("=" * 60)
