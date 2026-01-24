"""
Generate training data for Ditto model from S&P 500 gold standard
"""
from __future__ import annotations

import pandas as pd
import random
from typing import List, Dict, Tuple, Optional
from .preprocessor import EntityPreprocessor


class TrainingDataGenerator:
    """Generate training pairs for Ditto fine-tuning"""

    def __init__(self, seed: int = 42):
        self.preprocessor = EntityPreprocessor()
        self.seed = seed
        random.seed(seed)

    def generate_from_sp500(
        self,
        reference_df: pd.DataFrame,
        num_positive_pairs: int = 500,
        num_negative_pairs: int = 500
    ) -> pd.DataFrame:
        """
        Generate training pairs from S&P 500 reference data

        Args:
            reference_df: DataFrame with S&P Capital IQ reference data
            num_positive_pairs: Number of positive (matching) pairs
            num_negative_pairs: Number of negative (non-matching) pairs

        Returns:
            DataFrame with training pairs (left_entity, right_entity, label)
        """
        training_pairs = []

        # Generate positive pairs
        positive_pairs = self._generate_positive_pairs(
            reference_df,
            num_pairs=num_positive_pairs
        )
        training_pairs.extend(positive_pairs)

        # Generate negative pairs
        negative_pairs = self._generate_negative_pairs(
            reference_df,
            num_pairs=num_negative_pairs
        )
        training_pairs.extend(negative_pairs)

        # Convert to DataFrame
        df = pd.DataFrame(training_pairs)

        # Shuffle
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        print(f"Generated {len(df)} training pairs:")
        print(f"  - Positive: {len(df[df['label'] == 1])}")
        print(f"  - Negative: {len(df[df['label'] == 0])}")

        return df

    def _generate_positive_pairs(
        self,
        reference_df: pd.DataFrame,
        num_pairs: int
    ) -> List[Dict]:
        """Generate positive (matching) entity pairs"""
        positive_pairs = []

        for _, company in reference_df.iterrows():
            # Pair: official name vs alias
            if aliases := company.get("aliases"):
                if isinstance(aliases, list):
                    for alias in aliases[:2]:  # Limit to 2 aliases per company
                        positive_pairs.append({
                            "left_entity": self._format_entity(company, use_name=True),
                            "right_entity": self._format_entity(company, use_alias=alias),
                            "label": 1
                        })

            # Pair: official name vs ticker
            if ticker := company.get("primary_ticker"):
                positive_pairs.append({
                    "left_entity": self._format_entity(company, use_name=True),
                    "right_entity": self._format_entity(company, use_ticker=True),
                    "label": 1
                })

            # Pair: name with LEI vs name with CUSIP
            if company.get("lei") and company.get("cusip"):
                positive_pairs.append({
                    "left_entity": self._format_entity(company, include_lei=True),
                    "right_entity": self._format_entity(company, include_cusip=True),
                    "label": 1
                })

            # Stop if we have enough pairs
            if len(positive_pairs) >= num_pairs:
                break

        return positive_pairs[:num_pairs]

    def _generate_negative_pairs(
        self,
        reference_df: pd.DataFrame,
        num_pairs: int
    ) -> List[Dict]:
        """Generate negative (non-matching) entity pairs"""
        negative_pairs = []
        companies = reference_df.to_dict('records')

        # Generate random pairs of different companies
        for _ in range(num_pairs):
            c1, c2 = random.sample(companies, 2)

            # Make sure they're actually different
            if c1.get("ciq_id") != c2.get("ciq_id"):
                negative_pairs.append({
                    "left_entity": self._format_entity(c1),
                    "right_entity": self._format_entity(c2),
                    "label": 0
                })

        return negative_pairs

    def _format_entity(
        self,
        company: Dict,
        use_name: bool = True,
        use_ticker: bool = False,
        use_alias: Optional[str] = None,
        include_lei: bool = False,
        include_cusip: bool = False
    ) -> str:
        """
        Format entity as Ditto input string

        Format: COL name VAL value COL ticker VAL value ...
        """
        parts = []

        # Company name
        if use_alias:
            parts.append(f"COL name VAL {use_alias}")
        elif use_ticker:
            parts.append(f"COL name VAL {company.get('primary_ticker', '')}")
        elif use_name:
            parts.append(f"COL name VAL {company.get('company_name', '')}")

        # Ticker
        if company.get("primary_ticker") and not use_ticker:
            parts.append(f"COL ticker VAL {company['primary_ticker']}")

        # LEI
        if include_lei and company.get("lei"):
            parts.append(f"COL lei VAL {company['lei']}")

        # CUSIP
        if include_cusip and company.get("cusip"):
            parts.append(f"COL cusip VAL {company['cusip']}")

        # Industry
        if company.get("industry"):
            parts.append(f"COL industry VAL {company['industry']}")

        # Country
        if company.get("country"):
            parts.append(f"COL country VAL {company['country']}")

        return "\t".join(parts)

    def generate_from_manual_labels(
        self,
        filepath: str
    ) -> pd.DataFrame:
        """
        Load manually labeled entity pairs from CSV

        Expected format:
        source_entity, candidate_entity, label

        Args:
            filepath: Path to CSV file with manual labels

        Returns:
            DataFrame with training pairs
        """
        df = pd.read_csv(filepath)

        # Validate columns
        required_cols = ["source_entity", "candidate_entity", "label"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # Rename columns to match Ditto format
        df = df.rename(columns={
            "source_entity": "left_entity",
            "candidate_entity": "right_entity"
        })

        print(f"Loaded {len(df)} manually labeled pairs")
        return df

    def augment_training_data(
        self,
        df: pd.DataFrame,
        augmentation_factor: float = 0.2
    ) -> pd.DataFrame:
        """
        Augment training data with variations

        Args:
            df: Original training data
            augmentation_factor: Fraction of data to augment

        Returns:
            Augmented training data
        """
        augmented = []

        num_to_augment = int(len(df) * augmentation_factor)
        samples = df.sample(n=num_to_augment, random_state=self.seed)

        for _, row in samples.iterrows():
            # Create variation by removing some fields
            left_parts = row["left_entity"].split("\t")
            right_parts = row["right_entity"].split("\t")

            # Randomly remove 1-2 fields
            if len(left_parts) > 2:
                left_parts = random.sample(left_parts, len(left_parts) - 1)
            if len(right_parts) > 2:
                right_parts = random.sample(right_parts, len(right_parts) - 1)

            augmented.append({
                "left_entity": "\t".join(left_parts),
                "right_entity": "\t".join(right_parts),
                "label": row["label"]
            })

        # Combine original and augmented
        augmented_df = pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)
        augmented_df = augmented_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        print(f"Augmented training data: {len(df)} -> {len(augmented_df)} pairs")
        return augmented_df
