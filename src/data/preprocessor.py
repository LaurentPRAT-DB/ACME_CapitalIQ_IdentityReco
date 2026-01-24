"""
Data preprocessing and normalization utilities
"""
from __future__ import annotations

import re
import pandas as pd
from typing import Dict, Optional


class EntityPreprocessor:
    """Preprocess and normalize entity data for matching"""

    # Company name suffixes to remove/normalize
    SUFFIXES = [
        "Inc", "Inc.", "Incorporated",
        "Corp", "Corp.", "Corporation",
        "Ltd", "Ltd.", "Limited",
        "LLC", "L.L.C.", "L.L.C",
        "LP", "L.P.", "LLP",
        "PLC", "P.L.C.",
        "SA", "S.A.",
        "GmbH", "AG", "NV",
        "Co", "Co.", "Company"
    ]

    def __init__(self):
        self.suffix_pattern = self._build_suffix_pattern()

    def _build_suffix_pattern(self) -> str:
        """Build regex pattern for company suffixes"""
        escaped_suffixes = [re.escape(s) for s in self.SUFFIXES]
        return r'\b(' + '|'.join(escaped_suffixes) + r')\b'

    def normalize_company_name(self, name: str) -> str:
        """
        Normalize company name for matching

        Args:
            name: Raw company name

        Returns:
            Normalized company name
        """
        if not name or pd.isna(name):
            return ""

        # Convert to uppercase
        normalized = str(name).upper().strip()

        # Remove punctuation (except &)
        normalized = re.sub(r'[,.\']', '', normalized)

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove common suffixes
        normalized = re.sub(self.suffix_pattern, '', normalized, flags=re.IGNORECASE)

        # Remove leading/trailing whitespace again
        normalized = normalized.strip()

        return normalized

    def normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol"""
        if not ticker or pd.isna(ticker):
            return ""
        return str(ticker).upper().strip()

    def normalize_identifier(self, identifier: str) -> str:
        """Normalize identifiers (LEI, CUSIP, ISIN)"""
        if not identifier or pd.isna(identifier):
            return ""
        # Remove spaces and convert to uppercase
        return re.sub(r'\s+', '', str(identifier).upper())

    def create_search_text(self, entity: Dict) -> str:
        """
        Create searchable text representation of entity

        Args:
            entity: Dictionary with entity attributes

        Returns:
            Concatenated search text
        """
        parts = []

        # Company name (most important)
        if name := entity.get("company_name"):
            parts.append(self.normalize_company_name(name))

        # Ticker
        if ticker := entity.get("ticker"):
            parts.append(self.normalize_ticker(ticker))

        # Industry
        if industry := entity.get("industry"):
            parts.append(str(industry).strip())

        # Location
        if city := entity.get("city"):
            parts.append(str(city).strip())
        if country := entity.get("country"):
            parts.append(str(country).strip())

        return " ".join(filter(None, parts))

    def preprocess_entity(self, entity: Dict) -> Dict:
        """
        Preprocess entity dictionary

        Args:
            entity: Raw entity dictionary

        Returns:
            Preprocessed entity dictionary
        """
        processed = entity.copy()

        # Normalize company name
        if "company_name" in entity:
            processed["company_name_normalized"] = self.normalize_company_name(
                entity["company_name"]
            )

        # Normalize ticker
        if "ticker" in entity:
            processed["ticker_normalized"] = self.normalize_ticker(
                entity["ticker"]
            )

        # Normalize identifiers
        for id_field in ["lei", "cusip", "isin"]:
            if id_field in entity:
                processed[f"{id_field}_normalized"] = self.normalize_identifier(
                    entity[id_field]
                )

        # Create search text
        processed["search_text"] = self.create_search_text(entity)

        return processed

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess pandas DataFrame

        Args:
            df: DataFrame with entity records

        Returns:
            DataFrame with normalized fields
        """
        df = df.copy()

        # Normalize company name
        if "company_name" in df.columns:
            df["company_name_normalized"] = df["company_name"].apply(
                self.normalize_company_name
            )

        # Normalize ticker
        if "ticker" in df.columns:
            df["ticker_normalized"] = df["ticker"].apply(
                self.normalize_ticker
            )

        # Normalize identifiers
        for id_field in ["lei", "cusip", "isin"]:
            if id_field in df.columns:
                df[f"{id_field}_normalized"] = df[id_field].apply(
                    self.normalize_identifier
                )

        return df


def create_entity_features(entity: Dict) -> str:
    """
    Create feature string for Ditto model input

    Format: COL name VAL value COL ticker VAL value ...

    Args:
        entity: Entity dictionary

    Returns:
        Feature string for Ditto
    """
    parts = []

    # Define field order (most important first)
    fields = [
        ("company_name", "name"),
        ("ticker", "ticker"),
        ("lei", "lei"),
        ("cusip", "cusip"),
        ("isin", "isin"),
        ("industry", "industry"),
        ("country", "country"),
        ("city", "city")
    ]

    for field, label in fields:
        if value := entity.get(field):
            if value and not pd.isna(value):
                parts.append(f"COL {label} VAL {value}")

    return " ".join(parts)
