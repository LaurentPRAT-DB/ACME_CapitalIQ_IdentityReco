"""
Exact matching based on identifiers and normalized names
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, Optional
from ..data.preprocessor import EntityPreprocessor


class ExactMatcher:
    """Rule-based exact matching on identifiers"""

    def __init__(self, reference_df: pd.DataFrame):
        """
        Initialize exact matcher

        Args:
            reference_df: S&P Capital IQ reference data
        """
        self.reference_df = reference_df
        self.preprocessor = EntityPreprocessor()

        # Build lookup dictionaries for fast matching
        self._build_lookups()

    def _build_lookups(self):
        """Build fast lookup dictionaries"""
        self.lei_lookup = {}
        self.cusip_lookup = {}
        self.isin_lookup = {}
        self.ticker_lookup = {}
        self.name_lookup = {}

        for _, row in self.reference_df.iterrows():
            ciq_id = row["ciq_id"]

            # LEI lookup
            if lei := row.get("lei"):
                normalized_lei = self.preprocessor.normalize_identifier(lei)
                if normalized_lei:
                    self.lei_lookup[normalized_lei] = ciq_id

            # CUSIP lookup
            if cusip := row.get("cusip"):
                normalized_cusip = self.preprocessor.normalize_identifier(cusip)
                if normalized_cusip:
                    self.cusip_lookup[normalized_cusip] = ciq_id

            # ISIN lookup
            if isin := row.get("isin"):
                normalized_isin = self.preprocessor.normalize_identifier(isin)
                if normalized_isin:
                    self.isin_lookup[normalized_isin] = ciq_id

            # Ticker lookup
            if ticker := row.get("primary_ticker"):
                normalized_ticker = self.preprocessor.normalize_ticker(ticker)
                if normalized_ticker:
                    self.ticker_lookup[normalized_ticker] = ciq_id

            # Normalized name lookup
            if name := row.get("company_name"):
                normalized_name = self.preprocessor.normalize_company_name(name)
                if normalized_name:
                    self.name_lookup[normalized_name] = ciq_id

        print(f"Built lookups:")
        print(f"  - LEI: {len(self.lei_lookup)} entries")
        print(f"  - CUSIP: {len(self.cusip_lookup)} entries")
        print(f"  - ISIN: {len(self.isin_lookup)} entries")
        print(f"  - Ticker: {len(self.ticker_lookup)} entries")
        print(f"  - Name: {len(self.name_lookup)} entries")

    def match(self, entity: Dict) -> Optional[Dict]:
        """
        Attempt exact match on entity

        Args:
            entity: Source entity dictionary

        Returns:
            Match result or None if no exact match found
        """
        # Try LEI match (highest confidence)
        if lei := entity.get("lei"):
            normalized_lei = self.preprocessor.normalize_identifier(lei)
            if ciq_id := self.lei_lookup.get(normalized_lei):
                return {
                    "ciq_id": ciq_id,
                    "confidence": 1.0,
                    "match_method": "exact_lei",
                    "reasoning": f"Exact LEI match: {lei}"
                }

        # Try CUSIP match
        if cusip := entity.get("cusip"):
            normalized_cusip = self.preprocessor.normalize_identifier(cusip)
            if ciq_id := self.cusip_lookup.get(normalized_cusip):
                return {
                    "ciq_id": ciq_id,
                    "confidence": 1.0,
                    "match_method": "exact_cusip",
                    "reasoning": f"Exact CUSIP match: {cusip}"
                }

        # Try ISIN match
        if isin := entity.get("isin"):
            normalized_isin = self.preprocessor.normalize_identifier(isin)
            if ciq_id := self.isin_lookup.get(normalized_isin):
                return {
                    "ciq_id": ciq_id,
                    "confidence": 1.0,
                    "match_method": "exact_isin",
                    "reasoning": f"Exact ISIN match: {isin}"
                }

        # Try exact name match
        if name := entity.get("company_name"):
            normalized_name = self.preprocessor.normalize_company_name(name)
            if ciq_id := self.name_lookup.get(normalized_name):
                return {
                    "ciq_id": ciq_id,
                    "confidence": 0.95,  # Slightly lower confidence for name match
                    "match_method": "exact_name",
                    "reasoning": f"Exact normalized name match: {name}"
                }

        # Try ticker match (lower confidence - tickers can be reused)
        if ticker := entity.get("ticker"):
            normalized_ticker = self.preprocessor.normalize_ticker(ticker)
            if ciq_id := self.ticker_lookup.get(normalized_ticker):
                # Verify with name if available
                if name := entity.get("company_name"):
                    # Get reference entity
                    ref_entity = self.reference_df[
                        self.reference_df["ciq_id"] == ciq_id
                    ].iloc[0]

                    ref_name = self.preprocessor.normalize_company_name(
                        ref_entity["company_name"]
                    )
                    entity_name = self.preprocessor.normalize_company_name(name)

                    # Check name similarity
                    if ref_name in entity_name or entity_name in ref_name:
                        return {
                            "ciq_id": ciq_id,
                            "confidence": 0.90,
                            "match_method": "exact_ticker_verified",
                            "reasoning": f"Exact ticker match with name verification: {ticker}"
                        }

        return None

    def batch_match(self, entities: list) -> list:
        """
        Match multiple entities

        Args:
            entities: List of entity dictionaries

        Returns:
            List of match results (None for no match)
        """
        return [self.match(entity) for entity in entities]

    def get_coverage_stats(self, entities: list) -> Dict:
        """
        Calculate exact match coverage statistics

        Args:
            entities: List of entity dictionaries

        Returns:
            Dictionary with coverage statistics
        """
        results = self.batch_match(entities)
        matches = [r for r in results if r is not None]

        stats = {
            "total_entities": len(entities),
            "exact_matches": len(matches),
            "coverage_rate": len(matches) / len(entities) if entities else 0,
            "match_methods": {}
        }

        # Count by method
        for match in matches:
            method = match["match_method"]
            stats["match_methods"][method] = stats["match_methods"].get(method, 0) + 1

        return stats
