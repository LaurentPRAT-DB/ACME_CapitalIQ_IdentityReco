"""
Data loading utilities for various sources
"""
from __future__ import annotations

import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path


class DataLoader:
    """Load entity data from various sources"""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")

    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file"""
        return pd.read_csv(filepath, **kwargs)

    def load_parquet(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from Parquet file"""
        return pd.read_parquet(filepath, **kwargs)

    def load_sample_entities(self) -> List[Dict]:
        """
        Load sample entities for testing

        Returns:
            List of sample entity dictionaries
        """
        return [
            {
                "source_id": "CRM-001",
                "company_name": "Apple Inc.",
                "ticker": "AAPL",
                "lei": "HWUPKR0MPOU8FGXBT394",
                "country": "United States",
                "city": "Cupertino",
                "industry": "Technology Hardware"
            },
            {
                "source_id": "CRM-002",
                "company_name": "Microsoft Corporation",
                "ticker": "MSFT",
                "cusip": "594918104",
                "country": "United States",
                "city": "Redmond",
                "industry": "Software"
            },
            {
                "source_id": "CRM-003",
                "company_name": "Apple Computer Inc.",
                "ticker": "AAPL",
                "country": "USA",
                "industry": "Consumer Electronics"
            },
            {
                "source_id": "TRD-001",
                "company_name": "MSFT",
                "cusip": "594918104",
                "industry": "Technology"
            },
            {
                "source_id": "VND-001",
                "company_name": "Amazon.com Inc",
                "ticker": "AMZN",
                "country": "United States",
                "industry": "E-commerce"
            }
        ]

    def load_reference_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load S&P Capital IQ reference data

        Args:
            filepath: Path to reference data file

        Returns:
            DataFrame with S&P Capital IQ entities
        """
        if filepath:
            return self.load_parquet(filepath)

        # Return sample reference data for testing
        return pd.DataFrame([
            {
                "ciq_id": "IQ24937",
                "company_name": "Apple Inc.",
                "primary_ticker": "AAPL",
                "exchange": "NASDAQ",
                "country": "United States",
                "lei": "HWUPKR0MPOU8FGXBT394",
                "cusip": "037833100",
                "isin": "US0378331005",
                "industry": "Technology Hardware",
                "aliases": ["Apple Computer Inc.", "Apple Computer", "AAPL"]
            },
            {
                "ciq_id": "IQ4004",
                "company_name": "Microsoft Corporation",
                "primary_ticker": "MSFT",
                "exchange": "NASDAQ",
                "country": "United States",
                "cusip": "594918104",
                "isin": "US5949181045",
                "industry": "Software",
                "aliases": ["Microsoft Corp", "MSFT", "MS"]
            },
            {
                "ciq_id": "IQ112209",
                "company_name": "Amazon.com Inc",
                "primary_ticker": "AMZN",
                "exchange": "NASDAQ",
                "country": "United States",
                "industry": "E-commerce",
                "aliases": ["Amazon", "AMZN", "Amazon.com"]
            },
            {
                "ciq_id": "IQ18427",
                "company_name": "Tesla Inc",
                "primary_ticker": "TSLA",
                "exchange": "NASDAQ",
                "country": "United States",
                "industry": "Automotive",
                "aliases": ["Tesla", "TSLA", "Tesla Motors"]
            },
            {
                "ciq_id": "IQ6095",
                "company_name": "Alphabet Inc",
                "primary_ticker": "GOOGL",
                "exchange": "NASDAQ",
                "country": "United States",
                "industry": "Internet Services",
                "aliases": ["Google", "GOOGL", "GOOG", "Alphabet"]
            }
        ])

    def save_training_data(self, df: pd.DataFrame, filepath: str):
        """Save training data to CSV"""
        # Create parent directory if it doesn't exist
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} training pairs to {filepath}")

    def save_results(self, df: pd.DataFrame, filepath: str, format: str = "csv"):
        """Save matching results"""
        # Create parent directory if it doesn't exist
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "parquet":
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Saved {len(df)} results to {filepath}")
