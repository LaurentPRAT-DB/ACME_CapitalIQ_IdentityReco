"""
Generate large-scale test datasets for entity matching
Creates realistic reference and source entity datasets at scale
"""
from __future__ import annotations

import pandas as pd
import random
import string
from typing import List, Dict, Tuple
from datetime import datetime, timedelta


class LargeDatasetGenerator:
    """Generate large-scale test datasets with realistic variations"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

        # Common industry sectors
        self.industries = [
            "Technology Hardware", "Software", "Semiconductors", "Internet Services",
            "E-commerce", "Financial Services", "Banking", "Insurance",
            "Pharmaceuticals", "Biotechnology", "Medical Devices", "Healthcare Services",
            "Automotive", "Aerospace & Defense", "Industrial Machinery",
            "Oil & Gas", "Utilities", "Renewable Energy",
            "Retail", "Consumer Goods", "Food & Beverage", "Restaurants",
            "Telecommunications", "Media & Entertainment", "Real Estate",
            "Transportation", "Logistics", "Construction", "Materials"
        ]

        # Countries for diversification
        self.countries = [
            "United States", "United Kingdom", "Germany", "France", "Canada",
            "Japan", "China", "South Korea", "Australia", "Netherlands",
            "Switzerland", "Sweden", "Singapore", "Hong Kong", "India"
        ]

        # Company name patterns
        self.company_suffixes = [
            "Inc", "Inc.", "Corporation", "Corp", "Corp.", "Company", "Co",
            "Co.", "Limited", "Ltd", "Ltd.", "Group", "Holdings", "Holding",
            "Plc", "PLC", "S.A.", "SA", "GmbH", "AG", "AB"
        ]

    def generate_reference_entities(self, num_entities: int = 1000) -> pd.DataFrame:
        """
        Generate reference entities (S&P Capital IQ format)

        Args:
            num_entities: Number of reference entities to generate

        Returns:
            DataFrame with reference entities
        """
        entities = []

        for i in range(num_entities):
            company_base = self._generate_company_name(i)
            ticker = self._generate_ticker(company_base, i)
            industry = random.choice(self.industries)
            country = random.choice(self.countries)
            sector = self._map_industry_to_sector(industry)

            entity = {
                "ciq_id": f"IQ{100000 + i}",
                "company_name": company_base,
                "ticker": ticker,
                "lei": self._generate_lei(),
                "cusip": self._generate_cusip() if country == "United States" else None,
                "isin": self._generate_isin(country),
                "country": country,
                "industry": industry,
                "sector": sector,
                "market_cap": round(random.uniform(100_000_000, 500_000_000_000), 2),
                "last_updated": datetime.now() - timedelta(days=random.randint(0, 30))
            }

            entities.append(entity)

        df = pd.DataFrame(entities)
        print(f"Generated {len(df)} reference entities")
        return df

    def generate_source_entities(
        self,
        reference_df: pd.DataFrame,
        num_entities: int = 3000,
        match_ratio: float = 0.7
    ) -> pd.DataFrame:
        """
        Generate source entities with realistic variations

        Args:
            reference_df: Reference entities to base variations on
            num_entities: Number of source entities to generate
            match_ratio: Ratio of entities that should match (0.7 = 70% have matches)

        Returns:
            DataFrame with source entities
        """
        source_entities = []
        num_with_matches = int(num_entities * match_ratio)
        num_without_matches = num_entities - num_with_matches

        # Generate entities with matches (variations of reference entities)
        for i in range(num_with_matches):
            # Randomly select a reference entity
            ref_entity = reference_df.sample(n=1).iloc[0]

            # Create variation
            variation_type = random.choice([
                "name_variation", "abbreviation", "typo", "missing_suffix",
                "ticker_only", "with_identifiers", "partial_info"
            ])

            source_entity = self._create_entity_variation(ref_entity, i, variation_type)
            source_entities.append(source_entity)

        # Generate entities without matches (new companies)
        for i in range(num_without_matches):
            entity_id = num_with_matches + i
            source_entity = self._create_non_matching_entity(entity_id)
            source_entities.append(source_entity)

        df = pd.DataFrame(source_entities)

        # Shuffle to mix matched and non-matched
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        print(f"Generated {len(df)} source entities:")
        print(f"  - With potential matches: {num_with_matches} ({match_ratio*100:.0f}%)")
        print(f"  - Without matches: {num_without_matches} ({(1-match_ratio)*100:.0f}%)")

        return df

    def _create_entity_variation(
        self,
        ref_entity: pd.Series,
        index: int,
        variation_type: str
    ) -> Dict:
        """Create a variation of a reference entity"""
        source_system = random.choice([
            "Salesforce", "SAP", "Oracle", "Bloomberg", "Reuters",
            "FactSet", "Refinitiv", "S&P", "Moody's", "Fitch"
        ])

        base_entity = {
            "source_id": f"{source_system[:3].upper()}-{index:05d}",
            "source_system": source_system,
            "ingestion_timestamp": datetime.now()
        }

        if variation_type == "name_variation":
            # Slight name variation (e.g., "Inc" vs "Inc.")
            base_entity["company_name"] = self._vary_company_name(ref_entity["company_name"])
            base_entity["ticker"] = ref_entity["ticker"]
            base_entity["country"] = ref_entity["country"]
            base_entity["industry"] = ref_entity["industry"]

        elif variation_type == "abbreviation":
            # Abbreviated name
            base_entity["company_name"] = self._abbreviate_name(ref_entity["company_name"])
            base_entity["ticker"] = ref_entity["ticker"]
            base_entity["country"] = ref_entity["country"]

        elif variation_type == "typo":
            # Name with typo
            base_entity["company_name"] = self._add_typo(ref_entity["company_name"])
            base_entity["ticker"] = ref_entity["ticker"]
            base_entity["industry"] = ref_entity["industry"]

        elif variation_type == "missing_suffix":
            # Name without suffix (Inc, Corp, etc.)
            base_entity["company_name"] = self._remove_suffix(ref_entity["company_name"])
            base_entity["ticker"] = ref_entity["ticker"]
            base_entity["country"] = ref_entity["country"]

        elif variation_type == "ticker_only":
            # Only ticker, no name
            base_entity["company_name"] = ref_entity["ticker"]
            base_entity["ticker"] = ref_entity["ticker"]
            base_entity["country"] = ref_entity["country"]

        elif variation_type == "with_identifiers":
            # Correct name but with identifiers
            base_entity["company_name"] = ref_entity["company_name"]
            base_entity["ticker"] = ref_entity["ticker"]
            base_entity["lei"] = ref_entity["lei"] if random.random() > 0.5 else None
            base_entity["cusip"] = ref_entity["cusip"] if random.random() > 0.5 else None
            base_entity["isin"] = ref_entity["isin"] if random.random() > 0.5 else None

        elif variation_type == "partial_info":
            # Only some fields available (but company_name is always required)
            # Always include company_name (required field)
            base_entity["company_name"] = ref_entity["company_name"]

            # Randomly include other fields
            optional_fields = ["ticker", "country", "industry"]
            selected_optional = random.sample(optional_fields, k=random.randint(1, 2))

            if "ticker" in selected_optional:
                base_entity["ticker"] = ref_entity["ticker"]
            if "country" in selected_optional:
                base_entity["country"] = ref_entity["country"]
            if "industry" in selected_optional:
                base_entity["industry"] = ref_entity["industry"]

        # Final validation: Ensure company_name is always present (NOT NULL constraint)
        if "company_name" not in base_entity or not base_entity["company_name"]:
            base_entity["company_name"] = ref_entity["company_name"]

        return base_entity

    def _create_non_matching_entity(self, index: int) -> Dict:
        """Create an entity that won't match any reference"""
        source_system = random.choice([
            "Salesforce", "SAP", "Oracle", "Internal", "Legacy"
        ])

        return {
            "source_id": f"{source_system[:3].upper()}-{index:05d}",
            "source_system": source_system,
            "company_name": self._generate_unique_company_name(),
            "ticker": None if random.random() > 0.3 else self._generate_ticker("NonExistent", index),
            "country": random.choice(self.countries),
            "industry": random.choice(self.industries),
            "lei": None,
            "cusip": None,
            "isin": None,
            "ingestion_timestamp": datetime.now()
        }

    def _generate_company_name(self, index: int) -> str:
        """Generate a realistic company name"""
        prefixes = [
            "Global", "International", "United", "American", "National",
            "First", "Advanced", "Digital", "Tech", "Quantum",
            "Smart", "Future", "Next", "Prime", "Elite",
            "Apex", "Summit", "Pioneer", "Innovative", "Strategic"
        ]

        cores = [
            "Systems", "Solutions", "Technologies", "Services", "Products",
            "Industries", "Manufacturing", "Enterprises", "Partners", "Ventures",
            "Capital", "Financial", "Energy", "Materials", "Resources",
            "Communications", "Networks", "Data", "Cloud", "Software"
        ]

        prefix = random.choice(prefixes)
        core = random.choice(cores)
        suffix = random.choice(self.company_suffixes)

        # Sometimes skip prefix
        if random.random() > 0.3:
            return f"{prefix} {core} {suffix}"
        else:
            return f"{core} {suffix}"

    def _generate_unique_company_name(self) -> str:
        """Generate a unique company name that won't match"""
        return f"Unmatchable {random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'])} {random.randint(1000, 9999)} Corp"

    def _generate_ticker(self, company_name: str, index: int) -> str:
        """Generate a ticker symbol"""
        # Extract uppercase letters
        letters = ''.join([c for c in company_name if c.isupper()])

        if len(letters) >= 3:
            ticker = letters[:4]
        else:
            ticker = company_name[:3].upper() + str(index % 10)

        return ticker

    def _generate_lei(self) -> str:
        """Generate a fake LEI (20 alphanumeric characters)"""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))

    def _generate_cusip(self) -> str:
        """Generate a fake CUSIP (9 characters)"""
        return ''.join(random.choices(string.digits, k=9))

    def _generate_isin(self, country: str) -> str:
        """Generate a fake ISIN"""
        country_codes = {
            "United States": "US",
            "United Kingdom": "GB",
            "Germany": "DE",
            "France": "FR",
            "Canada": "CA",
            "Japan": "JP",
            "China": "CN"
        }

        country_code = country_codes.get(country, "XX")
        nsin = ''.join(random.choices(string.ascii_uppercase + string.digits, k=9))
        check_digit = random.choice(string.digits)

        return f"{country_code}{nsin}{check_digit}"

    def _map_industry_to_sector(self, industry: str) -> str:
        """Map industry to sector"""
        sector_map = {
            "Technology": ["Technology Hardware", "Software", "Semiconductors", "Internet Services"],
            "Communication Services": ["Telecommunications", "Media & Entertainment"],
            "Consumer Discretionary": ["E-commerce", "Retail", "Restaurants", "Automotive"],
            "Consumer Staples": ["Consumer Goods", "Food & Beverage"],
            "Financials": ["Financial Services", "Banking", "Insurance", "Real Estate"],
            "Healthcare": ["Pharmaceuticals", "Biotechnology", "Medical Devices", "Healthcare Services"],
            "Industrials": ["Aerospace & Defense", "Industrial Machinery", "Transportation", "Logistics", "Construction"],
            "Energy": ["Oil & Gas", "Utilities", "Renewable Energy"],
            "Materials": ["Materials"]
        }

        for sector, industries in sector_map.items():
            if industry in industries:
                return sector

        return "Other"

    def _vary_company_name(self, name: str) -> str:
        """Create variation of company name"""
        variations = [
            lambda n: n.replace(" Inc", " Inc."),
            lambda n: n.replace(" Inc.", " Inc"),
            lambda n: n.replace(" Corp", " Corporation"),
            lambda n: n.replace(" Corporation", " Corp"),
            lambda n: n.replace(" Ltd", " Limited"),
            lambda n: n.replace(" Limited", " Ltd"),
            lambda n: n.replace(",", ""),
            lambda n: n.replace(" & ", " and "),
        ]

        return random.choice(variations)(name)

    def _abbreviate_name(self, name: str) -> str:
        """Abbreviate company name"""
        # Remove suffix
        for suffix in self.company_suffixes:
            name = name.replace(f" {suffix}", "")

        # Sometimes abbreviate words
        words = name.split()
        if len(words) > 1 and random.random() > 0.5:
            return words[0]  # Just first word

        return name

    def _add_typo(self, name: str) -> str:
        """Add a realistic typo to the name"""
        if len(name) < 5:
            return name

        typo_type = random.choice(["swap", "double", "missing"])

        if typo_type == "swap" and len(name) > 3:
            # Swap two adjacent characters
            pos = random.randint(1, len(name) - 2)
            name_list = list(name)
            name_list[pos], name_list[pos + 1] = name_list[pos + 1], name_list[pos]
            return ''.join(name_list)

        elif typo_type == "double":
            # Double a character
            pos = random.randint(0, len(name) - 1)
            return name[:pos] + name[pos] + name[pos:]

        elif typo_type == "missing":
            # Remove a character
            pos = random.randint(0, len(name) - 1)
            return name[:pos] + name[pos + 1:]

        return name

    def _remove_suffix(self, name: str) -> str:
        """Remove company suffix"""
        for suffix in self.company_suffixes:
            name = name.replace(f" {suffix}", "")
            name = name.replace(f" {suffix}.", "")

        return name.strip()
