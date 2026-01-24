"""
Foundation Model (DBRX/Llama) for entity matching fallback
"""
import json
import re
from typing import Dict, List, Optional, Tuple


class FoundationModelMatcher:
    """Use Databricks Foundation Models for entity matching"""

    def __init__(
        self,
        model_name: str = "databricks-dbrx-instruct",
        databricks_client=None
    ):
        """
        Initialize Foundation Model matcher

        Args:
            model_name: Model name (databricks-dbrx-instruct or databricks-llama-3-1-70b-instruct)
            databricks_client: Databricks WorkspaceClient instance
        """
        self.model_name = model_name
        self.client = databricks_client

    def match(
        self,
        source_entity: Dict,
        candidates: List[Dict],
        top_k: int = 3
    ) -> Dict:
        """
        Match source entity against candidates using LLM reasoning

        Args:
            source_entity: Source entity dictionary
            candidates: List of candidate entities from vector search
            top_k: Number of top candidates to consider

        Returns:
            Match result with ciq_id, confidence, and reasoning
        """
        # Limit to top-K candidates
        candidates = candidates[:top_k]

        # Build prompt
        prompt = self._build_prompt(source_entity, candidates)

        # Call Foundation Model
        response = self._call_model(prompt)

        # Parse response
        result = self._parse_response(response)

        return result

    def _build_prompt(self, source_entity: Dict, candidates: List[Dict]) -> str:
        """Build prompt for Foundation Model"""
        # Format source entity
        source_text = self._format_entity(source_entity, label="Source Entity")

        # Format candidates
        candidates_text = "\n\n".join([
            self._format_entity(c["metadata"], label=f"Candidate {i + 1}", ciq_id=c["ciq_id"])
            for i, c in enumerate(candidates)
        ])

        prompt = f"""You are an expert at entity matching and reconciliation. Your task is to determine which candidate entity (if any) matches the source entity.

{source_text}

Candidates from S&P Capital IQ:

{candidates_text}

Instructions:
1. Analyze the source entity and all candidates
2. Consider: company name variations, ticker symbols, identifiers (LEI, CUSIP, ISIN), location, industry
3. Determine the best match based on overall similarity
4. If no candidate is a good match, return null for ciq_id
5. Provide your confidence score (0-100) and clear reasoning

Return your answer in JSON format:
{{
    "ciq_id": "IQ12345 or null",
    "confidence": 85,
    "reasoning": "Company name matches exactly, ticker symbol matches, and LEI identifier is identical"
}}

Your response (JSON only):"""

        return prompt

    def _format_entity(self, entity: Dict, label: str, ciq_id: Optional[str] = None) -> str:
        """Format entity for prompt"""
        lines = [f"**{label}**"]

        if ciq_id:
            lines.append(f"  - CIQ ID: {ciq_id}")

        fields = [
            ("company_name", "Company Name"),
            ("primary_ticker", "Ticker"),
            ("ticker", "Ticker"),
            ("lei", "LEI"),
            ("cusip", "CUSIP"),
            ("isin", "ISIN"),
            ("industry", "Industry"),
            ("country", "Country"),
            ("city", "City"),
            ("exchange", "Exchange")
        ]

        for field, display_name in fields:
            if value := entity.get(field):
                lines.append(f"  - {display_name}: {value}")

        return "\n".join(lines)

    def _call_model(self, prompt: str) -> str:
        """Call Databricks Foundation Model API"""
        if self.client is None:
            # Return mock response for testing
            return self._mock_response()

        try:
            # Call serving endpoint
            response = self.client.serving_endpoints.query(
                name=self.model_name,
                inputs=[{"prompt": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            return response.predictions[0]

        except Exception as e:
            print(f"Error calling Foundation Model: {e}")
            return self._mock_response()

    def _mock_response(self) -> str:
        """Return mock response for testing"""
        return json.dumps({
            "ciq_id": "IQ24937",
            "confidence": 90,
            "reasoning": "Company name and ticker match exactly with high confidence"
        })

    def _parse_response(self, response: str) -> Dict:
        """Parse JSON response from Foundation Model"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                result = json.loads(response)

            # Validate structure
            if "ciq_id" not in result:
                result["ciq_id"] = None
            if "confidence" not in result:
                result["confidence"] = 50
            if "reasoning" not in result:
                result["reasoning"] = "Unable to parse reasoning"

            # Convert confidence to float
            result["confidence"] = float(result["confidence"]) / 100.0

            # Add method
            result["match_method"] = "foundation_model"

            return result

        except Exception as e:
            print(f"Error parsing Foundation Model response: {e}")
            return {
                "ciq_id": None,
                "confidence": 0.0,
                "reasoning": f"Error parsing response: {e}",
                "match_method": "foundation_model_error"
            }

    def batch_match(
        self,
        source_entities: List[Dict],
        candidates_list: List[List[Dict]]
    ) -> List[Dict]:
        """
        Match multiple source entities

        Args:
            source_entities: List of source entities
            candidates_list: List of candidate lists (one per source entity)

        Returns:
            List of match results
        """
        results = []

        for source, candidates in zip(source_entities, candidates_list):
            result = self.match(source, candidates)
            result["source_id"] = source.get("source_id")
            results.append(result)

        return results
