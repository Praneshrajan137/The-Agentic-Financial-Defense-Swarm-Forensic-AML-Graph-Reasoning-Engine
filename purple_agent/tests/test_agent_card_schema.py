"""
Test: Agent Card JSON Schema Validation
PRD Reference: Task A2
"""
import json
import pytest
from pathlib import Path

AGENT_CARD_PATH = Path(__file__).parent.parent / "agent.json"


class TestAgentCardSchema:
    @pytest.fixture(autouse=True)
    def load_card(self):
        with open(AGENT_CARD_PATH) as f:
            self.card = json.load(f)

    def test_required_top_level_fields(self):
        required = {"name", "version", "description", "capabilities", "communication"}
        if not required.issubset(self.card.keys()):
            raise ValueError(
                f"Missing top-level fields: {required - self.card.keys()}"
            )

    def test_name_matches_project(self):
        if self.card["name"] != "Project_Gamma_Purple_Hunter":
            raise ValueError(
                f"Expected name 'Project_Gamma_Purple_Hunter', got '{self.card['name']}'"
            )

    def test_version_is_semver(self):
        parts = self.card["version"].split(".")
        if len(parts) != 3:
            raise ValueError(f"Version must be semver (3 parts), got {self.card['version']}")
        for p in parts:
            if not p.isdigit():
                raise ValueError(f"Version part '{p}' is not a digit")

    def test_capabilities_include_required(self):
        caps = self.card["capabilities"]
        if "financial_crime_detection" not in caps:
            raise ValueError("Missing capability: financial_crime_detection")
        if "sar_generation" not in caps:
            raise ValueError("Missing capability: sar_generation")

    def test_communication_endpoint(self):
        comm = self.card["communication"]
        if comm["protocol"] != "a2a":
            raise ValueError(f"Expected protocol 'a2a', got '{comm['protocol']}'")
        if "/a2a" not in comm["endpoint"]:
            raise ValueError(f"Endpoint must contain '/a2a', got '{comm['endpoint']}'")
        if "protobuf" not in comm["content_types"]:
            raise ValueError("content_types must include 'protobuf'")
        if "json" not in comm["content_types"]:
            raise ValueError("content_types must include 'json'")

    def test_confidence_score_type(self):
        for field in self.card.get("output_schema", {}).get("fields", []):
            if field.get("name") == "confidence_score":
                if field["type"] != "number":
                    raise ValueError(
                        f"confidence_score type must be 'number', got '{field['type']}'"
                    )
                return
        # confidence_score field not found is acceptable if output_schema is optional

    def test_health_check_endpoint(self):
        if "health_check" not in self.card["communication"]:
            raise ValueError("communication must include health_check")
        if self.card["communication"]["health_check"] != "/health":
            raise ValueError(
                f"Expected health_check '/health', got '{self.card['communication']['health_check']}'"
            )
