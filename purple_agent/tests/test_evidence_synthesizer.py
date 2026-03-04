"""
Tests for Evidence Synthesizer -- Purple Agent v8.0
PRD Reference: Task B4
TDD: Tests written FIRST, implementation SECOND.

Tests cover:
  - Amount discrepancy detection (text vs ledger cross-reference)
  - Entity extraction via spaCy NER (PERSON, ORG, MONEY)
  - Regex extraction (IFSC, PAN, SWIFT, IBAN, references)
  - Merge and deduplication of spaCy + regex results
  - Amount deduplication ("$10,000 USD" -> one Decimal)
  - Zero-ledger and None-ledger edge cases
  - Sum comparison for partial amounts (v7.0 P4-12)
  - v8.0 [P4v8-01]: SEARCH_COMPILED patterns find embedded codes
  - v8.0 [P4v8-01]: COMPILED (anchored) patterns reject embedded codes
"""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.core.evidence_synthesizer import EvidenceSynthesizer, EvidenceResult
from src.config import (
    IFSC_COMPILED,
    PAN_COMPILED,
    SWIFT_COMPILED,
    IBAN_COMPILED,
    IFSC_SEARCH_COMPILED,
    PAN_SEARCH_COMPILED,
    SWIFT_SEARCH_COMPILED,
    IBAN_SEARCH_COMPILED,
    EVIDENCE_DISCREPANCY_THRESHOLD_USD,
)


# ===================================================================
# Helper: Mock spaCy NLP for deterministic testing
# ===================================================================

def _make_mock_nlp(entities: list[tuple[str, str, int, int]]):
    """
    Create a mock spaCy Language that returns controlled entities.

    Args:
        entities: List of (text, label, start_char, end_char) tuples.

    Returns:
        Mock nlp callable producing a mock Doc with the given entities.
    """
    def _mock_call(text: str):
        doc = MagicMock()
        mock_ents = []
        for ent_text, label, start, end in entities:
            ent = MagicMock()
            ent.text = ent_text
            ent.label_ = label
            ent.start_char = start
            ent.end_char = end
            mock_ents.append(ent)
        doc.ents = mock_ents
        return doc
    return _mock_call


def _synth_with_mock_nlp(
    entities: list[tuple[str, str, int, int]] | None = None,
) -> EvidenceSynthesizer:
    """Create an EvidenceSynthesizer with optionally mocked spaCy."""
    synth = EvidenceSynthesizer()
    if entities is not None:
        synth._nlp = _make_mock_nlp(entities)
        synth._nlp_loaded = True
    else:
        synth._nlp = None
        synth._nlp_loaded = True
    return synth


# ===================================================================
# TestAmountDiscrepancy
# ===================================================================

class TestAmountDiscrepancy:

    def test_detects_amount_discrepancy(self) -> None:
        """Email says $10K, ledger says $9.5K -> SUSPICIOUS_DISCREPANCY."""
        synth = _synth_with_mock_nlp(entities=[])
        result = synth.synthesize(
            ledger_data={"total_inflow": Decimal("9500"), "entity_id": "src_01"},
            text_evidence=["Confirming transfer of $10,000 to Shell Corp LLC."],
            currency="USD",
        )
        assert result.verdict == "SUSPICIOUS_DISCREPANCY"
        assert len(result.discrepancies) >= 1
        assert any(d["type"] == "AMOUNT_MISMATCH" for d in result.discrepancies)

    def test_consistent_data_no_discrepancy(self) -> None:
        """Text and ledger agree on amount -> CONSISTENT."""
        synth = _synth_with_mock_nlp(entities=[])
        result = synth.synthesize(
            ledger_data={"total_inflow": Decimal("10000"), "entity_id": "src_01"},
            text_evidence=["Confirming transfer of $10,000 complete."],
            currency="USD",
        )
        assert result.verdict == "CONSISTENT"
        assert len(result.discrepancies) == 0

    def test_no_text_evidence_returns_insufficient_data(self) -> None:
        """Empty text evidence list -> INSUFFICIENT_DATA."""
        synth = _synth_with_mock_nlp(entities=[])
        result = synth.synthesize(
            ledger_data={"total_inflow": Decimal("10000"), "entity_id": "src_01"},
            text_evidence=[],
            currency="USD",
        )
        assert result.verdict == "INSUFFICIENT_DATA"

    def test_none_ledger_returns_insufficient_data(self) -> None:
        """Ledger total_inflow is None -> INSUFFICIENT_DATA."""
        synth = _synth_with_mock_nlp(entities=[])
        result = synth.synthesize(
            ledger_data={"total_inflow": None, "entity_id": "src_01"},
            text_evidence=["Some text evidence about $5,000."],
            currency="USD",
        )
        assert result.verdict == "INSUFFICIENT_DATA"

    def test_zero_ledger_still_checks_text(self) -> None:
        """
        Ledger=0 but text claims $50K -> SUSPICIOUS_DISCREPANCY.
        Rule: check 'ledger_amount is not None', NOT 'ledger_amount > 0'.
        Zero-value ledger with non-zero text claim is evidence of fraud.
        """
        synth = _synth_with_mock_nlp(entities=[])
        result = synth.synthesize(
            ledger_data={"total_inflow": Decimal("0"), "entity_id": "entity_x"},
            text_evidence=["Payment of $50,000 was processed."],
            currency="USD",
        )
        assert result.verdict == "SUSPICIOUS_DISCREPANCY"
        assert len(result.discrepancies) >= 1

    def test_partial_amounts_sum_to_ledger(self) -> None:
        """
        v7.0 [P4-16]: $5K + $3K text, $8K ledger -> CONSISTENT.
        Per-amount checks flag both (|5000-8000|>100, |3000-8000|>100),
        but sum comparison (5000+3000=8000 vs 8000) clears them.
        """
        synth = _synth_with_mock_nlp(entities=[])
        result = synth.synthesize(
            ledger_data={"total_inflow": Decimal("8000"), "entity_id": "ent_1"},
            text_evidence=[
                "First installment of $5,000 sent.",
                "Second installment of $3,000 sent.",
            ],
            currency="USD",
        )
        assert result.verdict == "CONSISTENT"
        assert len(result.discrepancies) == 0


# ===================================================================
# TestSpacyNER
# ===================================================================

class TestSpacyNER:

    def test_spacy_extracts_person_entities(self) -> None:
        """PERSON: 'John Smith' extracted via spaCy NER."""
        synth = _synth_with_mock_nlp(
            entities=[("John Smith", "PERSON", 0, 10)]
        )
        entities = synth.extract_entities_from_text("John Smith sent funds.")
        person_ents = [e for e in entities if e["label"] == "PERSON"]
        assert len(person_ents) >= 1
        assert person_ents[0]["text"] == "John Smith"
        assert person_ents[0]["source"] == "spacy_ner"

    def test_spacy_extracts_org_entities(self) -> None:
        """ORG: 'Shell Corp LLC' extracted via spaCy NER."""
        synth = _synth_with_mock_nlp(
            entities=[("Shell Corp LLC", "ORG", 12, 26)]
        )
        entities = synth.extract_entities_from_text(
            "Transfer to Shell Corp LLC completed."
        )
        org_ents = [e for e in entities if e["label"] == "ORG"]
        assert len(org_ents) >= 1
        assert org_ents[0]["text"] == "Shell Corp LLC"
        assert org_ents[0]["source"] == "spacy_ner"

    def test_spacy_extracts_money_entities(self) -> None:
        """MONEY: '$10,000' extracted via spaCy NER."""
        synth = _synth_with_mock_nlp(
            entities=[("$10,000", "MONEY", 15, 22)]
        )
        entities = synth.extract_entities_from_text(
            "Transfer amount $10,000 received."
        )
        money_ents = [e for e in entities if e["label"] == "MONEY"]
        assert len(money_ents) >= 1
        assert money_ents[0]["text"] == "$10,000"


# ===================================================================
# TestRegexExtraction
# ===================================================================

class TestRegexExtraction:

    def test_extracts_entity_references_from_text(self) -> None:
        """Regex extracts 'User 8492' and 'Account ACCT' references."""
        synth = _synth_with_mock_nlp(entities=[])
        text = "User 8492 sent funds to Account ACCT-src_01."
        entities = synth.extract_entities_from_text(text)
        ref_ents = [e for e in entities if e["label"] == "REFERENCE"]
        ref_ids = [e["id"] for e in ref_ents]
        assert "8492" in ref_ids
        assert "ACCT" in ref_ids

    def test_regex_extracts_ifsc(self) -> None:
        """IFSC: SBIN0001234 extracted from embedded text."""
        synth = _synth_with_mock_nlp(entities=[])
        entities = synth.extract_entities_from_text(
            "Branch IFSC: SBIN0001234 verified."
        )
        ifsc_ents = [e for e in entities if e["label"] == "IFSC"]
        assert len(ifsc_ents) >= 1
        assert ifsc_ents[0]["text"] == "SBIN0001234"

    def test_regex_extracts_pan(self) -> None:
        """PAN: ABCDE1234F extracted from embedded text."""
        synth = _synth_with_mock_nlp(entities=[])
        entities = synth.extract_entities_from_text(
            "Customer PAN: ABCDE1234F on file."
        )
        pan_ents = [e for e in entities if e["label"] == "PAN"]
        assert len(pan_ents) >= 1
        assert pan_ents[0]["text"] == "ABCDE1234F"

    def test_regex_extracts_swift(self) -> None:
        """SWIFT: SBININBB extracted from embedded text."""
        synth = _synth_with_mock_nlp(entities=[])
        entities = synth.extract_entities_from_text(
            "Routed via SWIFT: SBININBB for transfer."
        )
        swift_ents = [e for e in entities if e["label"] == "SWIFT"]
        assert len(swift_ents) >= 1
        assert swift_ents[0]["text"] == "SBININBB"

    def test_regex_extracts_iban(self) -> None:
        """IBAN: DE89370400440532013000 extracted from embedded text."""
        synth = _synth_with_mock_nlp(entities=[])
        entities = synth.extract_entities_from_text(
            "Beneficiary IBAN: DE89370400440532013000 confirmed."
        )
        iban_ents = [e for e in entities if e["label"] == "IBAN"]
        assert len(iban_ents) >= 1
        assert iban_ents[0]["text"] == "DE89370400440532013000"


# ===================================================================
# TestMergeAndDedup
# ===================================================================

class TestMergeAndDedup:

    def test_merge_spacy_and_regex_results(self) -> None:
        """
        spaCy + regex results merged without true duplicates.
        Same text with different labels (ORG from spaCy, IFSC from regex)
        produces distinct entries. spaCy PERSON entity also preserved.
        """
        synth = _synth_with_mock_nlp(
            entities=[
                ("John Smith", "PERSON", 0, 10),
                ("SBIN0001234", "ORG", 31, 42),
            ]
        )
        text = "John Smith transferred via IFSC SBIN0001234 to account."
        entities = synth.extract_entities_from_text(text)

        labels_for_sbin = {
            e["label"] for e in entities if e["text"] == "SBIN0001234"
        }
        assert "ORG" in labels_for_sbin
        assert "IFSC" in labels_for_sbin

        assert any(
            e["text"] == "John Smith" and e["label"] == "PERSON"
            for e in entities
        )

    def test_amount_deduplication(self) -> None:
        """
        '$10,000 USD' produces ONE Decimal, not two.
        usd_symbol matches '$10,000' -> 10000.
        currency_code matches '10,000 USD' -> 10000.
        Dedup yields exactly one Decimal('10000').
        """
        synth = _synth_with_mock_nlp(entities=[])
        amounts = synth.extract_amounts_from_text(
            "Transfer of $10,000 USD completed."
        )
        assert Decimal("10000") in amounts
        assert amounts.count(Decimal("10000")) == 1


# ===================================================================
# TestV8SearchPatterns
# ===================================================================

class TestV8SearchPatterns:

    def test_search_patterns_find_embedded_codes(self) -> None:
        """v8.0 [P4v8-01]: _SEARCH_COMPILED finds identifiers within text."""
        ifsc_text = "Sent via SBIN0001234 to account"
        assert len(IFSC_SEARCH_COMPILED.findall(ifsc_text)) >= 1

        pan_text = "Customer PAN ABCPP1234F registered"
        assert len(PAN_SEARCH_COMPILED.findall(pan_text)) >= 1

        swift_text = "SWIFT code SBININBB used for the transfer"
        assert len(SWIFT_SEARCH_COMPILED.findall(swift_text)) >= 1

        iban_text = "IBAN DE89370400440532013000 confirmed"
        assert len(IBAN_SEARCH_COMPILED.findall(iban_text)) >= 1

    def test_validation_patterns_do_not_match_embedded(self) -> None:
        """v8.0 [P4v8-01]: _COMPILED (anchored ^$) patterns reject embedded text."""
        assert len(list(IFSC_COMPILED.finditer(
            "Sent via SBIN0001234 to account"
        ))) == 0

        assert len(list(PAN_COMPILED.finditer(
            "PAN code ABCPP1234F on file"
        ))) == 0

        assert len(list(SWIFT_COMPILED.finditer(
            "SWIFT code SBININBB used"
        ))) == 0

        assert len(list(IBAN_COMPILED.finditer(
            "IBAN DE89370400440532013000 confirmed"
        ))) == 0
