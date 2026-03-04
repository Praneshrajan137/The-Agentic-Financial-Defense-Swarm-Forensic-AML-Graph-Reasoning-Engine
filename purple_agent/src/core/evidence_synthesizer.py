"""
Evidence Synthesizer -- Sherlock Holmes Multi-Modal Reasoning
PRD Reference: Task B4
Package Version: v8.0

CORE PRINCIPLE: Ledger data is GROUND TRUTH. Text discrepancy = EVIDENCE of crime.

v8.0 CRITICAL FIX [P4v8-01]:
  v7.0 switched to IFSC_COMPILED, PAN_COMPILED, SWIFT_COMPILED, IBAN_COMPILED
  from config.py. These patterns use ^$ anchors (VALIDATION mode) which
  require the ENTIRE input string to match. When used with .finditer(text),
  they return ZERO matches for identifiers embedded in longer text.

  v8.0 imports _SEARCH_COMPILED patterns which use \\b word boundaries:
    IFSC_SEARCH_COMPILED.finditer("via SBIN0001234 to") -> 1 match
    IFSC_COMPILED.finditer("via SBIN0001234 to")        -> 0 matches

  Both pattern sets exist in config.py:
    _COMPILED: for full-string validation (Protobuf ingestion boundary)
    _SEARCH_COMPILED: for finding identifiers within text (B4 evidence)

v8.0 BUG FIX [INR-CLEAN]:
  Prompt-level code only stripped "$" and "," from matched amounts.
  INR patterns (inr_symbol, inr_rs_prefix) produce raw strings containing
  "₹" or "Rs." prefixes which Decimal() cannot parse. Fixed by extending
  the cleaning pipeline to strip ₹ and Rs./Rs prefixes.

v7.0 [P4-12]: Sum comparison for component amounts.
v6.1 [ALN-06]: SPACY_MODEL_NAME from config.
v6.0: Lazy spaCy, amount dedup, zero-ledger fix.
"""
import re
import logging
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field

from src.config import (
    AMOUNT_PATTERNS,
    IFSC_SEARCH_COMPILED,
    PAN_SEARCH_COMPILED,
    SWIFT_SEARCH_COMPILED,
    IBAN_SEARCH_COMPILED,
    EVIDENCE_DISCREPANCY_THRESHOLD_USD,
    EVIDENCE_DISCREPANCY_THRESHOLD_INR,
    SPACY_MODEL_NAME,
)

logger = logging.getLogger(__name__)


@dataclass
class EvidenceResult:
    """Result of evidence synthesis."""
    verdict: str
    reasoning: str
    discrepancies: list[dict] = field(default_factory=list)
    extracted_entities: list[dict] = field(default_factory=list)
    extracted_amounts: list[Decimal] = field(default_factory=list)


class EvidenceSynthesizer:
    """
    Cross-references ledger data (ground truth) with unstructured text evidence.

    Two-pass entity extraction pipeline:
      Pass 1: spaCy statistical NER (PERSON, ORG, GPE, MONEY, DATE)
      Pass 2: Regex for structured financial IDs (IFSC, PAN, SWIFT, IBAN, refs)
              v8.0 [P4v8-01]: Uses _SEARCH_COMPILED patterns (word boundaries)
      Pass 3: Merge and deduplicate results

    spaCy model loaded lazily on first use (Rule 12).
    """

    def __init__(self) -> None:
        """Initialize synthesizer. spaCy model loaded on first use."""
        self._nlp = None
        self._nlp_loaded = False

    def _get_nlp(self):
        """
        Lazy-load spaCy model on first use.

        v6.1 [ALN-06]: Uses SPACY_MODEL_NAME from config.py.
        """
        if not self._nlp_loaded:
            self._nlp_loaded = True
            try:
                import spacy
                self._nlp = spacy.load(
                    SPACY_MODEL_NAME,
                    disable=["tagger", "parser", "lemmatizer"],
                )
                logger.info("spaCy %s loaded successfully.", SPACY_MODEL_NAME)
            except (OSError, ImportError) as e:
                self._nlp = None
                logger.warning(
                    "spaCy %s not available (%s). "
                    "Falling back to regex-only NER. "
                    "WARNING: This violates Rule 12 (spaCy + regex required).",
                    SPACY_MODEL_NAME, e,
                )
        return self._nlp

    def extract_entities_from_text(self, text: str) -> list[dict]:
        """
        Extract entities using spaCy NER + regex.

        v8.0 [P4v8-01]: Pass 2 uses _SEARCH_COMPILED patterns (word boundaries)
        instead of _COMPILED patterns (^$ anchors).

        Args:
            text: Unstructured text to extract entities from.

        Returns:
            Deduplicated list of entity dicts with text, label, source fields.
        """
        entities: list[dict] = []
        seen: set[tuple[str, str]] = set()

        # Pass 1: spaCy NER
        nlp = self._get_nlp()
        if nlp is not None:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "MONEY", "DATE"):
                    key = (ent.text, ent.label_)
                    if key not in seen:
                        seen.add(key)
                        entities.append({
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "source": "spacy_ner",
                        })

        # Pass 2: Regex (always runs regardless of spaCy availability)
        for match in re.finditer(
            r'(?:User|Account|Subject|Entity)\s+(\w+)', text, re.IGNORECASE,
        ):
            key = (match.group(0), "REFERENCE")
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": match.group(0),
                    "id": match.group(1),
                    "label": "REFERENCE",
                    "source": "regex_reference",
                })

        for match in IFSC_SEARCH_COMPILED.finditer(text):
            key = (match.group(), "IFSC")
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": match.group(),
                    "label": "IFSC",
                    "source": "regex_ifsc",
                })

        for match in PAN_SEARCH_COMPILED.finditer(text):
            key = (match.group(), "PAN")
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": match.group(),
                    "label": "PAN",
                    "source": "regex_pan",
                })

        for match in SWIFT_SEARCH_COMPILED.finditer(text):
            key = (match.group(), "SWIFT")
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": match.group(),
                    "label": "SWIFT",
                    "source": "regex_swift",
                })

        for match in IBAN_SEARCH_COMPILED.finditer(text):
            key = (match.group(), "IBAN")
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": match.group(),
                    "label": "IBAN",
                    "source": "regex_iban",
                })

        return entities

    def extract_amounts_from_text(self, text: str) -> list[Decimal]:
        """
        Extract monetary amounts from unstructured text.

        v8.0 [INR-CLEAN]: Extended cleaning to handle INR prefixes.
        v6.0 FIX: Deduplicates amounts via seen set.

        Args:
            text: Unstructured text.

        Returns:
            Deduplicated list of Decimal amounts.
        """
        amounts: list[Decimal] = []
        seen_amounts: set[Decimal] = set()

        for pattern_name, pattern in AMOUNT_PATTERNS.items():
            for match in re.finditer(pattern, text):
                try:
                    raw = match.group(0) if not match.groups() else match.group(1)

                    cleaned = (
                        raw.replace("$", "")
                        .replace("\u20b9", "")   # ₹ (U+20B9 Indian Rupee Sign)
                        .replace(",", "")
                    )
                    cleaned = re.sub(r"^Rs\.?\s*", "", cleaned).strip()

                    if pattern_name == "inr_lakh":
                        amount = Decimal(cleaned) * Decimal("100000")
                    elif pattern_name == "inr_crore":
                        amount = Decimal(cleaned) * Decimal("10000000")
                    elif pattern_name == "usd_k":
                        amount = Decimal(cleaned) * Decimal("1000")
                    else:
                        amount = Decimal(cleaned)

                    if not amount.is_finite():
                        logger.warning(
                            "Non-finite amount %s from pattern %s — skipped.",
                            amount, pattern_name,
                        )
                        continue

                    if amount not in seen_amounts:
                        seen_amounts.add(amount)
                        amounts.append(amount)
                except (InvalidOperation, ValueError):
                    continue

        return amounts

    def synthesize(
        self,
        ledger_data: dict,
        text_evidence: list[dict | str],
        currency: str = "USD",
    ) -> EvidenceResult:
        """
        Cross-reference ledger (ground truth) with text evidence.

        v7.0 [P4-12]: Two-level comparison:
          Level 1: Each text amount vs ledger total.
          Level 2: sum(text amounts) vs ledger total.

        Args:
            ledger_data: Dict with "total_inflow", "entity_id", "transactions".
            text_evidence: List of evidence dicts (with "content" key) or plain strings.
            currency: "USD" or "INR" for jurisdiction-specific thresholds.

        Returns:
            EvidenceResult with verdict, reasoning, and extracted data.
        """
        if not text_evidence:
            return EvidenceResult(
                verdict="INSUFFICIENT_DATA",
                reasoning="No unstructured text evidence available for cross-reference.",
            )

        ledger_amount = ledger_data.get("total_inflow")
        entity_id = ledger_data.get("entity_id", "unknown")

        if ledger_amount is None:
            return EvidenceResult(
                verdict="INSUFFICIENT_DATA",
                reasoning=f"No ledger inflow data available for entity {entity_id}.",
            )

        threshold = (
            EVIDENCE_DISCREPANCY_THRESHOLD_INR
            if currency == "INR"
            else EVIDENCE_DISCREPANCY_THRESHOLD_USD
        )

        all_entities: list[dict] = []
        all_amounts: list[Decimal] = []
        per_amount_discrepancies: list[dict] = []

        for evidence in text_evidence:
            text = evidence["content"] if isinstance(evidence, dict) else evidence

            entities = self.extract_entities_from_text(text)
            all_entities.extend(entities)

            amounts = self.extract_amounts_from_text(text)
            all_amounts.extend(amounts)

            for text_amount in amounts:
                if text_amount != ledger_amount:
                    diff = abs(text_amount - ledger_amount)
                    if diff > threshold:
                        per_amount_discrepancies.append({
                            "type": "AMOUNT_MISMATCH",
                            "text_claims": str(text_amount),
                            "ledger_shows": str(ledger_amount),
                            "difference": str(diff),
                            "entity": entity_id,
                        })

        # v7.0 [P4-12]: Level 2 — sum comparison clears false positives
        # when individual components (e.g., $5K + $3K) add up to ledger ($8K).
        discrepancies = per_amount_discrepancies
        if per_amount_discrepancies and len(all_amounts) >= 2:
            amount_sum = sum(all_amounts)
            sum_diff = abs(amount_sum - ledger_amount)
            if sum_diff <= threshold:
                logger.info(
                    "Entity %s: %d per-amount discrepancy(ies) cleared by sum "
                    "comparison. sum(text_amounts)=%s vs ledger=%s (diff=%s "
                    "<= threshold=%s).",
                    entity_id, len(per_amount_discrepancies),
                    amount_sum, ledger_amount, sum_diff, threshold,
                )
                discrepancies = []

        if discrepancies:
            verdict = "SUSPICIOUS_DISCREPANCY"
            reasoning = (
                f"DISCREPANCY DETECTED for entity {entity_id}: "
                f"Text evidence claims amounts that differ from ledger records. "
                f"Ledger (ground truth) shows {ledger_amount} {currency}. "
                f"Found {len(discrepancies)} discrepancy(ies). "
                f"This is EVIDENCE of potential structuring or misrepresentation."
            )
        else:
            verdict = "CONSISTENT"
            reasoning = (
                f"Text evidence for entity {entity_id} is consistent with "
                f"ledger data. "
                f"No discrepancies detected across {len(text_evidence)} "
                f"evidence items."
            )

        return EvidenceResult(
            verdict=verdict,
            reasoning=reasoning,
            discrepancies=discrepancies,
            extracted_entities=all_entities,
            extracted_amounts=all_amounts,
        )
