"""
Structured Result Types -- Green Agent v8.0
============================================
Typed dataclasses for assessment results, replacing raw dicts.

These types enforce type safety and make the assessment engine's outputs
inspectable and testable. They mirror the Purple Agent's structured
result types (StructuringResult, LayeringResult, etc.) but from the
*judge's* perspective -- scoring Purple's investigation quality.

CRITICAL: All monetary fields are Decimal. NEVER float.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Dict, Any, Optional


# ═══════════════════════════════════════════════════════════════════
# ENTITY-LEVEL METRICS (Precision / Recall / F1)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EntityMetrics:
    """Precision/Recall/F1 for entity identification.

    Measures how well Purple Agent identified the criminal entities
    from the ground truth injected by the Green Agent.
    """
    true_positives: List[str] = field(default_factory=list)
    false_positives: List[str] = field(default_factory=list)
    false_negatives: List[str] = field(default_factory=list)

    @property
    def precision(self) -> Decimal:
        """Of all entities Purple flagged, how many were actual criminals?"""
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        if tp + fp == 0:
            return Decimal("0")
        return Decimal(str(round(tp / (tp + fp), 4)))

    @property
    def recall(self) -> Decimal:
        """Of all actual criminals, how many did Purple find?"""
        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        if tp + fn == 0:
            return Decimal("0")
        return Decimal(str(round(tp / (tp + fn), 4)))

    @property
    def f1(self) -> Decimal:
        """Harmonic mean of precision and recall."""
        p = self.precision
        r = self.recall
        if p + r == Decimal("0"):
            return Decimal("0")
        return Decimal(str(round(2 * float(p) * float(r) / (float(p) + float(r)), 4)))


# ═══════════════════════════════════════════════════════════════════
# HALLUCINATION DETECTION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HallucinationCheck:
    """Result of checking Purple's SAR for hallucinated entities/amounts.

    A hallucination is any entity ID or monetary amount cited in the SAR
    that does NOT exist in the transaction graph.
    """
    hallucinated_entities: List[str] = field(default_factory=list)
    hallucinated_amounts: List[str] = field(default_factory=list)
    passed: bool = True
    details: str = ""

    @property
    def total_hallucinations(self) -> int:
        return len(self.hallucinated_entities) + len(self.hallucinated_amounts)


# ═══════════════════════════════════════════════════════════════════
# SAR FIVE Ws VALIDATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FiveWsValidation:
    """Validation of SAR narrative's Five Ws structure.

    A complete SAR must answer: WHO, WHAT, WHERE, WHEN, WHY.
    """
    who_present: bool = False
    what_present: bool = False
    where_present: bool = False
    when_present: bool = False
    why_present: bool = False
    missing_sections: List[str] = field(default_factory=list)

    @property
    def completeness_score(self) -> Decimal:
        """Fraction of Five Ws present (0.0 to 1.0)."""
        present = sum([
            self.who_present, self.what_present, self.where_present,
            self.when_present, self.why_present
        ])
        return Decimal(str(round(present / 5, 2)))

    @property
    def is_complete(self) -> bool:
        return len(self.missing_sections) == 0


# ═══════════════════════════════════════════════════════════════════
# TYPOLOGY SCORING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TypologyScore:
    """Score for whether Purple correctly identified the crime typology."""
    expected_typology: str = ""  # "structuring" or "layering"
    detected_typology: str = ""  # what Purple reported
    correct: bool = False
    confidence: Decimal = Decimal("0")
    reasoning: str = ""

    @property
    def score(self) -> Decimal:
        """1.0 if correct, 0.0 if wrong."""
        return Decimal("1") if self.correct else Decimal("0")


# ═══════════════════════════════════════════════════════════════════
# EFFICIENCY SCORING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EfficiencyScore:
    """Score based on Purple Agent's tool call count."""
    tool_call_count: int = 0
    tier: str = ""  # "excellent", "good", "fair", "poor"
    score: Decimal = Decimal("0")


# ═══════════════════════════════════════════════════════════════════
# OVERALL ASSESSMENT RESULT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AssessmentResult:
    """Complete assessment of Purple Agent's investigation.

    This is the top-level result returned by the assessment engine.
    It aggregates all sub-scores into a weighted final score.
    """
    # Sub-scores
    entity_metrics: EntityMetrics = field(default_factory=EntityMetrics)
    hallucination_check: HallucinationCheck = field(default_factory=HallucinationCheck)
    five_ws: FiveWsValidation = field(default_factory=FiveWsValidation)
    typology: TypologyScore = field(default_factory=TypologyScore)
    efficiency: EfficiencyScore = field(default_factory=EfficiencyScore)

    # Weighted final score
    overall_score: Decimal = Decimal("0")

    # Rubric breakdown (populated by assessment engine)
    rubric_breakdown: Dict[str, Decimal] = field(default_factory=dict)

    # Raw data for debugging
    raw_submission: Dict[str, Any] = field(default_factory=dict)
    ground_truth_summary: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    jurisdiction: str = "FinCEN"  # or "FIU-IND"
    currency: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON response. Converts Decimals to str."""
        return {
            "overall_score": str(self.overall_score),
            "entity_metrics": {
                "precision": str(self.entity_metrics.precision),
                "recall": str(self.entity_metrics.recall),
                "f1": str(self.entity_metrics.f1),
                "true_positives": self.entity_metrics.true_positives,
                "false_positives": self.entity_metrics.false_positives,
                "false_negatives": self.entity_metrics.false_negatives,
            },
            "hallucination_check": {
                "passed": self.hallucination_check.passed,
                "total_hallucinations": self.hallucination_check.total_hallucinations,
                "hallucinated_entities": self.hallucination_check.hallucinated_entities,
                "hallucinated_amounts": self.hallucination_check.hallucinated_amounts,
                "details": self.hallucination_check.details,
            },
            "five_ws": {
                "completeness_score": str(self.five_ws.completeness_score),
                "is_complete": self.five_ws.is_complete,
                "missing_sections": self.five_ws.missing_sections,
            },
            "typology": {
                "expected": self.typology.expected_typology,
                "detected": self.typology.detected_typology,
                "correct": self.typology.correct,
                "score": str(self.typology.score),
            },
            "efficiency": {
                "tool_call_count": self.efficiency.tool_call_count,
                "tier": self.efficiency.tier,
                "score": str(self.efficiency.score),
            },
            "rubric_breakdown": {
                k: str(v) for k, v in self.rubric_breakdown.items()
            },
            "jurisdiction": self.jurisdiction,
            "currency": self.currency,
        }


__all__ = [
    "EntityMetrics",
    "HallucinationCheck",
    "FiveWsValidation",
    "TypologyScore",
    "EfficiencyScore",
    "AssessmentResult",
]
