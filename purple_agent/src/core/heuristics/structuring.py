"""
Structuring (Smurfing) Detection Heuristic
PRD Reference: Task B2
Package Version: v8.0

Detects deliberate fragmentation of large transactions into smaller ones
to evade Currency Transaction Report (CTR) thresholds.

Detection Modes:
  - Fan-In:  Many sources -> one collector (classic smurfing)
  - Fan-Out: One source -> many targets (U-Turn structuring)

v8.0 [P4v8-02]: Transactions are FILTERED BY CURRENCY before threshold
comparison. Rule 14: "Group by currency BEFORE comparing against thresholds."
v7.0 passed all transactions regardless of currency, causing INR amounts to
be incorrectly matched against USD thresholds in mixed-currency scenarios.

v7.0 [P4-08]: Fan-in is checked FIRST and takes precedence.

Thresholds are jurisdiction-aware:
  - USD: $9,000 - $9,800 per transaction (FinCEN CTR = $10,000)
  - INR: 9,00,000 - 9,80,000 per transaction (PMLA CTR = 10,00,000)

CRITICAL: All amounts are Decimal. All comparisons use Decimal arithmetic.
"""
import logging
from dataclasses import dataclass, field
from decimal import Decimal

from src.config import (
    STRUCTURING_MIN_AMOUNT_USD,
    STRUCTURING_MAX_AMOUNT_USD,
    STRUCTURING_MIN_AMOUNT_INR,
    STRUCTURING_MAX_AMOUNT_INR,
    STRUCTURING_MIN_COUNT,
    STRUCTURING_MIN_COUNT_INR,
    STRUCTURING_TIME_WINDOW_SECONDS,
    STRUCTURING_TIME_WINDOW_SECONDS_INR,
)

logger = logging.getLogger(__name__)


@dataclass
class StructuringResult:
    """Result of structuring detection on a single node.

    v7.0 [P4-14]: counterpart_nodes contains:
      - Fan-In mode: the source node IDs (depositors / smurfs)
      - Fan-Out mode: the target node IDs (recipients)
    """
    detected: bool
    mode: str
    target_node: str
    counterpart_nodes: list[str] = field(default_factory=list)
    qualifying_transactions: list[dict] = field(default_factory=list)
    total_amount: Decimal = Decimal("0")
    transaction_count: int = 0
    currency: str = "USD"
    confidence: Decimal = Decimal("0")
    reasoning: str = ""

    @property
    def source_nodes(self) -> list[str]:
        """Backward-compatible alias for counterpart_nodes."""
        return self.counterpart_nodes


def _get_thresholds(currency: str) -> tuple[Decimal, Decimal, int, int]:
    """Get jurisdiction-specific thresholds."""
    if currency == "INR":
        return (
            STRUCTURING_MIN_AMOUNT_INR,
            STRUCTURING_MAX_AMOUNT_INR,
            STRUCTURING_MIN_COUNT_INR,
            STRUCTURING_TIME_WINDOW_SECONDS_INR,
        )
    return (
        STRUCTURING_MIN_AMOUNT_USD,
        STRUCTURING_MAX_AMOUNT_USD,
        STRUCTURING_MIN_COUNT,
        STRUCTURING_TIME_WINDOW_SECONDS,
    )


def _filter_by_currency(transactions: list[dict], currency: str) -> list[dict]:
    """
    Filter transactions to only those matching the target currency.

    v8.0 [P4v8-02]: Implements Rule 14 -- "Group by currency BEFORE
    comparing against thresholds." Without this filter, an INR transaction
    of 9,200 would incorrectly match USD $9,000-$9,800 thresholds.

    Args:
        transactions: All transactions (any currency).
        currency: Target currency to filter for ("USD" or "INR").

    Returns:
        Only transactions where tx["currency"] == currency.
    """
    return [tx for tx in transactions if tx.get("currency") == currency]


def _filter_qualifying_transactions(
    transactions: list[dict],
    min_amount: Decimal,
    max_amount: Decimal,
    time_window_seconds: int,
) -> list[dict]:
    """
    Filter transactions within the structuring amount range and time window.

    Uses a FIXED time window: [latest_timestamp - window, latest_timestamp].

    Args:
        transactions: List of CURRENCY-FILTERED transaction dicts.
        min_amount: Minimum amount (inclusive) for structuring range.
        max_amount: Maximum amount (inclusive) for structuring range.
        time_window_seconds: Time window in seconds.

    Returns:
        List of qualifying transactions within range and window.
    """
    if not transactions:
        return []

    amount_qualifying = [
        tx for tx in transactions
        if min_amount <= tx["amount"] <= max_amount
    ]
    if not amount_qualifying:
        return []

    latest_ts = max(tx["timestamp"] for tx in amount_qualifying)
    window_start = latest_ts - time_window_seconds

    return [tx for tx in amount_qualifying if tx["timestamp"] >= window_start]


def _compute_confidence(
    qualifying_count: int,
    unique_counterparts: int,
    min_count: int,
    time_window_seconds: int,
    qualifying_transactions: list[dict],
) -> Decimal:
    """
    Compute multi-factor confidence score for structuring detection.

    v7.0 [P4-09]: Multi-factor replaces binary min(count/min_count, 1).
    v8.0 [P4v8-09]: time_factor uses integer arithmetic (time_window // 4).

    Three factors:
      (a) Count ratio: [0, 0.5]. Saturates at 3x min_count.
      (b) Source diversity: [0, 0.3]. More unique counterparties = higher.
      (c) Time clustering: 0.2 if all TXs within 25% of time window.
    """
    count_factor = min(
        Decimal(str(qualifying_count)) / Decimal(str(min_count * 3)),
        Decimal("0.5"),
    )

    if qualifying_count > 0:
        diversity_factor = min(
            Decimal(str(unique_counterparts)) / Decimal(str(qualifying_count)),
            Decimal("1"),
        ) * Decimal("0.3")
    else:
        diversity_factor = Decimal("0")

    time_factor = Decimal("0")
    if qualifying_transactions:
        timestamps = [tx["timestamp"] for tx in qualifying_transactions]
        time_span = max(timestamps) - min(timestamps)
        tight_window = time_window_seconds // 4
        if time_span <= tight_window:
            time_factor = Decimal("0.2")

    confidence = min(count_factor + diversity_factor + time_factor, Decimal("1"))
    return confidence


def detect_structuring(
    graph_reasoner,
    node_id: str,
    currency: str = "USD",
) -> StructuringResult:
    """
    Detect structuring patterns centered on a specific node.

    v8.0 [P4v8-02]: Transactions filtered by currency BEFORE threshold
    comparison, implementing Rule 14.
    v7.0 [P4-08]: Fan-in checked FIRST and takes precedence.

    Args:
        graph_reasoner: Loaded GraphReasoner instance.
        node_id: Node ID to analyze.
        currency: "USD" or "INR" for jurisdiction-specific thresholds.

    Returns:
        StructuringResult with detection outcome and evidence.
    """
    min_amount, max_amount, min_count, time_window = _get_thresholds(currency)

    # === CHECK 1: Fan-In (Many -> node_id) ===
    incoming = graph_reasoner.get_1hop_incoming(node_id)
    if incoming:
        currency_filtered_in = _filter_by_currency(incoming, currency)
        qualifying_in = _filter_qualifying_transactions(
            currency_filtered_in, min_amount, max_amount, time_window,
        )
        if len(qualifying_in) >= min_count:
            counterparts = sorted(set(tx["source_node"] for tx in qualifying_in))
            total = sum(tx["amount"] for tx in qualifying_in)
            confidence = _compute_confidence(
                len(qualifying_in), len(counterparts), min_count,
                time_window, qualifying_in,
            )

            return StructuringResult(
                detected=True,
                mode="FAN_IN",
                target_node=node_id,
                counterpart_nodes=counterparts,
                qualifying_transactions=qualifying_in,
                total_amount=total,
                transaction_count=len(qualifying_in),
                currency=currency,
                confidence=confidence,
                reasoning=(
                    f"STRUCTURING (Fan-In) detected at node '{node_id}': "
                    f"{len(qualifying_in)} transactions in [{min_amount}-{max_amount}] "
                    f"{currency} range within {time_window}s window. "
                    f"Total: {total} {currency} from {len(counterparts)} unique sources."
                ),
            )

    # === CHECK 2: Fan-Out (node_id -> Many) ===
    outgoing = graph_reasoner.get_1hop_outgoing(node_id)
    if outgoing:
        currency_filtered_out = _filter_by_currency(outgoing, currency)
        qualifying_out = _filter_qualifying_transactions(
            currency_filtered_out, min_amount, max_amount, time_window,
        )
        if len(qualifying_out) >= min_count:
            counterparts = sorted(set(tx["target_node"] for tx in qualifying_out))
            total = sum(tx["amount"] for tx in qualifying_out)
            confidence = _compute_confidence(
                len(qualifying_out), len(counterparts), min_count,
                time_window, qualifying_out,
            )

            return StructuringResult(
                detected=True,
                mode="FAN_OUT",
                target_node=node_id,
                counterpart_nodes=counterparts,
                qualifying_transactions=qualifying_out,
                total_amount=total,
                transaction_count=len(qualifying_out),
                currency=currency,
                confidence=confidence,
                reasoning=(
                    f"STRUCTURING (Fan-Out) detected at node '{node_id}': "
                    f"{len(qualifying_out)} outgoing transactions in "
                    f"[{min_amount}-{max_amount}] {currency} range. "
                    f"Total: {total} {currency} to {len(counterparts)} unique targets."
                ),
            )

    # === No structuring detected ===
    return StructuringResult(
        detected=False,
        mode="NONE",
        target_node=node_id,
        currency=currency,
        reasoning=f"No structuring pattern detected at node '{node_id}'.",
    )
