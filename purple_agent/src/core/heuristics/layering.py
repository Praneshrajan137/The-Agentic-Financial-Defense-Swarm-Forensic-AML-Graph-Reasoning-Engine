"""
Layering Detection Heuristic
PRD Reference: Task B3
Package Version: v8.0

Detects money laundering layering patterns where funds flow through
a chain of intermediary accounts with systematic value decay (2-5%
per hop, representing "service fees" or deliberate obfuscation).

v8.0 [P4v8-05]: Currency consistency validation within chains.
  A chain mixing USD and INR transactions produces meaningless decay
  calculations. All transactions in a chain must share the same currency.
v8.0 [P4v8-08]: detect_layering accepts optional currency parameter.
  LayeringChain and LayeringResult record the detected currency for
  downstream SAR jurisdiction formatting.

v7.0 [P4-13]: chain_length renamed to hop_count for clarity.

CRITICAL: All amount comparisons use Decimal arithmetic.
"""
import logging
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from src.config import (
    DECAY_RATE_MIN,
    DECAY_RATE_MAX,
    DECAY_TOLERANCE,
    MIN_CHAIN_LENGTH,
    MAX_HOP_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)


@dataclass
class LayeringChain:
    """A single detected layering chain.

    v8.0 [P4v8-08]: currency records the chain's currency for SAR formatting.
    v7.0 [P4-13]: hop_count = number of transactions in the chain.
    """
    chain_nodes: list[str]
    chain_transactions: list[dict]
    decay_rates: list[Decimal]
    avg_decay_rate: Decimal = Decimal("0")
    hop_count: int = 0
    start_amount: Decimal = Decimal("0")
    end_amount: Decimal = Decimal("0")
    currency: str = "USD"

    @property
    def chain_length(self) -> int:
        """Backward-compatible alias for hop_count."""
        return self.hop_count


@dataclass
class LayeringResult:
    """Result of layering detection from a start node."""
    detected: bool
    start_node: str
    chains: list[LayeringChain] = field(default_factory=list)
    total_chains_found: int = 0
    confidence: Decimal = Decimal("0")
    currency: str = "USD"
    reasoning: str = ""


def _calculate_hop_decay(amount_in: Decimal, amount_out: Decimal) -> Decimal | None:
    """
    Calculate the decay rate for a single hop.

    Returns:
        Decay rate as Decimal (e.g., 0.03 for 3%), or None if invalid.
    """
    if amount_in <= Decimal("0"):
        return None
    try:
        decay = (amount_in - amount_out) / amount_in
        return decay
    except (InvalidOperation, ZeroDivisionError):
        return None


def _analyze_chain(chain: list[dict]) -> LayeringChain | None:
    """
    Analyze a single DFS chain for layering characteristics.

    v8.0 [P4v8-05]: Validates currency consistency across all transactions.
    Mixed-currency chains (e.g., USD->INR) are rejected because cross-currency
    decay calculations are meaningless.

    Args:
        chain: List of transaction dicts from dfs_trace_chains.

    Returns:
        LayeringChain if the chain meets all layering criteria, None otherwise.
    """
    if len(chain) < MIN_CHAIN_LENGTH:
        return None

    currencies_in_chain = set(tx.get("currency", "USD") for tx in chain)
    if len(currencies_in_chain) != 1:
        logger.info(
            f"Chain rejected: mixed currencies {currencies_in_chain} detected. "
            f"Cross-currency decay calculations are meaningless. "
            f"Chain nodes: {chain[0]['source_node']} -> ... -> {chain[-1]['target_node']}"
        )
        return None
    chain_currency = currencies_in_chain.pop()

    decay_rates: list[Decimal] = []
    effective_min = DECAY_RATE_MIN - DECAY_TOLERANCE
    effective_max = DECAY_RATE_MAX + DECAY_TOLERANCE

    for i in range(len(chain) - 1):
        current_tx = chain[i]
        next_tx = chain[i + 1]

        time_gap = abs(next_tx["timestamp"] - current_tx["timestamp"])
        if time_gap > MAX_HOP_DELAY_SECONDS:
            return None

        decay = _calculate_hop_decay(current_tx["amount"], next_tx["amount"])
        if decay is None:
            return None

        if not (effective_min <= decay <= effective_max):
            return None

        decay_rates.append(decay)

    if not decay_rates:
        return None

    chain_nodes = [chain[0]["source_node"]]
    for tx in chain:
        chain_nodes.append(tx["target_node"])

    avg_decay = sum(decay_rates) / Decimal(str(len(decay_rates)))

    return LayeringChain(
        chain_nodes=chain_nodes,
        chain_transactions=chain,
        decay_rates=decay_rates,
        avg_decay_rate=avg_decay,
        hop_count=len(chain),
        start_amount=chain[0]["amount"],
        end_amount=chain[-1]["amount"],
        currency=chain_currency,
    )


def detect_layering(
    graph_reasoner,
    node_id: str,
    max_depth: int | None = None,
    currency: str | None = None,
) -> LayeringResult:
    """
    Detect layering patterns originating from a specific node.

    v8.0 [P4v8-08]: Optional currency parameter. If provided, only chains
    matching that currency are included in results. If None, all chains
    (with internally consistent currencies) are accepted.

    Args:
        graph_reasoner: Loaded GraphReasoner instance.
        node_id: Node ID to trace chains from.
        max_depth: Optional DFS depth limit.
        currency: Optional currency filter for detected chains.

    Returns:
        LayeringResult with detection outcome and chain details.
    """
    raw_chains = graph_reasoner.dfs_trace_chains(node_id, max_depth=max_depth)

    detected_chains: list[LayeringChain] = []
    for raw_chain in raw_chains:
        analyzed = _analyze_chain(raw_chain)
        if analyzed is not None:
            if currency is not None and analyzed.currency != currency:
                continue
            detected_chains.append(analyzed)

    if detected_chains:
        longest = max(c.hop_count for c in detected_chains)
        chain_confidence = min(
            Decimal(str(longest)) / Decimal(str(MIN_CHAIN_LENGTH * 2)),
            Decimal("1"),
        )
        count_bonus = min(
            Decimal(str(len(detected_chains))) / Decimal("5"),
            Decimal("0.2"),
        )
        confidence = min(chain_confidence + count_bonus, Decimal("1"))

        result_currency = currency if currency else detected_chains[0].currency

        return LayeringResult(
            detected=True,
            start_node=node_id,
            chains=detected_chains,
            total_chains_found=len(detected_chains),
            confidence=confidence,
            currency=result_currency,
            reasoning=(
                f"LAYERING detected from node '{node_id}': "
                f"{len(detected_chains)} chain(s) with systematic "
                f"{DECAY_RATE_MIN*100}-{DECAY_RATE_MAX*100}% decay. "
                f"Longest chain: {longest} hop(s). "
                f"Currency: {result_currency}."
            ),
        )

    return LayeringResult(
        detected=False,
        start_node=node_id,
        currency=currency or "USD",
        reasoning=f"No layering pattern detected from node '{node_id}'.",
    )
