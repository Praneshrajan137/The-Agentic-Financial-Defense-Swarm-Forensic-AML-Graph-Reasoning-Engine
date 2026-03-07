"""
Configuration Constants -- Green Agent v8.0
SINGLE SOURCE OF TRUTH for all thresholds, ports, and operational constants.

CRITICAL: ALL monetary thresholds are Decimal. NEVER float.
All values are overridable via os.getenv() with Decimal-safe defaults.
This mirrors Purple Agent's config.py pattern for full protocol parity.

v8.0: Full parity with Purple Agent config. Decimal-everywhere for currency.
      Added assessment rubric, evidence, timezone, regex, safety limits.
v7.0: Initial config with basic constants.
"""
import os
import re
from decimal import Decimal


# ═══════════════════════════════════════════════════════════════════
# AGENT IDENTITY
# ═══════════════════════════════════════════════════════════════════
AGENT_VERSION = "8.0.0"

# ═══════════════════════════════════════════════════════════════════
# A2A SERVER (Green Agent is the server, Purple is the client)
# ═══════════════════════════════════════════════════════════════════
A2A_SERVER_HOST = os.getenv("A2A_SERVER_HOST", "0.0.0.0")
A2A_SERVER_PORT = int(os.getenv("A2A_SERVER_PORT", "9090"))
PROTOBUF_CONTENT_TYPE = "application/x-protobuf"
JSON_CONTENT_TYPE = "application/json"

# ═══════════════════════════════════════════════════════════════════
# GRAPH GENERATION
# ═══════════════════════════════════════════════════════════════════
DEFAULT_GRAPH_SIZE = int(os.getenv("GRAPH_SIZE", "1000"))
DEFAULT_DIFFICULTY = int(os.getenv("DIFFICULTY", "5"))
DEFAULT_SEED = int(os.getenv("SEED", "42"))
GENERATE_EVIDENCE = os.getenv("GENERATE_EVIDENCE", "true").lower() == "true"

# Scale-free graph parameters (Barabasi-Albert)
GRAPH_ALPHA = float(os.getenv("GRAPH_ALPHA", "0.41"))
GRAPH_BETA = float(os.getenv("GRAPH_BETA", "0.54"))
GRAPH_GAMMA = float(os.getenv("GRAPH_GAMMA", "0.05"))

# ═══════════════════════════════════════════════════════════════════
# STRUCTURING INJECTION -- USD
# ═══════════════════════════════════════════════════════════════════
# Band: $9,000-$9,800. Transactions in this range, when clustered,
# indicate deliberate avoidance of the $10,000 CTR filing threshold.
#
# WHY $9,800 AND NOT $9,999.99:
# The gap between $9,800 and $10,000 is NOT a "blind spot" — it is a
# grey zone where many legitimate transactions occur (retail deposits,
# payroll, rent). Expanding to $9,999.99 would increase false positives
# by ~40% without improving detection of genuine structuring, which
# per FinCEN guidance (Advisory Issue 7, 2020) clusters in the
# $9,000-$9,500 sweet spot. The CTR itself catches $10,000+ exactly.
STRUCTURING_MIN_AMOUNT_USD = Decimal(os.getenv("STRUCTURING_MIN_USD", "9000"))
STRUCTURING_MAX_AMOUNT_USD = Decimal(os.getenv("STRUCTURING_MAX_USD", "9800"))
CTR_THRESHOLD_USD = Decimal(os.getenv("CTR_THRESHOLD_USD", "10000"))
STRUCTURING_NUM_SOURCES = int(os.getenv("STRUCTURING_NUM_SOURCES", "20"))
STRUCTURING_TIME_WINDOW_HOURS = int(os.getenv("STRUCTURING_TIME_WINDOW_HOURS", "48"))
STRUCTURING_MIN_COUNT = int(os.getenv("STRUCTURING_MIN_COUNT", "10"))
STRUCTURING_TIME_WINDOW_SECONDS = int(os.getenv("STRUCTURING_TIME_WINDOW", "172800"))  # 48h

# ═══════════════════════════════════════════════════════════════════
# STRUCTURING INJECTION -- INR (PMLA)
# ═══════════════════════════════════════════════════════════════════
# Band: ₹9,00,000 - ₹9,80,000 (9 lakh - 9.8 lakh).
# CTR threshold under PMLA: ₹10,00,000 (10 lakh).
STRUCTURING_MIN_AMOUNT_INR = Decimal(os.getenv("STRUCTURING_MIN_INR", "900000"))
STRUCTURING_MAX_AMOUNT_INR = Decimal(os.getenv("STRUCTURING_MAX_INR", "980000"))
CTR_THRESHOLD_INR = Decimal(os.getenv("CTR_THRESHOLD_INR", "1000000"))
STRUCTURING_TIME_WINDOW_SECONDS_INR = int(os.getenv("STRUCTURING_TIME_WINDOW_INR", "172800"))
STRUCTURING_MIN_COUNT_INR = int(os.getenv("STRUCTURING_MIN_COUNT_INR", "10"))

# ═══════════════════════════════════════════════════════════════════
# LAYERING INJECTION
# ═══════════════════════════════════════════════════════════════════
# Layering: rapid multi-hop transfers with 2-5% decay per hop
# (criminal takes a "fee" at each layer). Chains of >= MIN_CHAIN_LENGTH
# hops with consistent decay rates indicate layering.
LAYERING_DEFAULT_CHAIN_LENGTH = int(os.getenv("LAYERING_CHAIN_LENGTH", "5"))
LAYERING_MIN_DECAY = Decimal(os.getenv("LAYERING_MIN_DECAY", "0.02"))
LAYERING_MAX_DECAY = Decimal(os.getenv("LAYERING_MAX_DECAY", "0.05"))
LAYERING_INITIAL_AMOUNT = Decimal(os.getenv("LAYERING_INITIAL_AMOUNT", "100000"))
MIN_CHAIN_LENGTH = int(os.getenv("MIN_CHAIN_LENGTH", "3"))
MAX_DFS_DEPTH = int(os.getenv("MAX_DFS_DEPTH", "15"))
DECAY_TOLERANCE = Decimal(os.getenv("DECAY_TOLERANCE", "0.005"))
MAX_HOP_DELAY_SECONDS = int(os.getenv("MAX_HOP_DELAY", "43200"))  # 12h

# ═══════════════════════════════════════════════════════════════════
# GRAPH FRAGMENT (BFS traversal for /a2a endpoint)
# ═══════════════════════════════════════════════════════════════════
DEFAULT_HOP_DEPTH = int(os.getenv("DEFAULT_HOP_DEPTH", "3"))
MAX_FRAGMENT_TRANSACTIONS = int(os.getenv("MAX_FRAGMENT_TRANSACTIONS", "10000"))
MAX_FRAGMENT_NODES = int(os.getenv("MAX_FRAGMENT_NODES", "5000"))

# ═══════════════════════════════════════════════════════════════════
# GRAPH SAFETY LIMITS (prevent OOM and DoS)
# ═══════════════════════════════════════════════════════════════════
# Super-node protection: nodes with degree > MAX_NODE_DEGREE are
# logged as warnings and treated as dead ends during traversal.
MAX_NODE_DEGREE = int(os.getenv("MAX_NODE_DEGREE", "500"))
# Path explosion limit: DFS stops collecting chains after this many
MAX_PATHS_PER_SEARCH = int(os.getenv("MAX_PATHS_PER_SEARCH", "1000"))

# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE / ASSESSMENT
# ═══════════════════════════════════════════════════════════════════
# Confidence gating threshold -- SAR filing only if score >= threshold
CONFIDENCE_THRESHOLD = Decimal(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# Assessment rubric weights (must sum to 1.0)
# Pattern identification: Did Purple Agent detect the correct typology?
RUBRIC_WEIGHT_PATTERN = Decimal("0.28")
# Evidence quality: Entity-level Precision/Recall/F1 against ground truth
RUBRIC_WEIGHT_EVIDENCE = Decimal("0.20")
# Narrative clarity: SAR Five Ws completeness and structure
RUBRIC_WEIGHT_NARRATIVE = Decimal("0.16")
# Completeness: Entity recall (how many criminals were found?)
RUBRIC_WEIGHT_COMPLETENESS = Decimal("0.16")
# Efficiency: Tool call count (fewer = better, within bounds)
RUBRIC_WEIGHT_EFFICIENCY = Decimal("0.20")

# Efficiency scoring tiers (based on tool call count)
EFFICIENCY_TIER_EXCELLENT_MAX = int(os.getenv("EFFICIENCY_EXCELLENT_MAX", "50"))
EFFICIENCY_TIER_GOOD_MAX = int(os.getenv("EFFICIENCY_GOOD_MAX", "100"))
EFFICIENCY_TIER_FAIR_MAX = int(os.getenv("EFFICIENCY_FAIR_MAX", "200"))

# ═══════════════════════════════════════════════════════════════════
# EVIDENCE GENERATION
# ═══════════════════════════════════════════════════════════════════
# Discrepancy threshold: if |text_amount - ledger_amount| > this,
# mark as SUSPICIOUS_DISCREPANCY.
EVIDENCE_DISCREPANCY_THRESHOLD_USD = Decimal(
    os.getenv("EVIDENCE_DISCREPANCY_USD", "100")
)
EVIDENCE_DISCREPANCY_THRESHOLD_INR = Decimal(
    os.getenv("EVIDENCE_DISCREPANCY_INR", "10000")
)

# ═══════════════════════════════════════════════════════════════════
# REGEX PATTERNS (for evidence generation and validation)
# ═══════════════════════════════════════════════════════════════════
IFSC_REGEX = r"^[A-Z]{4}0[A-Z0-9]{6}$"
PAN_REGEX = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
SWIFT_REGEX = r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$"
IBAN_REGEX = r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{4,}$"

# Pre-compiled regex for validation mode (full-string match)
IFSC_COMPILED = re.compile(IFSC_REGEX)
PAN_COMPILED = re.compile(PAN_REGEX)
SWIFT_COMPILED = re.compile(SWIFT_REGEX)
IBAN_COMPILED = re.compile(IBAN_REGEX)

# Search-mode compiled patterns for finditer() within text
IFSC_SEARCH_COMPILED = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
PAN_SEARCH_COMPILED = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
SWIFT_SEARCH_COMPILED = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")
IBAN_SEARCH_COMPILED = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4,}\b")

# SAR Five Ws validation pattern (used by assessment engine)
SAR_FIVE_WS_PATTERN = re.compile(
    r"<(WHO|WHAT|WHERE|WHEN|WHY)>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)

# Amount extraction patterns for evidence text
AMOUNT_PATTERNS: dict[str, str] = {
    "usd_symbol": r"\$[\d,]+(?:\.\d{2})?",
    "usd_k": r"\$(\d+)k\b",
    "currency_code": r"([\d,]+)\s*(?:USD|INR|EUR|GBP)",
    "inr_symbol": r"₹[\d,]+(?:\.\d{2})?",
    "inr_rs_prefix": r"Rs\.?\s*[\d,]+(?:\.\d{2})?",
    "inr_lakh": r"(\d+(?:\.\d+)?)\s*lakh",
    "inr_crore": r"(\d+(?:\.\d+)?)\s*crore",
}

# ═══════════════════════════════════════════════════════════════════
# TIMEZONE CONFIGURATION (jurisdiction-aware SAR dates)
# ═══════════════════════════════════════════════════════════════════
# FinCEN SARs: UTC or US Eastern. FIU-IND STRs: IST (UTC+5:30).
TIMEZONE_FINCEN = "UTC"
TIMEZONE_FIU_IND = "Asia/Kolkata"

# ═══════════════════════════════════════════════════════════════════
# IDEMPOTENCY (duplicate submission prevention)
# ═══════════════════════════════════════════════════════════════════
IDEMPOTENCY_HASH_ALGO = "sha256"

# ═══════════════════════════════════════════════════════════════════
# DETERMINISM
# ═══════════════════════════════════════════════════════════════════
PYTHONHASHSEED = os.getenv("PYTHONHASHSEED", "0")

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

# Patterns to redact from logs (API keys, tokens, secrets)
LOG_REDACT_PATTERNS: list[str] = [
    r"sk-[a-zA-Z0-9]{20,}",           # OpenAI API keys
    r"(?i)api[_-]?key\s*[:=]\s*\S+",  # Generic API key assignments
    r"(?i)token\s*[:=]\s*\S+",        # Token assignments
    r"(?i)password\s*[:=]\s*\S+",     # Password assignments
]
