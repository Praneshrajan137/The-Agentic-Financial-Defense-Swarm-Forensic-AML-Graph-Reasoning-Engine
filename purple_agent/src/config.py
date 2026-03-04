"""
Configuration Constants -- Purple Agent v7.0
PRD Reference: All Tasks
CRITICAL: ALL monetary thresholds are Decimal. NEVER float.

This file is the SINGLE SOURCE OF TRUTH for all thresholds, timeouts,
regex patterns, and magic numbers. EVERY implementation file imports from here.

v7.0: All operational values overridable via os.getenv() with Decimal-safe
defaults. ALN-02 constants (SAR_LLM_SEED, SAR_MAX_NARRATIVE_CHARS,
STRUCTURING_MIN_COUNT_INR) are included natively — no addendum required.
"""
import os
import re
from decimal import Decimal


# ═══════════════════════════════════════════════════════════════════
# AGENT IDENTITY
# ═══════════════════════════════════════════════════════════════════
AGENT_VERSION = "7.0.0"

# ═══════════════════════════════════════════════════════════════════
# STRUCTURING (SMURFING) DETECTION -- USD
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
STRUCTURING_MIN_COUNT = int(os.getenv("STRUCTURING_MIN_COUNT", "10"))
STRUCTURING_TIME_WINDOW_SECONDS = int(os.getenv("STRUCTURING_TIME_WINDOW", "172800"))  # 48h
VELOCITY_THRESHOLD = Decimal(os.getenv("VELOCITY_THRESHOLD", "5000"))

# ═══════════════════════════════════════════════════════════════════
# STRUCTURING (SMURFING) DETECTION -- INR (Indian Rupees / PMLA)
# ═══════════════════════════════════════════════════════════════════
# Band: ₹9,00,000 - ₹9,80,000 (9 lakh - 9.8 lakh).
# CTR threshold under PMLA: ₹10,00,000 (10 lakh).
STRUCTURING_MIN_AMOUNT_INR = Decimal(os.getenv("STRUCTURING_MIN_INR", "900000"))
STRUCTURING_MAX_AMOUNT_INR = Decimal(os.getenv("STRUCTURING_MAX_INR", "980000"))
CTR_THRESHOLD_INR = Decimal(os.getenv("CTR_THRESHOLD_INR", "1000000"))
STRUCTURING_TIME_WINDOW_SECONDS_INR = int(os.getenv("STRUCTURING_TIME_WINDOW_INR", "172800"))
# v7.0 [ALN-02 FIX]: INR structuring count threshold. Same logic as USD:
# minimum number of transactions in the band within the time window.
STRUCTURING_MIN_COUNT_INR = int(os.getenv("STRUCTURING_MIN_COUNT_INR", "10"))

# ═══════════════════════════════════════════════════════════════════
# LAYERING DETECTION
# ═══════════════════════════════════════════════════════════════════
# Layering: rapid multi-hop transfers with 2-5% decay per hop
# (criminal takes a "fee" at each layer). Chains of >= MIN_CHAIN_LENGTH
# hops with consistent decay rates indicate layering.
MIN_CHAIN_LENGTH = int(os.getenv("MIN_CHAIN_LENGTH", "3"))
MAX_DFS_DEPTH = int(os.getenv("MAX_DFS_DEPTH", "15"))
DECAY_RATE_MIN = Decimal(os.getenv("DECAY_RATE_MIN", "0.02"))
DECAY_RATE_MAX = Decimal(os.getenv("DECAY_RATE_MAX", "0.05"))
DECAY_TOLERANCE = Decimal(os.getenv("DECAY_TOLERANCE", "0.005"))
MAX_HOP_DELAY_SECONDS = int(os.getenv("MAX_HOP_DELAY", "43200"))  # 12h

# ═══════════════════════════════════════════════════════════════════
# GRAPH SAFETY LIMITS (prevent OOM and DoS)
# ═══════════════════════════════════════════════════════════════════
# Super-node protection: nodes with degree > MAX_NODE_DEGREE are
# logged as warnings and treated as dead ends during traversal.
# This prevents "payroll node" explosions (e.g., Amazon Payroll with
# millions of edges causing OOM or infinite traversal).
MAX_NODE_DEGREE = int(os.getenv("MAX_NODE_DEGREE", "500"))

# Path explosion limit: DFS stops collecting chains after this many
# to prevent factorial blowup in clique-like subgraphs.
MAX_PATHS_PER_SEARCH = int(os.getenv("MAX_PATHS_PER_SEARCH", "1000"))

# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE SCORING
# ═══════════════════════════════════════════════════════════════════
# SAR filing is gated on this threshold. If confidence_score < threshold,
# typology is set to NONE and SAR is not filed (LOW_CONFIDENCE path).
# This prevents filing of weak/speculative SARs.
#
# Scoring formula:
#   base = 0.3 per detected typology (structuring OR layering)
#   base = 0.6 if BOTH detected
#   + 0.2 if text evidence corroborates ledger findings
#   + 0.2 if SUSPICIOUS_DISCREPANCY found (text vs ledger mismatch)
#   Result clamped to [0.0, 1.0]
CONFIDENCE_THRESHOLD = Decimal(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# ═══════════════════════════════════════════════════════════════════
# A2A CLIENT -- RETRY & CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════
# Circuit breaker state machine:
#   CLOSED (normal) → OPEN (tripped after failure_threshold exceeded)
#   OPEN → HALF_OPEN (after window_seconds, probe one request)
#   HALF_OPEN → CLOSED (probe succeeded) or OPEN (probe failed)
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "5"))
RETRY_BASE_DELAY_SECONDS = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
JITTER_FACTOR = float(os.getenv("JITTER_FACTOR", "0.5"))
CIRCUIT_BREAKER_FAILURE_THRESHOLD = float(os.getenv("CB_FAILURE_THRESHOLD", "0.6"))
CIRCUIT_BREAKER_WINDOW_SECONDS = int(os.getenv("CB_WINDOW_SECONDS", "60"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT", "30.0"))

# ═══════════════════════════════════════════════════════════════════
# A2A PROTOCOL
# ═══════════════════════════════════════════════════════════════════
GREEN_AGENT_URL = os.getenv("GREEN_AGENT_URL", "http://localhost:9090")
PROTOBUF_CONTENT_TYPE = "application/x-protobuf"
JSON_CONTENT_TYPE = "application/json"
A2A_SERVER_HOST = os.getenv("A2A_SERVER_HOST", "0.0.0.0")
A2A_SERVER_PORT = int(os.getenv("A2A_SERVER_PORT", "8080"))

# ═══════════════════════════════════════════════════════════════════
# SAR GENERATION
# ═══════════════════════════════════════════════════════════════════
SAR_MAX_RETRY = int(os.getenv("SAR_MAX_RETRY", "3"))
SAR_LLM_MODEL = os.getenv("SAR_LLM_MODEL", "gpt-4o-mini")
SAR_LLM_TEMPERATURE = float(os.getenv("SAR_LLM_TEMPERATURE", "0.0"))
# v7.0 [ALN-02 FIX]: LLM seed for deterministic narrative generation.
# Model-specific behavior: OpenAI gpt-4o-mini respects seed parameter.
SAR_LLM_SEED = int(os.getenv("SAR_LLM_SEED", "42"))
# v7.0 [ALN-02 FIX]: Maximum narrative length in characters.
# FinCEN e-filing has a practical limit; mechanical SAR fallback
# should also respect this to avoid truncation at submission.
SAR_MAX_NARRATIVE_CHARS = int(os.getenv("SAR_MAX_NARRATIVE_CHARS", "10000"))

# ═══════════════════════════════════════════════════════════════════
# SAR GENERATION — v8.0 ADDITIONS (required by C1: SAR Drafter)
# ═══════════════════════════════════════════════════════════════════
# Max transactions to include in LLM prompt. Caps prompt length to
# prevent context window overflow. 50 covers the structuring scenario
# (20 TXs) and layering (4 TXs) with headroom for mixed scenarios.
MAX_PROMPT_TRANSACTIONS = int(os.getenv("MAX_PROMPT_TRANSACTIONS", "50"))

# ═══════════════════════════════════════════════════════════════════
# EVIDENCE SYNTHESIS
# ═══════════════════════════════════════════════════════════════════
# Discrepancy threshold: if |text_amount - ledger_amount| > this,
# mark as SUSPICIOUS_DISCREPANCY. Moved from hardcoded magic number.
EVIDENCE_DISCREPANCY_THRESHOLD_USD = Decimal(
    os.getenv("EVIDENCE_DISCREPANCY_USD", "100")
)
EVIDENCE_DISCREPANCY_THRESHOLD_INR = Decimal(
    os.getenv("EVIDENCE_DISCREPANCY_INR", "10000")
)

# spaCy model: configurable for environments with different
# Docker size constraints. sm=66MB, md=91MB, trf=400MB+.
# Default sm + regex fallback provides sufficient accuracy.
SPACY_MODEL_NAME = os.getenv("SPACY_MODEL", "en_core_web_sm")

# ═══════════════════════════════════════════════════════════════════
# REGEX PATTERNS
# ═══════════════════════════════════════════════════════════════════
IFSC_REGEX = r"^[A-Z]{4}0[A-Z0-9]{6}$"
PAN_REGEX = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
SWIFT_REGEX = r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$"
IBAN_REGEX = r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{4,}$"

# Amount extraction patterns for Evidence Synthesizer
# v7.0: Added ₹ symbol and "Rs." prefix patterns for Indian text evidence
AMOUNT_PATTERNS: dict[str, str] = {
    "usd_symbol": r"\$[\d,]+(?:\.\d{2})?",
    "usd_k": r"\$(\d+)k\b",
    "currency_code": r"([\d,]+)\s*(?:USD|INR|EUR|GBP)",
    "inr_symbol": r"₹[\d,]+(?:\.\d{2})?",
    "inr_rs_prefix": r"Rs\.?\s*[\d,]+(?:\.\d{2})?",
    "inr_lakh": r"(\d+(?:\.\d+)?)\s*lakh",
    "inr_crore": r"(\d+(?:\.\d+)?)\s*crore",
}

# Pre-compiled regex for hot-path performance (full-string VALIDATION mode)
IFSC_COMPILED = re.compile(IFSC_REGEX)
PAN_COMPILED = re.compile(PAN_REGEX)
SWIFT_COMPILED = re.compile(SWIFT_REGEX)
IBAN_COMPILED = re.compile(IBAN_REGEX)

# v8.0 [P4v8-01]: SEARCH-mode compiled patterns for finditer() within text.
# The _COMPILED patterns above use ^$ anchors (validation mode) which require
# the ENTIRE input string to match — they return zero results when used with
# .finditer() on prose text. These _SEARCH_COMPILED variants use \b word
# boundaries to locate identifiers embedded in longer strings.
IFSC_SEARCH_COMPILED = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
PAN_SEARCH_COMPILED = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
SWIFT_SEARCH_COMPILED = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")
IBAN_SEARCH_COMPILED = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4,}\b")

# ═══════════════════════════════════════════════════════════════════
# RALPH WIGGUM LOOP
# ═══════════════════════════════════════════════════════════════════
COMPLETION_SIGNAL = "<promise>complete</promise>"
MAX_RALPH_ITERATIONS = int(os.getenv("MAX_RALPH_ITERATIONS", "25"))
PRD_FILE_PATH = os.getenv("PRD_FILE_PATH", "plans/prd.json")
PROGRESS_FILE_PATH = os.getenv("PROGRESS_FILE_PATH", "progress.txt")
RALPH_ITERATION_TIMEOUT_SECONDS = int(os.getenv("RALPH_TIMEOUT", "120"))

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_JSON_FORMAT = os.getenv("LOG_JSON_FORMAT", "true").lower() == "true"

# Patterns to redact from logs (API keys, tokens, secrets)
LOG_REDACT_PATTERNS: list[str] = [
    r"sk-[a-zA-Z0-9]{20,}",           # OpenAI API keys
    r"(?i)api[_-]?key\s*[:=]\s*\S+",  # Generic API key assignments
    r"(?i)token\s*[:=]\s*\S+",        # Token assignments
    r"(?i)password\s*[:=]\s*\S+",     # Password assignments
]

# ═══════════════════════════════════════════════════════════════════
# TIMEZONE CONFIGURATION (jurisdiction-aware SAR dates)
# ═══════════════════════════════════════════════════════════════════
# FinCEN SARs: UTC or US Eastern. FIU-IND STRs: IST (UTC+5:30).
# Unix timestamps are timezone-agnostic; we must convert explicitly
# based on jurisdiction to avoid reporting wrong dates.
TIMEZONE_FINCEN = "UTC"
TIMEZONE_FIU_IND = "Asia/Kolkata"

# ═══════════════════════════════════════════════════════════════════
# IDEMPOTENCY (duplicate submission prevention)
# ═══════════════════════════════════════════════════════════════════
# Idempotency key = SHA-256( case_id + typology + sorted(involved_entities) )
# This ensures duplicate A2A submissions are detected and rejected.
IDEMPOTENCY_HASH_ALGO = "sha256"
