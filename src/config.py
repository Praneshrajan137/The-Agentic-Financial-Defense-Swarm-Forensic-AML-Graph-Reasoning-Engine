"""
Configuration Constants -- Green Agent v7.0
SINGLE SOURCE OF TRUTH for all thresholds, ports, and operational constants.

All values are overridable via os.getenv() with safe defaults.
This mirrors Purple Agent's config.py pattern for consistency.
"""
import os
from decimal import Decimal


# ═══════════════════════════════════════════════════════════════════
# AGENT IDENTITY
# ═══════════════════════════════════════════════════════════════════
AGENT_VERSION = "7.0.0"

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
STRUCTURING_MIN_AMOUNT_USD = Decimal(os.getenv("STRUCTURING_MIN_USD", "9000"))
STRUCTURING_MAX_AMOUNT_USD = Decimal(os.getenv("STRUCTURING_MAX_USD", "9800"))
CTR_THRESHOLD_USD = Decimal(os.getenv("CTR_THRESHOLD_USD", "10000"))
STRUCTURING_NUM_SOURCES = int(os.getenv("STRUCTURING_NUM_SOURCES", "20"))
STRUCTURING_TIME_WINDOW_HOURS = int(os.getenv("STRUCTURING_TIME_WINDOW_HOURS", "48"))

# ═══════════════════════════════════════════════════════════════════
# STRUCTURING INJECTION -- INR (PMLA)
# ═══════════════════════════════════════════════════════════════════
STRUCTURING_MIN_AMOUNT_INR = Decimal(os.getenv("STRUCTURING_MIN_INR", "900000"))
STRUCTURING_MAX_AMOUNT_INR = Decimal(os.getenv("STRUCTURING_MAX_INR", "980000"))
CTR_THRESHOLD_INR = Decimal(os.getenv("CTR_THRESHOLD_INR", "1000000"))

# ═══════════════════════════════════════════════════════════════════
# LAYERING INJECTION
# ═══════════════════════════════════════════════════════════════════
LAYERING_DEFAULT_CHAIN_LENGTH = int(os.getenv("LAYERING_CHAIN_LENGTH", "5"))
LAYERING_MIN_DECAY = float(os.getenv("LAYERING_MIN_DECAY", "0.02"))
LAYERING_MAX_DECAY = float(os.getenv("LAYERING_MAX_DECAY", "0.05"))
LAYERING_INITIAL_AMOUNT = float(os.getenv("LAYERING_INITIAL_AMOUNT", "100000.0"))

# ═══════════════════════════════════════════════════════════════════
# GRAPH FRAGMENT (BFS traversal for /a2a endpoint)
# ═══════════════════════════════════════════════════════════════════
DEFAULT_HOP_DEPTH = int(os.getenv("DEFAULT_HOP_DEPTH", "3"))
MAX_FRAGMENT_TRANSACTIONS = int(os.getenv("MAX_FRAGMENT_TRANSACTIONS", "10000"))
MAX_FRAGMENT_NODES = int(os.getenv("MAX_FRAGMENT_NODES", "5000"))

# ═══════════════════════════════════════════════════════════════════
# DETERMINISM
# ═══════════════════════════════════════════════════════════════════
PYTHONHASHSEED = os.getenv("PYTHONHASHSEED", "0")

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
