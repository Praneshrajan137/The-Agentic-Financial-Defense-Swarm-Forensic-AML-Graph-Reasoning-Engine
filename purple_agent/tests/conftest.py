"""
Shared Test Infrastructure -- Purple Agent v12.0
Contains ALL 16 fixtures used across test files.
Every amount is Decimal. Every timestamp is deterministic.
Every list output uses sorted() for determinism (PYTHONHASHSEED-safe).

Fixtures:
  1.  structuring_scenario           -- 20 criminal sources -> 1 mule (USD)
  2.  layering_scenario              -- 4-hop chain with ~3% decay (5 nodes, 4 edges)
  3.  mixed_scenario                 -- Both typologies + 50 noise nodes (cycle stress)
  4.  sample_text_evidence           -- Text with planted discrepancies
  5.  mock_llm_response              -- Deterministic SAR narrative (Five Ws)
  6.  indian_scenario                -- INR structuring for PMLA/FIU-IND compliance
  7.  multi_edge_scenario            -- 3 transactions between same A->B pair
  8.  structuring_boundary_scenario  -- Exact threshold boundary testing
  9.  empty_graph_scenario           -- Zero nodes, zero transactions
  10. legitimate_only_scenario       -- Clean graph, zero crime (false-positive test)
  11. layering_boundary_scenario     -- Decay rate boundary testing (exact 2%, 5%)
  12. duplicate_txid_scenario        -- Duplicate tx IDs (keep first, log warning)
  13. self_loop_scenario             -- Self-loops (source==target, log not reject)
  14. confidence_scoring_scenario    -- Confidence gate testing (above/below/boundary)
  15. super_node_scenario            -- High-degree hub node (> MAX_NODE_DEGREE)
  16. mixed_currency_scenario        -- USD + INR on same mule (group by currency)

MATHEMATICAL VERIFICATION (all independently computed with Python Decimal):
  Structuring scenario total: sum(9000 + 40*i for i in 1..20) = 20*9000 + 40*(210) = $188,400
  Structuring scenario timestamps: first=1700001800 (i=1), last=1700036000 (i=20), span=9.5h
  Layering decay chain: 100000 * 0.97^0 = 100000, *0.97 = 97000, *0.97 = 94090, *0.97 = 91267.30
  Mixed structuring: sum(9100 + 40*i for i in 1..15) = 15*9100 + 40*(120) = $141,300
  Mixed layering decay: 50000, 48500 (3% of 50000=1500), 47045 (3% of 48500=1455)
  Indian scenario: sum(900000 + 5000*i for i in 1..15) = 15*900000 + 5000*(120) = ₹14,100,000
  Mixed currency USD: sum(9060 + 40*i for i in 1..12) = 12*9060 + 40*(78) = $111,840
  Mixed currency INR: sum(900000 + 5000*i for i in 1..12) = 10800000 + 390000 = ₹11,190,000

IEEE 754 SAFETY PROOF (confidence scoring values):
  0.3 + 0.2 = 0.5 exactly in IEEE 754 (errors cancel) ✓
  0.6 + 0.2 = 0.8 exactly ✓
  0.3 + 0.2 + 0.2 = 0.7 exactly (0.5 + 0.2, both representable) ✓
  All confidence_scoring_scenario expected_score values survive float comparison.
  BITWISE VERIFIED: struct.pack('d', computed) == struct.pack('d', expected) for all 5 cases.

CHAIN_LENGTH SEMANTICS:
  "chain_length" = number of HOPS (directed edges/transactions), NOT number of nodes.
  A chain: A -> B -> C -> D has chain_length=3 (3 hops) and 4 nodes.
  MIN_CHAIN_LENGTH=3 means at least 3 hops are required to qualify as layering.

INGESTION RULE (v10.0 Rule 3, inherited from v7.0):
  Do NOT quantize Decimal at ingestion. Factory methods produce exact Decimal values.
  Quantization happens ONLY at threshold comparison or SAR formatting.

v12.0 RETAINED ALL v11.0 + EARLIER FIXES:
  - make_transaction rejects zero-amount (strict positive) [MED-03]
  - Fixture 14 uses boolean evidence fields [CRIT-05]
  - Fixture 14 includes exact_boundary test [ENH-01]
  - Fixture 3 includes expected aggregates [MED-10]
  - Fixture 4 documents SWIFT discrepancy [MED-04]
  - Fixture 16 math verified in module header [MED-14]
  - Fixture 11 "below1pct" documented as 1.0% [MED-18]
  - Fixture 3 Phase B limitation on expected_confidence_score_minimum [LOW-09]
  - Fixture 5 mock_llm_response corrected [CRIT-10]
  - Fixture 6 expected_total_criminal_inflow added [MED-23]
  - Fixture 3 layering chain docstring disambiguated [MED-24]
  - Fixture 14 confidence_threshold is float [LOW-10]

v12.0 ADDITIONS:
  - make_transaction: source/target non-empty validation [LOW-12]
  - make_node: node_id non-empty validation [LOW-13]
  - Fixture 2: expected_amounts added [MED-30]
  - Fixture 3: noise ring cycle documented, ring metadata added [MED-29]
  - Fixture 6: legitimate_nodes=[] added for structural parity [MED-28]
  - Fixture 8: should_detect → in_range_sources, expected_scenario_detected added [MED-27]
  - Fixture 10: expected_node_count, expected_tx_count added [LOW-14]
  - Fixture 12: expected_kept_amounts, expected_kept_routes added [HIGH-20]
  - Fixture 14: state_builder per test case for zero-mapping test wiring [HIGH-19]
  - Fixture 15: structuring confound documented, expected_structuring_confound added [MED-26]
"""
import pytest
from decimal import Decimal
from typing import Any


# ===================================================================
# HELPER FACTORIES
# ===================================================================

def make_transaction(
    tx_id: str,
    source: str,
    target: str,
    amount: Decimal,
    currency: str = "USD",
    timestamp: int = 1700000000,
    tx_type: str = "WIRE",
    reference: str = "",
    branch_code: str = "",
) -> dict[str, Any]:
    """Create a transaction dict matching Protobuf schema.

    Args:
        tx_id: Unique transaction identifier (e.g., "TX-STRUCT-001").
        source: Source entity node ID. Must be non-empty.
        target: Target entity node ID. Must be non-empty.
        amount: Transaction amount as Decimal (NEVER float). Must be strictly positive.
        currency: ISO 4217 currency code (USD, INR, EUR, GBP). Must be non-empty.
        timestamp: Unix epoch seconds (must be positive).
        tx_type: Transaction type (WIRE, ACH, CASH_DEPOSIT, RTGS, NEFT, CHECK).
        reference: Optional reference string.
        branch_code: Optional branch identifier.

    Returns:
        Dict with all transaction fields matching proto schema.

    Raises:
        ValueError: If amount is NaN, Infinity, zero, or negative.
        ValueError: If timestamp is non-positive.
        ValueError: If currency is empty.
        ValueError: If source or target is empty.
        TypeError: If amount is not Decimal.

    v9.0 NOTE: Strict positive validation (amount <= 0 rejected) is ALIGNED
    with _validate_amount in a2a_client.py [CRIT-07 fix]. Both boundaries
    must match to prevent conftest passing but production diverging.
    v12.0 [LOW-12]: source/target non-empty validation added. Empty-string
    node IDs create structurally invalid NetworkX nodes (falsy key, collides
    with "no node" semantics in Python boolean checks).
    """

    if not isinstance(amount, Decimal):
        raise TypeError(f"Transaction {tx_id}: amount must be Decimal, got {type(amount)}")
    if not amount.is_finite():
        raise ValueError(f"Transaction {tx_id}: amount must be finite, got {amount}")
    # v8.0 [MED-03] + v9.0 [CRIT-07 alignment]: Strict positive.
    # Zero-amount transactions are anomalous in financial crime detection.
    if amount <= Decimal("0"):
        raise ValueError(f"Transaction {tx_id}: amount must be strictly positive, got {amount}")
    # v7.0 [MED-01]: Timestamp must be positive (Unix epoch > 0)
    if timestamp <= 0:
        raise ValueError(f"Transaction {tx_id}: timestamp must be positive, got {timestamp}")
    # v7.0 [MED-02]: Currency must be non-empty (Step 2.5 validation)
    if not currency or not currency.strip():
        raise ValueError(f"Transaction {tx_id}: currency must be non-empty")
    # v12.0 [LOW-12]: Source and target must be non-empty.
    # Empty-string node IDs create attribute-less sentinel nodes in NetworkX
    # that break boolean checks (e.g., `if node_id:` evaluates False for "").
    # Also prevents silent creation of a single "" node shared by all
    # transactions with missing source or target fields.
    if not source or not source.strip():
        raise ValueError(f"Transaction {tx_id}: source must be non-empty")
    if not target or not target.strip():
        raise ValueError(f"Transaction {tx_id}: target must be non-empty")

    return {
        "id": tx_id,
        "source_node": source,
        "target_node": target,
        "amount": amount,
        "currency": currency,
        "timestamp": timestamp,
        "type": tx_type,
        "reference": reference,
        "branch_code": branch_code,
    }


def make_node(
    node_id: str,
    name: str,
    entity_type: str = "individual",
    jurisdiction: str = "US",
    account_id: str = "",
    ifsc_code: str = "",
    pan_number: str = "",
    address: str = "",
    risk_rating: str = "low",
    swift_code: str = "",
) -> dict[str, Any]:
    """Create a node attributes dict matching Protobuf schema.

    Args:
        node_id: Unique node identifier. Must be non-empty.
        name: Human-readable entity name.
        entity_type: One of "individual", "business", "bank", "government".
        jurisdiction: ISO 3166 country code (US, IN, GB, DE, etc.).
        account_id: Bank account identifier.
        ifsc_code: Indian Financial System Code (format: 4 alpha + 0 + 6 alphanum).
        pan_number: Permanent Account Number (format: 5 alpha + 4 digit + 1 alpha).
        address: Physical address string.
        risk_rating: One of "low", "medium", "high".
        swift_code: SWIFT/BIC code for international transfers.

    Returns:
        Dict with all node attribute fields matching proto schema.

    Raises:
        ValueError: If node_id is empty.

    v12.0 [LOW-13]: node_id non-empty validation. NetworkX uses node_id as
    the dict key — empty string is a legal Python key but semantically invalid
    and creates collision hazards when multiple nodes have id="".
    """

    # v12.0 [LOW-13]: Node ID must be non-empty.
    if not node_id or not node_id.strip():
        raise ValueError(f"make_node: node_id must be non-empty")

    return {
        "id": node_id,
        "name": name,
        "entity_type": entity_type,
        "jurisdiction": jurisdiction,
        "account_id": account_id or f"ACCT-{node_id}",
        "ifsc_code": ifsc_code,
        "pan_number": pan_number,
        "address": address,
        "risk_rating": risk_rating,
        "swift_code": swift_code,
    }


# ===================================================================
# FIXTURE 1: Structuring Scenario
# 20 criminal sources -> 1 mule (Shell Corp LLC)
# Amounts: $9,040 to $9,800 in $40 increments
# All below $10,000 CTR threshold (Bank Secrecy Act)
# MATH: sum(9000 + 40*i for i=1..20) = 180000 + 8400 = $188,400
# TIMESTAMPS: first=1700001800 (i=1), last=1700036000 (i=20), span=9.5h
# ===================================================================

@pytest.fixture
def structuring_scenario() -> dict[str, Any]:
    """
    Structuring (smurfing) in USD: 20 criminal sources sending to one mule.
    Amounts: $9,040 to $9,800 (all below $10,000 CTR)
    Time: 30-minute intervals (1800s) over ~9.5 hours.
    Timestamps: TX-STRUCT-001 at 1700001800 (i=1), TX-STRUCT-020 at 1700036000 (i=20).
    Includes 5 legitimate employers sending $3,000 salaries.
    """
    base_time = 1700000000
    transactions: list[dict] = []
    nodes: dict[str, dict] = {}
    criminal_sources: list[str] = []

    nodes["mule_1"] = make_node(
        "mule_1", "Shell Corp LLC", "business",
        swift_code="SCLLUS33",
    )

    for i in range(1, 21):
        src_id = f"src_{i:02d}"
        criminal_sources.append(src_id)
        nodes[src_id] = make_node(src_id, f"Criminal Source {i}")
        amount = Decimal("9000") + Decimal(str(i * 40))
        tx_time = base_time + (i * 1800)
        transactions.append(make_transaction(
            tx_id=f"TX-STRUCT-{i:03d}",
            source=src_id,
            target="mule_1",
            amount=amount,
            timestamp=tx_time,
        ))

    for i in range(1, 6):
        emp_id = f"employer_{i}"
        nodes[emp_id] = make_node(emp_id, f"LegitCorp {i}", "business")
        transactions.append(make_transaction(
            tx_id=f"TX-LEGIT-{i:03d}",
            source=emp_id,
            target="mule_1",
            amount=Decimal("3000"),
            timestamp=base_time + (i * 86400),
        ))

    return {
        "transactions": transactions,
        "nodes": nodes,
        "criminal_nodes": sorted(criminal_sources + ["mule_1"]),
        "mule_node": "mule_1",
        "criminal_sources": sorted(criminal_sources),
        "legitimate_nodes": sorted([f"employer_{i}" for i in range(1, 6)]),
        "jurisdiction": "fincen",
        "expected_total_criminal_inflow": Decimal("188400"),
    }


# ===================================================================
# FIXTURE 2: Layering Scenario
# 4-hop chain (5 nodes) with ~3% decay per hop
# CHAIN_LENGTH = 4 hops (>= MIN_CHAIN_LENGTH=3) ✓
# MATH: 100000 -> 97000 (3.0%) -> 94090 (3.0%) -> 91267.30 (3.0%)
# TIME: 2-hour gaps between hops (< MAX_HOP_DELAY_SECONDS=43200) ✓
# DECAY: 3.0% per hop (within [DECAY_RATE_MIN=2%, DECAY_RATE_MAX=5%]) ✓
# v12.0 [MED-30]: Added expected_amounts for test assertion convenience
# ===================================================================

@pytest.fixture
def layering_scenario() -> dict[str, Any]:
    """
    Chain: layer_origin -> layer_1 -> layer_2 -> layer_3 -> layer_4
    Hops: 4 (chain_length=4, satisfies MIN_CHAIN_LENGTH=3)
    Decay rate: exactly 3% per hop.
    Amounts: 100000 -> 97000 -> 94090 -> 91267.30
    Decay verification:
      Hop 1: (100000 - 97000) / 100000 = 0.03 (3.0%) ✓
      Hop 2: (97000 - 94090) / 97000 = 0.03 (3.0%) ✓
      Hop 3: (94090 - 91267.30) / 94090 = 0.03 (3.0%) ✓
    Time: 2-hour gaps (7200s < MAX_HOP_DELAY_SECONDS=43200s) ✓
    All chain nodes are criminal.
    """
    base_time = 1700000000
    chain_nodes = ["layer_origin", "layer_1", "layer_2", "layer_3", "layer_4"]
    nodes = {nid: make_node(nid, f"Layer Entity {i}") for i, nid in enumerate(chain_nodes)}

    amounts = [
        Decimal("100000"),
        Decimal("97000"),
        Decimal("94090"),
        Decimal("91267.30"),
    ]

    transactions = []
    for i in range(4):
        transactions.append(make_transaction(
            tx_id=f"TX-LAYER-{i+1:03d}",
            source=chain_nodes[i],
            target=chain_nodes[i + 1],
            amount=amounts[i],
            timestamp=base_time + (i * 7200),
        ))

    return {
        "transactions": transactions,
        "nodes": nodes,
        "chain_nodes": chain_nodes,
        "criminal_nodes": sorted(chain_nodes),
        "origin": "layer_origin",
        "terminal": "layer_4",
        "hop_count": 4,
        "expected_decay_rate": Decimal("0.03"),
        # v12.0 [MED-30]: Added for parity with mixed_scenario.expected_layering_amounts.
        # Tests can assert layering amounts directly without extracting from transactions.
        "expected_amounts": [
            Decimal("100000"), Decimal("97000"), Decimal("94090"), Decimal("91267.30")
        ],
        "jurisdiction": "fincen",
    }


# ===================================================================
# FIXTURE 3: Mixed Scenario (Both typologies + Noise)
# Tests full pipeline for false positive suppression
# Structuring: 15 smurfs -> mule_1 ($9,140-$9,700)
# Layering: 3-hop chain with 4 nodes (chain_0 -> chain_1 -> chain_2 -> chain_3)
#   Amounts: chain_0 $50K -> chain_1 $48.5K -> chain_2 $47,045 -> chain_3
# Noise: 50 legitimate entities in CIRCULAR RING (cycle stress test)
# v8.0 [MED-10]: Added expected aggregate values for test assertions
# v10.0 [LOW-09]: expected_confidence_score_minimum documented as Phase B+ only
# v11.0 [MED-24]: Layering chain docstring notation disambiguated
# v12.0 [MED-29]: Noise ring documented as cycle-breaking stress test
# ===================================================================

@pytest.fixture
def mixed_scenario() -> dict[str, Any]:
    """
    Combines structuring + layering + 50 legitimate noise nodes.
    Tests the full pipeline for false positive suppression.

    Structuring: 15 smurfs -> mule_1 ($9,140-$9,700)

    Layering: 3-hop chain with 4 nodes:
      chain_0 --$50K--> chain_1 --$48.5K--> chain_2 --$47,045--> chain_3
      3 transactions = 3 hops (satisfies MIN_CHAIN_LENGTH=3)
      Decay: 3% per hop (within [DECAY_RATE_MIN=2%, DECAY_RATE_MAX=5%])

    Noise: 50 legitimate entities sending $2,500 each (circular ring)
      v12.0 [MED-29] CYCLE STRESS TEST:
      The noise layer creates a CIRCULAR topology:
        legit_000 → legit_001 → ... → legit_049 → legit_000
      This is a cycle with 50 hops (>> MIN_CHAIN_LENGTH=3) and 0% decay
      (all amounts = $2,500). While 0% decay correctly falls outside
      [DECAY_RATE_MIN, DECAY_RATE_MAX] and should NOT trigger layering,
      the cycle tests a CRITICAL invariant: the layering detector MUST
      implement cycle-breaking (visited-node tracking) to avoid infinite
      traversal. Without it, this ring will cause a stack overflow or
      infinite loop during DFS/BFS path enumeration.

    Expected confidence_score:
      structuring detected (0.3) + layering detected (0.3) = 0.6 (BOTH)
      0.6 >= CONFIDENCE_THRESHOLD (0.5) -> SAR should be generated ✓

    v10.0 [LOW-09] PHASE LIMITATION:
      expected_confidence_score_minimum is ONLY valid after Phase B wiring.
      In Phase A, detect_structuring and detect_layering return empty lists,
      so confidence_score = 0.0. Do NOT use this field in Phase A tests.
    """
    base_time = 1700000000
    transactions: list[dict] = []
    nodes: dict[str, dict] = {}
    structuring_criminal: list[str] = []
    layering_criminal: list[str] = []

    nodes["mule_1"] = make_node("mule_1", "Shell Corp LLC", "business")
    for i in range(1, 16):
        src_id = f"smurf_{i:02d}"
        structuring_criminal.append(src_id)
        nodes[src_id] = make_node(src_id, f"Smurf {i}")
        amount = Decimal("9100") + Decimal(str(i * 40))
        transactions.append(make_transaction(
            f"TX-MIX-S-{i:03d}", src_id, "mule_1", amount,
            timestamp=base_time + (i * 1800),
        ))
    structuring_criminal.append("mule_1")

    chain = ["chain_0", "chain_1", "chain_2", "chain_3"]
    for nid in chain:
        nodes[nid] = make_node(nid, f"Chain {nid}")
    layer_amounts = [Decimal("50000"), Decimal("48500"), Decimal("47045")]
    for i in range(3):
        transactions.append(make_transaction(
            f"TX-MIX-L-{i+1:03d}", chain[i], chain[i+1], layer_amounts[i],
            timestamp=base_time + 100000 + (i * 7200),
        ))
    layering_criminal = chain

    for i in range(50):
        nid = f"legit_{i:03d}"
        nodes[nid] = make_node(nid, f"Legitimate Entity {i}")
        partner = f"legit_{(i + 1) % 50:03d}"
        transactions.append(make_transaction(
            f"TX-NOISE-{i:03d}", nid, partner, Decimal("2500"),
            timestamp=base_time + 200000 + (i * 3600),
        ))

    all_criminal = sorted(set(structuring_criminal + layering_criminal))

    return {
        "transactions": transactions,
        "nodes": nodes,
        "structuring_criminal": sorted(structuring_criminal),
        "layering_criminal": sorted(layering_criminal),
        "all_criminal": all_criminal,
        "mule_node": "mule_1",
        "jurisdiction": "fincen",
        "expected_structuring_total": Decimal("141300"),
        "expected_layering_amounts": [Decimal("50000"), Decimal("48500"), Decimal("47045")],
        # v10.0 [LOW-09]: Only valid after Phase B wiring (Phase A returns 0.0)
        "expected_confidence_score_minimum": 0.6,
        # v12.0 [MED-29]: Noise ring metadata for cycle-breaking assertions.
        # Tests can verify the layering detector does not traverse the cycle
        # endlessly and does not produce false positive detections from it.
        "noise_ring_is_cycle": True,
        "noise_ring_hop_count": 50,
        "noise_ring_amount": Decimal("2500"),
    }


# ===================================================================
# FIXTURE 4: Sample Text Evidence
# AMOUNT DISCREPANCY: email claims $10,000 but ledger shows $9,040
# SWIFT DISCREPANCY: EV-003 memo references "SWIFT: SBININBB" for
# mule_1, but structuring_scenario gives mule_1 swift_code="SCLLUS33"
# v8.0 [MED-04]: Both discrepancies documented in docstring.
# ===================================================================

@pytest.fixture
def sample_text_evidence() -> list[dict[str, Any]]:
    """
    Text evidence with planted discrepancies for evidence synthesis testing.

    CRITICAL AMOUNT DISCREPANCY (for test_evidence_synthesis_detects_discrepancy):
    EV-001 claims "$10,000" for a transfer from src_01, but the actual
    ledger transaction TX-STRUCT-001 shows $9,040. The evidence synthesizer
    must compare the text-extracted amount ($10,000) against the INDIVIDUAL
    transaction amount ($9,040), NOT against the aggregate total ($188,400).
    diff = |$10,000 - $9,040| = $960 > EVIDENCE_DISCREPANCY_THRESHOLD_USD ($100)
    => has_suspicious_discrepancy = True

    SECONDARY SWIFT DISCREPANCY (for multi-field cross-validation):
    EV-003 references SWIFT code "SBININBB" for mule_1, but the
    structuring_scenario ledger has mule_1 swift_code="SCLLUS33".
    This tests that the evidence synthesizer checks non-amount fields too.

    Contains Indian financial identifiers (PAN, IFSC) for dual-jurisdiction testing.
    Contains ₹ symbol and "lakh" notation for INR pattern testing.
    """

    return [
        {
            "id": "EV-001",
            "source_type": "email",
            "content": (
                "Hi, confirming the transfer of $10,000 from John Smith "
                "(Account ACCT-src_01) to Shell Corp LLC on 2023-11-14. "
                "Please process ASAP. Regards, User 8492"
            ),
            "associated_entity": "src_01",
            "timestamp": 1700000100,
        },
        {
            "id": "EV-002",
            "source_type": "teller_note",
            "content": (
                "Customer Jane Doe (PAN: ABCPP1234F, IFSC: SBIN0001234) "
                "visited branch requesting wire transfer to account ACCT-mule_1. "
                "Appeared nervous. Amount: 9.5 lakh INR. "
                "Also referenced ₹950,000 in conversation."
            ),
            "associated_entity": "mule_1",
            "timestamp": 1700003600,
        },
        {
            "id": "EV-003",
            "source_type": "memo",
            "content": (
                "Internal memo: Flagged account mule_1 (Shell Corp LLC, "
                "SWIFT: SBININBB) for unusual inbound volume. "
                "15 deposits in 48 hours, all below CTR threshold."
            ),
            "associated_entity": "mule_1",
            "timestamp": 1700010000,
        },
    ]


# ===================================================================
# FIXTURE 5: Mock LLM Response
# Deterministic SAR narrative for testing. All Five Ws present.
# v11.0 [CRIT-10]: THREE factual errors corrected from v10.0:
#   a) TX-STRUCT-020 timestamp: 1700037800 → 1700036000 (verified: base+20*1800)
#   b) "48-hour window" → removed (actual span: 9.5 hours)
#   c) "approximately 10 hours" → "approximately 9.5 hours" (verified)
# TIMESTAMP PROOF:
#   TX-STRUCT-001: base_time + 1*1800 = 1700001800
#   TX-STRUCT-020: base_time + 20*1800 = 1700000000 + 36000 = 1700036000
#   Span: (1700036000 - 1700001800) / 3600 = 34200 / 3600 = 9.5 hours
# ===================================================================

@pytest.fixture
def mock_llm_response() -> str:
    """Deterministic SAR narrative for testing. All Five Ws present.
    MATH VERIFICATION: Total $188,400 = sum of 20 transactions
    sum(9000 + 40*i for i in range(1, 21)) = 20*9000 + 40*sum(1..20)
    = 180,000 + 40*210 = 180,000 + 8,400 = 188,400 ✓
    TIMESTAMP VERIFICATION (v11.0 [CRIT-10]):
    TX-STRUCT-001: 1700000000 + 1*1800 = 1700001800 ✓
    TX-STRUCT-020: 1700000000 + 20*1800 = 1700036000 ✓
    Span: (1700036000 - 1700001800) = 34200s = 9.5 hours ✓
    """

    return (
        "WHO: Subject Shell Corp LLC (Account ACCT-mule_1), a business entity "
        "receiving funds from 20 individual senders including src_01 through src_20.\n\n"
        "WHAT: Structuring (smurfing) detected. 20 deposits ranging from $9,040 to "
        "$9,800 each, all below the $10,000 CTR threshold, were received within "
        "approximately 9.5 hours. Total inflow: $188,400 "
        "[TX-STRUCT-001 through TX-STRUCT-020].\n\n"
        "WHERE: Jurisdiction: United States (FinCEN). All transactions processed "
        "domestically.\n\n"
        "WHEN: Activity occurred between timestamps 1700001800 and 1700036000, "
        "spanning approximately 9.5 hours. Individual transactions: "
        "TX-STRUCT-001 ($9,040, t=1700001800), "
        "TX-STRUCT-020 ($9,800, t=1700036000).\n\n"
        "WHY: The pattern of multiple sub-threshold deposits from distinct senders "
        "to a single business account within a compressed timeframe is consistent "
        "with structuring to evade CTR reporting requirements under 31 USC 5324. "
        "The velocity and uniformity of amounts suggest coordinated activity."
    )


# ===================================================================
# FIXTURE 6: Indian Jurisdiction Scenario (INR + PMLA)
# 15 criminal sources sending 9.05L-9.75L INR each
# PAN entity type codes: P=Person (individual), C=Company (business)
# IFSC format: 4 alpha bank code + 0 + 6 alphanumeric branch code
# v11.0 [MED-23]: Added expected_total_criminal_inflow
# v12.0 [MED-28]: Added legitimate_nodes=[] for structural parity
# ===================================================================

@pytest.fixture
def indian_scenario() -> dict[str, Any]:
    """
    Structuring in INR for PMLA compliance testing.
    Mule: "mule_in_1" (Mumbai business).
    15 criminal sources sending 9.05L-9.75L INR each.
    Valid IFSC codes and PAN numbers with correct entity type codes.
    Jurisdiction: fiu_ind (FIU-IND STR format).

    PAN FORMAT: AAAAA9999A
    - Position 4 (0-indexed) = entity type
    - P = Individual Person, C = Company, H = Hindu Undivided Family
    Mule PAN: AABCC1234D (position 4 = C for Company) ✓
    Source PAN: ABCPP####F (position 4 = P for Person) ✓

    15 sources >= STRUCTURING_MIN_COUNT_INR (10) ✓
    Thresholds: STRUCTURING_MIN_AMOUNT_INR=900000, STRUCTURING_MAX_AMOUNT_INR=980000
    MATH: sum(900000 + 5000*i for i=1..15) = 13500000 + 600000 = ₹14,100,000
    """
    base_time = 1700000000
    transactions: list[dict] = []
    nodes: dict[str, dict] = {}
    criminal_sources: list[str] = []

    nodes["mule_in_1"] = make_node(
        "mule_in_1", "Shell Trading Pvt Ltd", "business", "IN",
        ifsc_code="HDFC0001234", pan_number="AABCC1234D",
        address="14 MG Road, Mumbai, Maharashtra 400001",
    )

    for i in range(1, 16):
        src_id = f"src_in_{i:02d}"
        criminal_sources.append(src_id)
        nodes[src_id] = make_node(
            src_id, f"Source Person IN {i}", "individual", "IN",
            ifsc_code=f"SBIN0{i:06d}",
            pan_number=f"ABCPP{i:04d}F",
            address=f"{i} Nehru Nagar, Mumbai, MH 400001",
        )
        amount = Decimal("900000") + Decimal(str(i * 5000))
        transactions.append(make_transaction(
            tx_id=f"TX-IN-STRUCT-{i:03d}",
            source=src_id,
            target="mule_in_1",
            amount=amount,
            currency="INR",
            timestamp=base_time + (i * 3600),
        ))

    return {
        "transactions": transactions,
        "nodes": nodes,
        "criminal_nodes": sorted(criminal_sources + ["mule_in_1"]),
        "mule_node": "mule_in_1",
        "criminal_sources": sorted(criminal_sources),
        # v12.0 [MED-28]: Added for structural parity with structuring_scenario.
        # This fixture has no legitimate nodes, but including the key prevents
        # KeyError in test code that iterates scenario["legitimate_nodes"].
        "legitimate_nodes": [],
        "jurisdiction": "fiu_ind",
        # v11.0 [MED-23]: Added for parity with structuring_scenario.
        # MATH: sum(900000 + 5000*i for i=1..15) = 13,500,000 + 600,000 = 14,100,000
        "expected_total_criminal_inflow": Decimal("14100000"),
    }


# ===================================================================
# FIXTURE 7: Multi-edge Scenario (same source->target, multiple txns)
# Tests that MultiDiGraph preserves all edges (DiGraph loses 2 of 3)
# ===================================================================

@pytest.fixture
def multi_edge_scenario() -> dict[str, Any]:
    """
    Tests MultiDiGraph: 3 transactions between same A->B pair.
    DiGraph would silently overwrite -- only 1 edge would survive.
    MultiDiGraph preserves all 3.
    This is a CRITICAL test: if this fails, the system is using DiGraph
    and silently losing transaction data, which is a regulatory violation.
    """

    return {
        "transactions": [
            make_transaction("TX-ME-1", "A", "B", Decimal("9100"), timestamp=1700000000),
            make_transaction("TX-ME-2", "A", "B", Decimal("9200"), timestamp=1700003600),
            make_transaction("TX-ME-3", "A", "B", Decimal("9300"), timestamp=1700007200),
        ],
        "nodes": {
            "A": make_node("A", "Source A"),
            "B": make_node("B", "Target B"),
        },
    }


# ===================================================================
# FIXTURE 8: Structuring Boundary Scenario
# Tests exact threshold boundaries for STRUCTURING_MIN/MAX_AMOUNT_USD
# Also tests STRUCTURING_MIN_COUNT boundary (exactly 8 in-range < 10)
# v12.0 [MED-27]: should_detect → in_range_sources; expected_scenario_detected added
# ===================================================================

@pytest.fixture
def structuring_boundary_scenario() -> dict[str, Any]:
    """
    Boundary testing for structuring detection thresholds.

    Tests three categories of individual amounts:
    - BELOW threshold ($8,999): Must NOT be counted as in-range
    - EXACT threshold ($9,000): Must be counted (inclusive: >=)
    - AT max ($9,800): Must be counted (inclusive: <=)
    - ABOVE max ($9,801): Must NOT be counted
    - AT CTR ($10,000): Must NOT be counted

    AGGREGATE-LEVEL DETECTION:
    This fixture has exactly 8 sources with in-range amounts.
    8 < STRUCTURING_MIN_COUNT=10, so structuring SHOULD NOT be detected
    at the scenario level even though individual amounts are in the
    suspicious band.

    v12.0 [MED-27]: Renamed `should_detect` → `in_range_sources` to eliminate
    semantic confusion. `should_detect` implied the scenario triggers detection,
    but it does NOT (8 < 10). Added `expected_scenario_detected: False` to make
    the expected aggregate-level outcome explicit.

    NOTE [LOW-04]: A separate fixture with exactly 10 in-range sources
    (at the MIN_COUNT boundary) is deferred to Phase B testing.
    """
    base_time = 1700000000
    transactions: list[dict] = []
    nodes: dict[str, dict] = {}

    nodes["boundary_mule"] = make_node("boundary_mule", "Boundary Test Corp", "business")

    boundary_amounts = [
        ("below_1", Decimal("8999"), False),
        ("exact_min", Decimal("9000"), True),
        ("above_min", Decimal("9001"), True),
        ("mid_range_1", Decimal("9200"), True),
        ("mid_range_2", Decimal("9400"), True),
        ("mid_range_3", Decimal("9500"), True),
        ("mid_range_4", Decimal("9600"), True),
        ("mid_range_5", Decimal("9700"), True),
        ("exact_max", Decimal("9800"), True),
        ("above_max", Decimal("9801"), False),
        ("well_below", Decimal("3000"), False),
        ("at_ctr", Decimal("10000"), False),
    ]

    criminal_sources: list[str] = []
    in_range_sources: list[str] = []
    for i, (name, amount, is_in_range) in enumerate(boundary_amounts):
        src_id = f"bnd_{name}"
        nodes[src_id] = make_node(src_id, f"Boundary Source {name}")
        transactions.append(make_transaction(
            tx_id=f"TX-BND-{i+1:03d}",
            source=src_id,
            target="boundary_mule",
            amount=amount,
            timestamp=base_time + (i * 1800),
        ))
        if is_in_range:
            criminal_sources.append(src_id)
            in_range_sources.append(src_id)

    return {
        "transactions": transactions,
        "nodes": nodes,
        "mule_node": "boundary_mule",
        "criminal_sources": sorted(criminal_sources),
        # v12.0 [MED-27]: Renamed from `should_detect` to eliminate ambiguity.
        # These are sources with amounts in [$9,000, $9,800] — they have
        # individually suspicious amounts but the aggregate count (8) is
        # below STRUCTURING_MIN_COUNT (10).
        "in_range_sources": sorted(in_range_sources),
        "should_not_detect": sorted([
            "bnd_below_1", "bnd_above_max", "bnd_well_below", "bnd_at_ctr"
        ]),
        "jurisdiction": "fincen",
        "total_in_range": 8,
        # v12.0 [MED-27]: Explicit aggregate-level expected outcome.
        # 8 in-range sources < STRUCTURING_MIN_COUNT (10) → no detection.
        "expected_scenario_detected": False,
    }


# ===================================================================
# FIXTURE 9: Empty Graph Scenario
# Zero nodes, zero transactions -- tests null/edge cases
# ===================================================================

@pytest.fixture
def empty_graph_scenario() -> dict[str, Any]:
    """
    Empty graph with no nodes and no transactions.
    Tests Rule 19: Empty graph fetch = FAIL (raise ValueError).
    """

    return {
        "transactions": [],
        "nodes": {},
        "criminal_nodes": [],
        "text_evidence": [],
        "jurisdiction": "fincen",
    }


# ===================================================================
# FIXTURE 10: Legitimate-Only Scenario
# Completely clean graph with ONLY normal, low-value transactions.
# v12.0 [LOW-14]: Added expected_node_count, expected_tx_count.
# ===================================================================

@pytest.fixture
def legitimate_only_scenario() -> dict[str, Any]:
    """
    A clean financial graph with no criminal activity whatsoever.
    - 5 employers × 4 employees = 20 transactions
    - 5 employer nodes + 20 employee nodes = 25 total nodes
    - All amounts $3,000 (well below STRUCTURING_MIN_AMOUNT_USD = $9,000)
    - Fan-OUT pattern (employer → employees), NOT fan-IN
    - No layering chains (all paths are length 1)
    Tests: detect_structuring -> False, detect_layering -> False
    """
    base_time = 1700000000
    transactions: list[dict] = []
    nodes: dict[str, dict] = {}

    for emp_i in range(1, 6):
        emp_id = f"legit_corp_{emp_i}"
        nodes[emp_id] = make_node(emp_id, f"Legitimate Corp {emp_i}", "business")
        for emp_j in range(1, 5):
            person_id = f"employee_{emp_i}_{emp_j}"
            nodes[person_id] = make_node(person_id, f"Employee {emp_i}-{emp_j}")
            transactions.append(make_transaction(
                tx_id=f"TX-SAL-{emp_i}-{emp_j}",
                source=emp_id,
                target=person_id,
                amount=Decimal("3000"),
                timestamp=base_time + (emp_i * 86400) + (emp_j * 3600),
            ))

    return {
        "transactions": transactions,
        "nodes": nodes,
        "criminal_nodes": [],
        "jurisdiction": "fincen",
        "expected_confidence_score": 0.0,
        # v12.0 [LOW-14]: Explicit counts for quick sanity assertions.
        "expected_node_count": 25,   # 5 employers + 20 employees
        "expected_tx_count": 20,     # 5 employers × 4 employees
    }


# ===================================================================
# FIXTURE 11: Layering Boundary Scenario
# Decay rate boundary testing with 6 chains
# v9.0 [MED-18]: "below1pct" is actually exactly 1% — noted in comments
# ===================================================================

@pytest.fixture
def layering_boundary_scenario() -> dict[str, Any]:
    """
    Six chains testing decay rate boundaries:
    1. min2pct:   2% decay (= DECAY_RATE_MIN) -> SHOULD detect ✓ (inclusive)
    2. max5pct:   5% decay (= DECAY_RATE_MAX) -> SHOULD detect ✓ (inclusive)
    3. exact1pct: 1% decay (< DECAY_RATE_MIN) -> should NOT detect ✗
       (v9.0 [MED-18]: renamed from "below1pct" in docs; chain ID retained)
    4. above8pct: 8% decay (> DECAY_RATE_MAX) -> should NOT detect ✗
    5. nodecay:   0% decay -> should NOT detect ✗
    6. short:     Only 2 hops (< MIN_CHAIN_LENGTH=3) -> should NOT detect ✗

    Mathematical verification of each chain:
      min2pct:  (100000-98000)/100000 = 2.0%, (98000-96040)/98000 = 2.0%,
                (96040-94119.20)/96040 = 2.0%. 4 hops >= 3 ✓
      max5pct:  (100000-95000)/100000 = 5.0%, (95000-90250)/95000 = 5.0%,
                (90250-85737.50)/90250 = 5.0%. 4 hops >= 3 ✓
      exact1pct:(100000-99000)/100000 = 1.0%. 4 hops >= 3 but 1% < 2% ✗
      above8pct:(100000-92000)/100000 = 8.0%. 4 hops >= 3 but 8% > 5% ✗
      nodecay:  (100000-100000)/100000 = 0.0%. 3 hops >= 3 but 0% < 2% ✗
      short:    (100000-97000)/100000 = 3.0%. 2 hops < 3 ✗
    """
    base_time = 1700000000
    transactions: list[dict] = []
    nodes: dict[str, dict] = {}

    def _add_chain(prefix: str, amounts: list[Decimal], time_offset: int) -> list[str]:
        chain_ids = [f"{prefix}_{j}" for j in range(len(amounts) + 1)]
        for cid in chain_ids:
            nodes[cid] = make_node(cid, f"Boundary {cid}")
        for j, amt in enumerate(amounts):
            transactions.append(make_transaction(
                tx_id=f"TX-LBD-{prefix}-{j+1:03d}",
                source=chain_ids[j],
                target=chain_ids[j + 1],
                amount=amt,
                timestamp=base_time + time_offset + (j * 7200),
            ))
        return chain_ids

    chain1 = _add_chain("min2pct", [
        Decimal("100000"), Decimal("98000"), Decimal("96040"), Decimal("94119.20")
    ], 0)

    chain2 = _add_chain("max5pct", [
        Decimal("100000"), Decimal("95000"), Decimal("90250"), Decimal("85737.50")
    ], 100000)

    chain3 = _add_chain("below1pct", [  # Actually 1.0% — see MED-18 note
        Decimal("100000"), Decimal("99000"), Decimal("98010"), Decimal("97029.90")
    ], 200000)

    chain4 = _add_chain("above8pct", [
        Decimal("100000"), Decimal("92000"), Decimal("84640"), Decimal("77868.80")
    ], 300000)

    chain5 = _add_chain("nodecay", [
        Decimal("100000"), Decimal("100000"), Decimal("100000")
    ], 400000)

    chain6 = _add_chain("short", [
        Decimal("100000"), Decimal("97000")
    ], 500000)

    return {
        "transactions": transactions,
        "nodes": nodes,
        "should_detect_chains": {
            "min2pct": chain1,
            "max5pct": chain2,
        },
        "should_not_detect_chains": {
            "below1pct": chain3,
            "above8pct": chain4,
            "nodecay": chain5,
            "short": chain6,
        },
        "jurisdiction": "fincen",
    }


# ===================================================================
# FIXTURES 12-16: v7.0+ additions
# ===================================================================

# ===================================================================
# FIXTURE 12: Duplicate Transaction ID Scenario
# v12.0 [HIGH-20]: Added expected_kept_amounts and expected_kept_routes
# ===================================================================

@pytest.fixture
def duplicate_txid_scenario() -> dict[str, Any]:
    """
    GraphFragment with duplicate transaction IDs.
    Rule 27: keep FIRST occurrence, log warning for each duplicate.
    Expected after deduplication: 3 transactions, 2 warnings.

    DEDUP VERIFICATION (v12.0 [HIGH-20]):
    Input order:
      1. TX-DUP-001: A→B, $9,100, t=1700000000  ← KEPT (first occurrence)
      2. TX-DUP-001: A→B, $9,500, t=1700003600  ← DROPPED (duplicate ID)
      3. TX-DUP-002: B→C, $9,200, t=1700007200  ← KEPT (unique ID)
      4. TX-DUP-003: C→D, $9,300, t=1700010800  ← KEPT (first occurrence)
      5. TX-DUP-003: A→D, $9,400, t=1700014400  ← DROPPED (duplicate ID)

    CRITICAL: TX-DUP-003's duplicate has a DIFFERENT route (A→D vs C→D).
    Keeping the wrong version changes graph topology: the edge C→D would
    be replaced by A→D, altering node D's predecessors and breaking any
    path analysis that depends on the B→C→D chain.

    After dedup:
      TX-DUP-001: A→B, $9,100 (first-seen amount, NOT $9,500)
      TX-DUP-002: B→C, $9,200 (unique, unchanged)
      TX-DUP-003: C→D, $9,300 (first-seen route, NOT A→D)
    """

    return {
        "transactions": [
            make_transaction("TX-DUP-001", "A", "B", Decimal("9100"), timestamp=1700000000),
            make_transaction("TX-DUP-001", "A", "B", Decimal("9500"), timestamp=1700003600),
            make_transaction("TX-DUP-002", "B", "C", Decimal("9200"), timestamp=1700007200),
            make_transaction("TX-DUP-003", "C", "D", Decimal("9300"), timestamp=1700010800),
            make_transaction("TX-DUP-003", "A", "D", Decimal("9400"), timestamp=1700014400),
        ],
        "nodes": {
            "A": make_node("A", "Entity A"),
            "B": make_node("B", "Entity B"),
            "C": make_node("C", "Entity C"),
            "D": make_node("D", "Entity D"),
        },
        "expected_tx_count_after_dedup": 3,
        "expected_kept_tx_ids": sorted(["TX-DUP-001", "TX-DUP-002", "TX-DUP-003"]),
        "expected_warnings": 2,
        # v12.0 [HIGH-20]: Amounts of the FIRST-SEEN versions (keep-first rule).
        # Tests must verify these specific amounts to confirm correct dedup order.
        # If a "keep-last" bug exists, amounts would be [9500, 9200, 9400].
        "expected_kept_amounts": [Decimal("9100"), Decimal("9200"), Decimal("9300")],
        # v12.0 [HIGH-20]: Routes of the FIRST-SEEN versions.
        # TX-DUP-003 first-seen is C→D, NOT A→D. Verifying routes ensures
        # the graph topology after dedup is correct (B→C→D chain intact).
        "expected_kept_routes": [("A", "B"), ("B", "C"), ("C", "D")],
    }


@pytest.fixture
def self_loop_scenario() -> dict[str, Any]:
    """
    Graph containing self-loops (source_node == target_node).
    Step 2.5: Self-loops LOGGED as warnings but NOT rejected.
    Expected: All 4 transactions ingested. 2 self-loop warnings logged.
    """

    return {
        "transactions": [
            make_transaction("TX-SELF-001", "A", "A", Decimal("5000"), timestamp=1700000000),
            make_transaction("TX-SELF-002", "A", "B", Decimal("9200"), timestamp=1700003600),
            make_transaction("TX-SELF-003", "B", "B", Decimal("3000"), timestamp=1700007200),
            make_transaction("TX-SELF-004", "B", "C", Decimal("9100"), timestamp=1700010800),
        ],
        "nodes": {
            "A": make_node("A", "Entity A"),
            "B": make_node("B", "Entity B"),
            "C": make_node("C", "Entity C"),
        },
        "expected_self_loops": 2,
        "expected_total_tx": 4,
    }


# ===================================================================
# FIXTURE 14: Confidence Scoring Scenario
# v8.0 [CRIT-05]: Boolean evidence fields. v8.0 [ENH-01]: exact_boundary.
# v11.0 [LOW-10]: confidence_threshold changed from Decimal to float.
# v12.0 [HIGH-19]: state_builder added per test case for zero-mapping wiring.
# ===================================================================

@pytest.fixture
def confidence_scoring_scenario() -> dict[str, Any]:
    """
    Fixture providing data to test confidence score computation and gating.

    v10.0 Confidence Scoring Formula (unchanged from v8.0):
      base = 0.3 per detected typology (structuring OR layering)
      base = 0.6 if BOTH detected
      + 0.2 if text_corroborates_ledger is True
      + 0.2 if has_suspicious_discrepancy is True
      Result clamped to [0.0, 1.0]

    IEEE 754 SAFETY (v10.0 proven, v12.0 BITWISE verified):
      All expected_score values in this fixture survive float comparison:
        0.7 = 0.3 + 0.2 + 0.2 (exact), 0.3 (exact), 0.8 = 0.6 + 0.2 (exact),
        0.0 (exact), 0.5 = 0.3 + 0.2 (exact — errors cancel in IEEE 754).
      struct.pack('d', computed) == struct.pack('d', expected) for ALL 5 cases.

    v11.0 [LOW-10]: confidence_threshold is float (not Decimal) because
    compute_confidence uses _CONFIDENCE_THRESHOLD_FLOAT for comparison.

    v12.0 [HIGH-19] STATE BUILDER:
    Each test case now includes a `state_builder` dict containing the EXACT
    InvestigationState fields that compute_confidence reads. This eliminates
    the field-name mapping burden on test authors.

    FIELD NAME MAPPING (v12.0 documentation):
    Fixture semantic field          → InvestigationState field
    ─────────────────────────────── → ──────────────────────────────────────
    structuring_detected: True      → structuring_results: [{"id": "stub"}]
    structuring_detected: False     → structuring_results: []
    layering_detected: True         → layering_results: [{"id": "stub"}]
    layering_detected: False        → layering_results: []
    evidence_corroborates: True     → evidence_package.text_corroborates_ledger: True
    discrepancy_found: True         → evidence_package.has_suspicious_discrepancy: True

    USAGE IN TESTS:
      for name, case in scenario["test_cases"].items():
          state = {**initial_state, **case["state_builder"]}
          result = compute_confidence(state)
          assert result["confidence_score"] == case["expected_score"]
    Zero mapping. Zero ambiguity. No undocumented translation layer.

    Test Cases (5):
    1. HIGH_CONFIDENCE: structuring(0.3) + corroboration(0.2) + discrepancy(0.2) = 0.7
    2. LOW_CONFIDENCE: only layering hint, no evidence -> 0.3 < threshold -> suppressed
    3. BOTH_TYPOLOGIES: 0.3 + 0.3 + corroboration(0.2) = 0.8
    4. ZERO_CONFIDENCE: no detection -> 0.0
    5. EXACT_BOUNDARY: structuring(0.3) + corroboration(0.2) = 0.5 == threshold -> SAR generated
    """

    # v12.0 [HIGH-19]: Stub result dicts for populating state_builder.
    # compute_confidence calls bool(state.get("structuring_results")) —
    # any non-empty list evaluates to True. A single-element list with
    # a minimal dict is the smallest truthy value.
    _STUB_STRUCTURING = [{"id": "stub-struct", "involved_entities": []}]
    _STUB_LAYERING = [{"id": "stub-layer", "involved_entities": []}]

    return {
        "test_cases": {
            "high_confidence": {
                "structuring_detected": True,
                "layering_detected": False,
                "evidence_corroborates": True,
                "discrepancy_found": True,
                "expected_score": 0.7,
                "expected_sar_generated": True,
                # v12.0 [HIGH-19]: Direct state fields for test wiring.
                "state_builder": {
                    "structuring_results": _STUB_STRUCTURING,
                    "layering_results": [],
                    "evidence_package": {
                        "text_corroborates_ledger": True,
                        "has_suspicious_discrepancy": True,
                        "discrepancies": [],
                        "corroborations": [],
                    },
                },
            },
            "low_confidence": {
                "structuring_detected": False,
                "layering_detected": True,
                "evidence_corroborates": False,
                "discrepancy_found": False,
                "expected_score": 0.3,
                "expected_sar_generated": False,
                "state_builder": {
                    "structuring_results": [],
                    "layering_results": _STUB_LAYERING,
                    "evidence_package": {
                        "text_corroborates_ledger": False,
                        "has_suspicious_discrepancy": False,
                        "discrepancies": [],
                        "corroborations": [],
                    },
                },
            },
            "both_typologies": {
                "structuring_detected": True,
                "layering_detected": True,
                "evidence_corroborates": True,
                "discrepancy_found": False,
                "expected_score": 0.8,
                "expected_sar_generated": True,
                "state_builder": {
                    "structuring_results": _STUB_STRUCTURING,
                    "layering_results": _STUB_LAYERING,
                    "evidence_package": {
                        "text_corroborates_ledger": True,
                        "has_suspicious_discrepancy": False,
                        "discrepancies": [],
                        "corroborations": [],
                    },
                },
            },
            "zero_confidence": {
                "structuring_detected": False,
                "layering_detected": False,
                "evidence_corroborates": False,
                "discrepancy_found": False,
                "expected_score": 0.0,
                "expected_sar_generated": False,
                "state_builder": {
                    "structuring_results": [],
                    "layering_results": [],
                    "evidence_package": {
                        "text_corroborates_ledger": False,
                        "has_suspicious_discrepancy": False,
                        "discrepancies": [],
                        "corroborations": [],
                    },
                },
            },
            "exact_boundary": {
                "structuring_detected": True,
                "layering_detected": False,
                "evidence_corroborates": True,
                "discrepancy_found": False,
                "expected_score": 0.5,
                "expected_sar_generated": True,  # 0.5 is NOT < 0.5
                "state_builder": {
                    "structuring_results": _STUB_STRUCTURING,
                    "layering_results": [],
                    "evidence_package": {
                        "text_corroborates_ledger": True,
                        "has_suspicious_discrepancy": False,
                        "discrepancies": [],
                        "corroborations": [],
                    },
                },
            },
        },
        # v11.0 [LOW-10]: float, not Decimal. Matches compute_confidence usage.
        "confidence_threshold": 0.5,
    }


# ===================================================================
# FIXTURE 15: Super Node Scenario
# v12.0 [MED-26]: Documented structuring confound. Added metadata.
# ===================================================================

@pytest.fixture
def super_node_scenario() -> dict[str, Any]:
    """
    Graph with a super-node (hub) that exceeds MAX_NODE_DEGREE (500).
    Rule 21: Nodes with degree > MAX_NODE_DEGREE are logged and
    treated as dead ends during traversal.

    v12.0 [MED-26] STRUCTURING CONFOUND:
    All 501 transactions send $9,100 each, which is inside the structuring
    range [$9,000, $9,800]. With 501 sources >> STRUCTURING_MIN_COUNT=10,
    the hub_node technically qualifies as a structuring mule if the
    structuring detector evaluates it.

    EXPECTED BEHAVIOR: Rule 21 (super-node pruning) should take priority
    over structuring detection. A node exceeding MAX_NODE_DEGREE is a
    high-volume aggregator (payroll hub, exchange, clearing house) — NOT
    a structuring mule. Structuring detection MUST exclude nodes that
    have been flagged as super-nodes.

    If a test sees both super-node WARNING and structuring DETECTION for
    hub_node, the detection order is wrong: super-node pruning must
    execute BEFORE (or as a filter within) structuring fan-in analysis.

    NOTE [LOW-05]: Boundary test at exactly 500 (should NOT skip)
    is deferred to Phase B testing. Current fixture tests 501 (SHOULD skip).
    """
    base_time = 1700000000
    transactions: list[dict] = []
    nodes: dict[str, dict] = {}

    nodes["hub_node"] = make_node("hub_node", "Mega Corp Payroll Hub", "business")
    for i in range(1, 502):
        src_id = f"super_src_{i:04d}"
        nodes[src_id] = make_node(src_id, f"Super Source {i}")
        transactions.append(make_transaction(
            tx_id=f"TX-SUPER-{i:04d}",
            source=src_id,
            target="hub_node",
            amount=Decimal("9100"),
            timestamp=base_time + (i * 60),
        ))

    return {
        "transactions": transactions,
        "nodes": nodes,
        "hub_node": "hub_node",
        "hub_degree": 501,
        "max_node_degree": 500,
        "jurisdiction": "fincen",
        # v12.0 [MED-26]: Explicit documentation of the confound.
        # All 501 sources send $9,100 (in structuring range), count >> MIN_COUNT.
        # Tests should assert that structuring detection is SUPPRESSED for
        # hub_node because Rule 21 takes priority. If structuring_results
        # contains hub_node, the detection ordering is incorrect.
        "expected_structuring_confound": True,
        "expected_structuring_suppressed_for_hub": True,
    }


@pytest.fixture
def mixed_currency_scenario() -> dict[str, Any]:
    """
    Same mule receives BOTH USD and INR structured deposits.
    Rule 14: Group transactions by currency BEFORE threshold comparison.

    USD cluster: 12 sources sending $9,100-$9,540 each
      - 12 >= STRUCTURING_MIN_COUNT(10) ✓
      - All in [$9,000, $9,800] ✓
      - Total: $111,840 (verified in module header)

    INR cluster: 12 sources sending ₹9.05L-₹9.60L each
      - 12 >= STRUCTURING_MIN_COUNT_INR(10) ✓
      - All in [₹9,00,000, ₹9,80,000] ✓
      - Total: ₹11,190,000 (verified in module header)
    """
    base_time = 1700000000
    transactions: list[dict] = []
    nodes: dict[str, dict] = {}

    nodes["mixed_mule"] = make_node(
        "mixed_mule", "Global Shell LLC", "business", "US",
        swift_code="GSHLUS33",
    )

    usd_criminals: list[str] = []
    inr_criminals: list[str] = []

    for i in range(1, 13):
        src_id = f"usd_src_{i:02d}"
        usd_criminals.append(src_id)
        nodes[src_id] = make_node(src_id, f"USD Source {i}")
        amount = Decimal("9060") + Decimal(str(i * 40))
        transactions.append(make_transaction(
            tx_id=f"TX-MXC-USD-{i:03d}",
            source=src_id,
            target="mixed_mule",
            amount=amount,
            currency="USD",
            timestamp=base_time + (i * 1800),
        ))

    for i in range(1, 13):
        src_id = f"inr_src_{i:02d}"
        inr_criminals.append(src_id)
        nodes[src_id] = make_node(
            src_id, f"INR Source {i}", "individual", "IN",
            ifsc_code=f"HDFC0{i:06d}",
            pan_number=f"MIXPP{i:04d}F",
        )
        amount = Decimal("900000") + Decimal(str(i * 5000))
        transactions.append(make_transaction(
            tx_id=f"TX-MXC-INR-{i:03d}",
            source=src_id,
            target="mixed_mule",
            amount=amount,
            currency="INR",
            timestamp=base_time + 100000 + (i * 1800),
        ))

    return {
        "transactions": transactions,
        "nodes": nodes,
        "mule_node": "mixed_mule",
        "usd_criminal_sources": sorted(usd_criminals),
        "inr_criminal_sources": sorted(inr_criminals),
        "all_criminal": sorted(usd_criminals + inr_criminals + ["mixed_mule"]),
        "jurisdiction": "fincen",
        "expected_usd_total": Decimal("111840"),
        "expected_inr_total": Decimal("11190000"),
    }
