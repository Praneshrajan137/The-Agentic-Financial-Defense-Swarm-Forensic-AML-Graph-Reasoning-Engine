"""
Baseline Purple Agent
=====================
Minimal viable investigator that demonstrates the A2A protocol.

This agent is intentionally SIMPLE. It:
1. Connects to the Green Agent
2. Calls get_transactions for a few nodes
3. Applies a simple heuristic (count incoming edges)
4. Returns a verdict

Purpose: Prove the benchmark is solvable and the interface works.
"""

import requests
from typing import Dict, Any, List, Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselinePurpleAgent:
    """Simplest possible investigation agent."""
    
    def __init__(self, green_agent_url: str = "http://localhost:8000", participant_id: str = "baseline_purple_agent"):
        """
        Initialize with Green Agent URL.
        
        Args:
            green_agent_url: URL of the Green Agent A2A interface
            participant_id: Unique identifier for this agent
        """
        self.base_url = green_agent_url
        self.participant_id = participant_id
        self.tool_calls = 0
        
        # Headers for all requests
        self.headers = {
            "Content-Type": "application/json",
            "X-Participant-ID": self.participant_id
        }
    
    def get_transactions(self, account_id: str, limit: int = 100) -> List[Dict]:
        """
        Call Green Agent's get_transactions tool.
        
        Args:
            account_id: Account ID to query
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction dictionaries
        """
        self.tool_calls += 1
        
        try:
            response = requests.post(
                f"{self.base_url}/a2a/tools/get_transactions",
                headers=self.headers,
                json={
                    "account_id": account_id,
                    "limit": limit,
                    "direction": "both"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('transactions', [])
            else:
                logger.warning(f"Error getting transactions: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Exception getting transactions: {e}")
            return []
    
    def get_kyc_profile(self, account_id: str) -> Dict:
        """
        Call Green Agent's get_kyc_profile tool.
        
        Args:
            account_id: Account ID to query
            
        Returns:
            KYC profile dictionary
        """
        self.tool_calls += 1
        
        try:
            response = requests.post(
                f"{self.base_url}/a2a/tools/get_kyc_profile",
                headers=self.headers,
                json={"account_id": account_id},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            logger.error(f"Exception getting KYC profile: {e}")
            return {}
    
    def get_connections(self, account_id: str) -> List[Dict]:
        """
        Call Green Agent's get_account_connections tool.
        
        Args:
            account_id: Account ID to query
            
        Returns:
            List of connection dictionaries
        """
        self.tool_calls += 1
        
        try:
            response = requests.post(
                f"{self.base_url}/a2a/tools/get_account_connections",
                headers=self.headers,
                json={"account_id": account_id, "depth": 1},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('connections', [])
            else:
                return []
        except Exception as e:
            logger.error(f"Exception getting connections: {e}")
            return []
    
    def get_evidence(self, keyword: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Call Green Agent's get_evidence tool.
        
        Args:
            keyword: Optional keyword to search for
            limit: Maximum number of documents to return
            
        Returns:
            List of evidence document dictionaries
        """
        self.tool_calls += 1
        
        try:
            payload = {"limit": limit}
            if keyword:
                payload["contains_keyword"] = keyword
            
            response = requests.post(
                f"{self.base_url}/a2a/tools/get_evidence",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('documents', [])
            else:
                return []
        except Exception as e:
            logger.error(f"Exception getting evidence: {e}")
            return []
    
    def simple_structuring_heuristic(self, start_node: str) -> Dict[str, Any]:
        """
        Extremely simple structuring detector.
        
        Logic:
        1. Get connections for start node
        2. If a node has 15+ senders with similar amounts, flag as structuring
        3. Return verdict
        
        Args:
            start_node: Node ID to investigate
            
        Returns:
            Investigation result dictionary
        """
        logger.info(f"Investigating node: {start_node}")
        
        # Get this node's incoming connections
        connections = self.get_connections(start_node)
        
        # Filter to senders only
        senders = [c for c in connections if c.get('relationship') in ['sender', 'both']]
        
        logger.info(f"  Found {len(senders)} incoming connections")
        
        # Simple heuristic: If 15+ senders, it's suspicious
        if len(senders) >= 15:
            logger.warning("  SUSPICIOUS: High fan-in pattern detected!")
            
            # Get transaction details to check amounts
            txns = self.get_transactions(start_node, limit=50)
            
            # Check if amounts cluster near $9,000-$9,800
            structuring_amounts = [
                t for t in txns 
                if t.get('target') == start_node and 9000 <= t.get('amount', 0) <= 9800
            ]
            
            if len(structuring_amounts) >= 10:
                return {
                    "verdict": "STRUCTURING_DETECTED",
                    "confidence": 0.85,
                    "evidence": {
                        "pattern": "fan_in",
                        "sender_count": len(senders),
                        "suspicious_tx_count": len(structuring_amounts),
                        "mule_node": start_node
                    }
                }
        
        logger.info("  No obvious structuring detected")
        return {
            "verdict": "CLEAN",
            "confidence": 0.6,
            "evidence": {}
        }
    
    def simple_layering_heuristic(self, start_node: str, max_depth: int = 5) -> Dict[str, Any]:
        """
        Simple layering detector using BFS traversal.
        
        Logic:
        1. Follow outgoing transactions
        2. Look for rapid chain of transfers with decaying amounts
        
        Args:
            start_node: Node ID to start from
            max_depth: Maximum depth to traverse
            
        Returns:
            Investigation result dictionary
        """
        logger.info(f"Checking for layering from node: {start_node}")
        
        # Get transactions
        txns = self.get_transactions(start_node, limit=50)
        
        # Look for outgoing wire transfers with significant amounts
        outgoing = [
            t for t in txns 
            if t.get('source') == start_node 
            and t.get('amount', 0) > 10000
            and t.get('transaction_type') == 'wire'
        ]
        
        if len(outgoing) >= 3:
            # Check if amounts are decreasing (decay pattern)
            amounts = sorted([t.get('amount', 0) for t in outgoing], reverse=True)
            if len(amounts) >= 3:
                # Check for decay pattern
                decay_detected = all(
                    amounts[i] > amounts[i+1] * 0.95  # Allow 5% variance
                    for i in range(len(amounts)-1)
                )
                
                if decay_detected:
                    return {
                        "verdict": "LAYERING_SUSPECTED",
                        "confidence": 0.7,
                        "evidence": {
                            "pattern": "chain_decay",
                            "source_node": start_node,
                            "outgoing_count": len(outgoing),
                            "amount_range": [min(amounts), max(amounts)]
                        }
                    }
        
        return {
            "verdict": "CLEAN",
            "confidence": 0.6,
            "evidence": {}
        }
    
    def run_investigation(self, start_nodes: List[str]) -> Dict[str, Any]:
        """
        Run investigation on multiple nodes.
        
        Args:
            start_nodes: List of account IDs to investigate
            
        Returns:
            Investigation report dictionary
        """
        findings = []
        identified_crimes = []
        suspicious_accounts = []
        
        # Check for evidence documents first
        evidence_docs = self.get_evidence(keyword="structuring", limit=10)
        if evidence_docs:
            logger.info(f"Found {len(evidence_docs)} evidence documents")
        
        for node in start_nodes:
            # Check for structuring
            structuring_result = self.simple_structuring_heuristic(node)
            if structuring_result['verdict'] != 'CLEAN':
                findings.append(structuring_result)
                identified_crimes.append({
                    'crime_type': 'structuring',
                    'nodes': [node]
                })
                suspicious_accounts.append(node)
            
            # Check for layering
            layering_result = self.simple_layering_heuristic(node)
            if layering_result['verdict'] != 'CLEAN':
                findings.append(layering_result)
                identified_crimes.append({
                    'crime_type': 'layering',
                    'nodes': [node]
                })
                suspicious_accounts.append(node)
        
        return {
            "participant_id": self.participant_id,
            "total_nodes_checked": len(start_nodes),
            "suspicious_nodes": len(findings),
            "findings": findings,
            "tool_calls_used": self.tool_calls,
            "identified_crimes": identified_crimes,
            "suspicious_accounts": list(set(suspicious_accounts)),
            "narrative": self._generate_narrative(findings)
        }
    
    def _generate_narrative(self, findings: List[Dict]) -> str:
        """Generate a simple narrative from findings."""
        if not findings:
            return "Investigation complete. No suspicious activity detected."
        
        narrative = "Investigation Summary:\n"
        for i, finding in enumerate(findings, 1):
            verdict = finding.get('verdict', 'UNKNOWN')
            confidence = finding.get('confidence', 0)
            narrative += f"{i}. {verdict} (confidence: {confidence:.0%})\n"
        
        return narrative
    
    def submit_investigation(self, investigation_data: Dict) -> Dict:
        """
        Submit investigation for assessment.
        
        Args:
            investigation_data: Investigation findings to submit
            
        Returns:
            Assessment response dictionary
        """
        try:
            response = requests.post(
                f"{self.base_url}/a2a/investigation_assessment",
                headers=self.headers,
                json={
                    "participant_id": self.participant_id,
                    "investigation_data": investigation_data
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Assessment error: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Exception submitting investigation: {e}")
            return {}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Baseline Purple Investigation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python baseline_agent.py --nodes 0 42 100
    python baseline_agent.py --green-url http://localhost:8000 --nodes 0 1 2
        """
    )
    parser.add_argument(
        "--green-url",
        default="http://localhost:8000",
        help="Green Agent URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        default=["0", "1", "2", "42", "100"],
        help="Node IDs to investigate (default: 0 1 2 42 100)"
    )
    parser.add_argument(
        "--participant-id",
        default="baseline_purple_agent",
        help="Participant ID for tracking (default: baseline_purple_agent)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BASELINE PURPLE AGENT - Simple Investigation")
    print("=" * 60)
    print(f"Green Agent URL: {args.green_url}")
    print(f"Participant ID: {args.participant_id}")
    print(f"Nodes to investigate: {args.nodes}")
    print("=" * 60)
    
    agent = BaselinePurpleAgent(
        green_agent_url=args.green_url,
        participant_id=args.participant_id
    )
    
    # Run investigation
    report = agent.run_investigation(args.nodes)
    
    print("\n" + "=" * 60)
    print("INVESTIGATION REPORT")
    print("=" * 60)
    print(f"Nodes checked: {report['total_nodes_checked']}")
    print(f"Suspicious findings: {report['suspicious_nodes']}")
    print(f"Tool calls used: {report['tool_calls_used']}")
    
    if report['findings']:
        print("\nFindings:")
        for finding in report['findings']:
            print(f"  - {finding['verdict']} (confidence: {finding['confidence']:.0%})")
    
    print(f"\nNarrative:\n{report['narrative']}")
    
    # Submit for assessment
    print("\n" + "=" * 60)
    print("SUBMITTING FOR ASSESSMENT...")
    print("=" * 60)
    
    assessment = agent.submit_investigation(report)
    
    if assessment:
        print(f"\nSCORE: {assessment.get('score', 'N/A')}/100")
        print(f"Tool Calls: {assessment.get('tool_call_count', 0)}")
        print(f"Efficiency: {assessment.get('efficiency_score', 0):.1f} ({assessment.get('efficiency_rank', 'unknown')})")
        print(f"\nFeedback:\n{assessment.get('feedback', 'No feedback available')}")
    else:
        print("Failed to get assessment (Green Agent may need to load data first)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
