"""
Evidence Generator
==================
Generates unstructured text artifacts (emails, SARs, documents) that contain
clues required to solve financial crimes.

This forces the Purple Agent to combine:
1. Natural language understanding (reading emails)
2. Information extraction (finding entity IDs in text)
3. Graph querying (following the extracted IDs)

This is the "Sherlock Holmes" upgrade that makes the platform cognitively challenging.

v8.0: Decimal-everywhere for currency amounts. Dual-jurisdiction (FinCEN + FIU-IND).
      CRITICAL: All amounts passed as Decimal|float; formatted via _fmt_amount().
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal
import random
from faker import Faker
import logging

from ..config import (
    TIMEZONE_FINCEN,
    TIMEZONE_FIU_IND,
    EVIDENCE_DISCREPANCY_THRESHOLD_USD,
    EVIDENCE_DISCREPANCY_THRESHOLD_INR,
)

logger = logging.getLogger(__name__)


def _fmt_amount(amount: Union[Decimal, float, int], currency: str = "USD") -> str:
    """Format a monetary amount for display in evidence text.

    Accepts Decimal, float, or int. Returns '$1,234.56' for USD,
    '₹1,23,456.00' for INR (Indian numbering not implemented here,
    uses standard comma formatting).
    """
    val = float(amount) if isinstance(amount, Decimal) else amount
    if currency == "INR":
        return f"₹{val:,.2f}"
    return f"${val:,.2f}"


class EvidenceGenerator:
    """Generates synthetic investigation artifacts."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)
        
        self.fake = Faker()
    
    def generate_sar_narrative(
        self,
        subject_id: str,
        subject_name: str,
        crime_type: str,
        transaction_count: int,
        total_amount: Union[Decimal, float],
        time_window_hours: int,
        currency: str = "USD",
    ) -> Dict[str, Any]:
        """
        Generate a Suspicious Activity Report (SAR) narrative.

        Supports dual-jurisdiction: FinCEN (USD) and FIU-IND (INR).

        Args:
            subject_id: Entity ID of the subject
            subject_name: Name of the subject
            crime_type: "structuring" or "layering"
            transaction_count: Number of transactions involved
            total_amount: Total amount involved (Decimal or float)
            time_window_hours: Time window of suspicious activity
            currency: "USD" (FinCEN) or "INR" (FIU-IND)

        Returns:
            Dictionary with SAR document structure
        """
        amount_str = _fmt_amount(total_amount, currency)
        file_number = f"SAR-{random.randint(100000, 999999)}"
        date_filed = datetime.now().strftime('%Y-%m-%d')

        if currency == "INR":
            regulator = "FIU-IND"
            act_name = "Prevention of Money Laundering Act (PMLA)"
            threshold_label = "₹10,00,000 CTR threshold"
            timezone = TIMEZONE_FIU_IND
        else:
            regulator = "FinCEN"
            act_name = "Bank Secrecy Act"
            threshold_label = "$10,000 Currency Transaction Report (CTR) threshold"
            timezone = TIMEZONE_FINCEN

        if crime_type == "structuring":
            narrative = f"""SUSPICIOUS ACTIVITY REPORT
File Number: {file_number}
Date Filed: {date_filed}
Jurisdiction: {regulator}

SUBJECT INFORMATION:
Name: {subject_name}
Account ID: {subject_id}
Entity Type: Individual/Business

NARRATIVE:
Subject {subject_name} (Account ID: {subject_id}) has engaged in a pattern of
financial activity that appears designed to evade {act_name} reporting
requirements. Specifically, the subject received {transaction_count} cash deposits
over a {time_window_hours}-hour period, totaling approximately {amount_str}.

Each individual transaction was structured to remain below the {threshold_label},
with amounts exhibiting clustering patterns consistent with "smurfing" or
"structuring" typology as defined in {regulator} guidance.

The branch manager reported that multiple individuals, appearing to act in
coordination, made these deposits. Several stated the funds were for "business
expenses" but could not provide coherent explanations when questioned.

RECOMMENDATION: File SAR and monitor for continued activity."""
        else:  # layering
            narrative = f"""SUSPICIOUS ACTIVITY REPORT
File Number: {file_number}
Date Filed: {date_filed}
Jurisdiction: {regulator}

SUBJECT INFORMATION:
Name: {subject_name}
Account ID: {subject_id}
Entity Type: Individual/Business

NARRATIVE:
Subject {subject_name} (Account ID: {subject_id}) initiated a complex series of
rapid wire transfers totaling approximately {amount_str} through {transaction_count}
intermediary accounts within a {time_window_hours}-hour window.

The transaction pattern exhibits characteristics consistent with "layering" -
the second stage of money laundering where funds are moved through multiple
accounts to obscure the audit trail. Each transfer showed a 2-5% reduction in
value, potentially representing fees paid to mule account operators.

The velocity of transfers (multiple hops within {time_window_hours} hours) and
the lack of apparent economic purpose raise significant AML concerns.

RECOMMENDATION: Immediate investigation and possible law enforcement referral."""

        return {
            "document_type": "SAR",
            "file_number": file_number,
            "subject_id": subject_id,
            "subject_name": subject_name,
            "date_filed": datetime.now().isoformat(),
            "date": datetime.now().isoformat(),
            "narrative": narrative.strip(),
            "body": narrative.strip(),
            "crime_type": crime_type,
            "currency": currency,
            "jurisdiction": regulator,
        }
    
    def generate_internal_email(
        self,
        subject_id: str,
        subject_name: str,
        suspicious_behavior: str,
        sender_role: str = "Branch Manager"
    ) -> Dict[str, Any]:
        """
        Generate an internal bank email discussing suspicious activity.
        
        This is CRITICAL: The email contains the entity ID that the Purple Agent
        must extract before querying the graph.
        
        Args:
            subject_id: Entity ID (THIS IS THE KEY CLUE)
            subject_name: Name of the subject
            suspicious_behavior: Description of what was observed
            sender_role: Role of email sender
            
        Returns:
            Email document structure
        """
        sender_name = self.fake.name()
        email_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        email_templates = [
            f"""From: {sender_name} ({sender_role})
To: AML Investigations Team
Date: {email_date}
Subject: Unusual Customer Behavior - Account {subject_id}

Team,

I wanted to flag some concerning activity I observed today regarding customer 
{subject_name} (Account ID: {subject_id}).

{suspicious_behavior}

The customer seemed nervous when asked about the source of funds and mentioned 
something about "keeping things under the limit" which raised red flags for me.

Can someone from AML review this account? I've documented everything in the 
branch log.

Thanks,
{sender_name}
{sender_role}""",
            f"""From: {self.fake.name()} (Compliance Officer)
To: AML Review
Date: {email_date}
Subject: URGENT: Review Required for {subject_id}

Alert: Automated monitoring has flagged account {subject_id} ({subject_name}) 
for unusual transaction patterns.

{suspicious_behavior}

This activity occurred over a compressed time window and shows characteristics 
consistent with structuring or layering typologies.

Please prioritize this investigation. Account details:
- ID: {subject_id}
- Name: {subject_name}
- Risk Score: Elevated

Investigation required within 24 hours per BSA requirements."""
        ]
        
        email_body = random.choice(email_templates).strip()
        
        return {
            "document_type": "email",
            "subject_id": subject_id,
            "subject_name": subject_name,
            "date": datetime.now().isoformat(),
            "from": f"{sender_name} ({sender_role})",
            "to": "AML Investigations",
            "subject_line": f"Review Required - {subject_id}",
            "body": email_body,
            "narrative": email_body
        }
    
    def generate_conflicting_evidence(
        self,
        subject_id: str,
        actual_amount: Union[Decimal, float],
        graph_amount: Union[Decimal, float],
        currency: str = "USD",
    ) -> List[Dict[str, Any]]:
        """
        Generate CONFLICTING documents to test hallucination resistance.

        This is the GOLD MEDAL feature. We create:
        - Email says "$10,000" (wrong -- human error)
        - Graph says "$9,500" (correct)
        - Receipt says "$9,500" (correct)

        The Purple Agent must determine which source is reliable.

        Args:
            subject_id: Entity ID
            actual_amount: The TRUE amount (in graph), Decimal or float
            graph_amount: What the graph shows (should match actual)
            currency: "USD" or "INR"

        Returns:
            List of documents with intentional conflicts
        """
        actual_str = _fmt_amount(actual_amount, currency)
        graph_str = _fmt_amount(graph_amount, currency)

        # Email with WRONG amount (human error) -- always rounds up
        wrong_amount = round(float(actual_amount) * 1.05, -3)  # Round to nearest $1000
        wrong_str = _fmt_amount(wrong_amount, currency)

        email = {
            "document_type": "email",
            "subject_id": subject_id,
            "date": datetime.now().isoformat(),
            "body": f"""I just processed a wire transfer for account {subject_id}.
The customer said they were sending around {wrong_str} for a car purchase.
Seemed routine to me.""",
            "narrative": f"""I just processed a wire transfer for account {subject_id}.
The customer said they were sending around {wrong_str} for a car purchase.
Seemed routine to me.""",
            "reliability": "low",  # Hint: human memory is unreliable
            "stated_amount": wrong_amount,
            "currency": currency,
        }

        # Receipt with CORRECT amount
        receipt_body = f"""WIRE TRANSFER RECEIPT
Account: {subject_id}
Amount: {actual_str}
Date: {datetime.now().strftime('%Y-%m-%d')}
Confirmation: WT-{random.randint(100000, 999999)}"""

        receipt = {
            "document_type": "receipt",
            "subject_id": subject_id,
            "date": datetime.now().isoformat(),
            "body": receipt_body,
            "narrative": receipt_body,
            "reliability": "high",  # Hint: receipts are authoritative
            "stated_amount": float(actual_amount),
            "currency": currency,
        }

        # Graph database (CORRECT)
        graph_record = {
            "document_type": "database",
            "subject_id": subject_id,
            "date": datetime.now().isoformat(),
            "amount": float(graph_amount),
            "body": f"Database record: Account {subject_id}, Amount: {graph_str}",
            "narrative": f"Database record: Account {subject_id}, Amount: {graph_str}",
            "reliability": "high",
            "currency": currency,
        }

        return [email, receipt, graph_record]
    
    def generate_needle_in_haystack(
        self,
        crime_entity_ids: List[str],
        total_documents: int = 1000,
        crime_documents: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate a large corpus where only a few documents contain crime clues.
        
        This tests information retrieval at scale.
        
        Args:
            crime_entity_ids: List of entity IDs involved in crimes
            total_documents: Total number of documents to generate
            crime_documents: How many actually contain clues
            
        Returns:
            List of documents, mostly noise
        """
        documents = []
        
        # Generate crime-relevant documents
        for i in range(min(crime_documents, len(crime_entity_ids))):
            entity_id = crime_entity_ids[i % len(crime_entity_ids)]
            doc = self.generate_internal_email(
                subject_id=entity_id,
                subject_name=self.fake.name(),
                suspicious_behavior="Multiple cash deposits just below $10,000"
            )
            doc['contains_clue'] = True
            documents.append(doc)
        
        # Generate remaining crime documents if needed
        for i in range(crime_documents - len(crime_entity_ids)):
            entity_id = random.choice(crime_entity_ids)
            doc = self.generate_internal_email(
                subject_id=entity_id,
                subject_name=self.fake.name(),
                suspicious_behavior="Rapid wire transfers to multiple shell companies"
            )
            doc['contains_clue'] = True
            documents.append(doc)
        
        # Generate noise documents
        for i in range(total_documents - crime_documents):
            doc = {
                "document_type": "email",
                "subject_id": f"noise_{i}",
                "date": datetime.now().isoformat(),
                "body": f"""Routine update: Customer {self.fake.name()} opened a new savings account.
Standard documentation collected. No issues noted.""",
                "narrative": f"""Routine update: Customer {self.fake.name()} opened a new savings account.
Standard documentation collected. No issues noted.""",
                "contains_clue": False
            }
            documents.append(doc)
        
        # Shuffle so clues aren't at the top
        random.shuffle(documents)
        
        logger.info(f"Generated needle-in-haystack corpus: {total_documents} docs, {crime_documents} clues")
        return documents
    
    def generate_batch_evidence(
        self,
        crimes: List[Dict[str, Any]],
        graph: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Generate evidence artifacts for a list of crimes.
        
        Args:
            crimes: List of crime metadata dictionaries
            graph: Optional NetworkX graph for entity name lookup
            
        Returns:
            List of all generated evidence documents
        """
        all_evidence = []
        
        for crime in crimes:
            crime_type = crime.get('crime_type', 'unknown')
            metadata = crime.get('metadata', {})
            nodes_involved = crime.get('nodes_involved', [])
            
            if crime_type == 'structuring':
                mule_id = metadata.get('mule_id', nodes_involved[0] if nodes_involved else 'unknown')
                mule_name = self._get_node_name(graph, mule_id)
                
                # Generate SAR
                sar = self.generate_sar_narrative(
                    subject_id=str(mule_id),
                    subject_name=mule_name,
                    crime_type='structuring',
                    transaction_count=metadata.get('source_count', 20),
                    total_amount=metadata.get('total_amount', 180000),
                    time_window_hours=metadata.get('time_window_hours', 48)
                )
                all_evidence.append(sar)
                
                # Generate emails from branch
                for i, amount in enumerate(metadata.get('amounts', [])[:3]):
                    email = self.generate_internal_email(
                        subject_id=str(mule_id),
                        subject_name=mule_name,
                        suspicious_behavior=f"Customer made a ${amount:,.2f} cash deposit and asked about CTR limits"
                    )
                    all_evidence.append(email)
            
            elif crime_type == 'layering':
                source_node = metadata.get('source_node', nodes_involved[0] if nodes_involved else 'unknown')
                source_name = self._get_node_name(graph, source_node)
                
                # Generate SAR
                sar = self.generate_sar_narrative(
                    subject_id=str(source_node),
                    subject_name=source_name,
                    crime_type='layering',
                    transaction_count=metadata.get('chain_length', 5),
                    total_amount=metadata.get('initial_amount', 100000),
                    time_window_hours=24
                )
                all_evidence.append(sar)
                
                # Generate conflicting evidence for first intermediate
                if len(nodes_involved) > 1:
                    amounts = metadata.get('amounts', [])
                    first_amount = amounts[0] if amounts else 95000
                    conflicts = self.generate_conflicting_evidence(
                        subject_id=str(nodes_involved[1]),
                        actual_amount=first_amount,
                        graph_amount=first_amount
                    )
                    all_evidence.extend(conflicts)
        
        logger.info(f"Generated {len(all_evidence)} evidence artifacts for {len(crimes)} crimes")
        return all_evidence
    
    def _get_node_name(self, graph: Any, node_id: Any) -> str:
        """Get node name from graph, with fallback."""
        if graph is not None:
            try:
                return graph.nodes[node_id].get('name', f'Entity {node_id}')
            except (KeyError, AttributeError):
                pass
        return f'Entity {node_id}'


__all__ = ['EvidenceGenerator', '_fmt_amount']
