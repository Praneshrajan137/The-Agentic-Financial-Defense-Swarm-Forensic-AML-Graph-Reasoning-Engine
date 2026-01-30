"""
SDV Correlation Models
======================
Defines seed datasets for Gaussian Copula training.

These seeds encode the statistical relationships we want to preserve:
1. High risk_score -> High transaction amounts
2. entity_type='bank' -> More frequent transactions
3. International transactions -> Larger amounts

Reference: Follows patterns from .claude/skills/data-generation/scripts/sdv_correlations.py
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_transaction_seed(num_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Create seed dataset for transaction correlation modeling.
    
    Correlation Rules:
    - risk_score (0-1) correlates positively with amount
    - is_international (bool) correlates with 2x higher amounts
    - transaction_type affects amount distribution
    
    Args:
        num_samples: Number of seed samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with correlated transaction features
    """
    np.random.seed(seed)
    
    # Generate risk scores (beta distribution for realistic skew)
    risk_scores = np.random.beta(2, 5, num_samples)
    
    # Base amounts follow log-normal (realistic for financial data)
    base_amounts = np.random.lognormal(mean=8.0, sigma=1.5, size=num_samples)
    
    # Apply risk multiplier (high risk = larger amounts)
    risk_multiplier = 1 + (risk_scores * 3)  # 1x to 4x multiplier
    amounts = base_amounts * risk_multiplier
    
    # International transactions are 2x larger on average
    is_international = np.random.choice([False, True], num_samples, p=[0.7, 0.3])
    amounts = amounts * (1 + is_international.astype(int))
    
    # Transaction types aligned with existing schema
    # ['wire', 'ach', 'cash', 'internal'] from graph_generator.py
    transaction_types = np.random.choice(
        ['wire', 'ach', 'cash', 'internal'],
        num_samples,
        p=[0.3, 0.4, 0.1, 0.2]
    )
    
    # Transaction frequency correlates with amount (log relationship)
    frequency = np.log1p(amounts) * np.random.uniform(0.5, 1.5, num_samples)
    
    df = pd.DataFrame({
        'amount': np.round(amounts, 2),
        'risk_score': np.round(risk_scores, 2),
        'is_international': is_international,
        'transaction_type': transaction_types,
        'frequency': np.round(frequency, 2)
    })
    # Ensure is_international is boolean dtype (required by SDV)
    df['is_international'] = df['is_international'].astype(bool)
    return df


def create_entity_seed(num_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Create seed dataset for entity attribute correlation.
    
    Correlation Rules:
    - entity_type='bank' has higher verification rates
    - Higher risk_score correlates with newer accounts
    - Companies have higher transaction volumes than persons
    
    Args:
        num_samples: Number of seed samples
        seed: Random seed
        
    Returns:
        DataFrame with correlated entity features
    """
    np.random.seed(seed)
    
    # Entity type distribution
    entity_types = np.random.choice(
        ['person', 'company', 'bank'],
        num_samples,
        p=[0.7, 0.25, 0.05]
    )
    
    # Risk scores (banks are lower risk)
    risk_scores = np.where(
        entity_types == 'bank',
        np.random.beta(1, 10, num_samples),  # Low risk
        np.where(
            entity_types == 'company',
            np.random.beta(2, 5, num_samples),  # Medium risk
            np.random.beta(3, 3, num_samples)   # Higher risk for persons
        )
    )
    
    # Verification status (banks always verified)
    verification_status = np.where(
        entity_types == 'bank',
        'verified',
        np.random.choice(['verified', 'pending', 'failed'], num_samples, p=[0.85, 0.10, 0.05])
    )
    
    # Transaction volume (companies > persons)
    base_volume = np.random.lognormal(mean=3.0, sigma=1.0, size=num_samples)
    volume_multiplier = np.where(
        entity_types == 'company', 3.0,
        np.where(entity_types == 'bank', 10.0, 1.0)
    )
    monthly_tx_volume = base_volume * volume_multiplier
    
    return pd.DataFrame({
        'entity_type': entity_types,
        'risk_score': np.round(risk_scores, 2),
        'verification_status': verification_status,
        'monthly_tx_volume': np.round(monthly_tx_volume, 0)
    })


# Global synthesizers (lazy-loaded)
_transaction_synthesizer = None
_entity_synthesizer = None


def get_transaction_synthesizer(retrain: bool = False):
    """
    Get or create transaction synthesizer.
    
    Args:
        retrain: Force retraining of the model
        
    Returns:
        Fitted GaussianCopulaSynthesizer for transactions
    """
    global _transaction_synthesizer
    
    if _transaction_synthesizer is None or retrain:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
        
        # Create seed data
        seed_df = create_transaction_seed()
        
        # Define metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(seed_df)
        
        # Override specific column types if needed
        metadata.update_column(
            column_name='is_international',
            sdtype='boolean'
        )
        metadata.update_column(
            column_name='transaction_type',
            sdtype='categorical'
        )
        
        # Create and fit synthesizer
        synthesizer = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            numerical_distributions={
                'amount': 'truncnorm',
                'risk_score': 'beta'
            }
        )
        
        logger.info("Training Transaction Gaussian Copula...")
        synthesizer.fit(seed_df)
        logger.info("Transaction synthesizer ready")
        
        _transaction_synthesizer = synthesizer
    
    return _transaction_synthesizer


def get_entity_synthesizer(retrain: bool = False):
    """
    Get or create entity synthesizer.
    
    Args:
        retrain: Force retraining of the model
        
    Returns:
        Fitted GaussianCopulaSynthesizer for entities
    """
    global _entity_synthesizer
    
    if _entity_synthesizer is None or retrain:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
        
        seed_df = create_entity_seed()
        
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(seed_df)
        
        synthesizer = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True
        )
        
        logger.info("Training Entity Gaussian Copula...")
        synthesizer.fit(seed_df)
        logger.info("Entity synthesizer ready")
        
        _entity_synthesizer = synthesizer
    
    return _entity_synthesizer


__all__ = [
    'create_transaction_seed',
    'create_entity_seed',
    'get_transaction_synthesizer',
    'get_entity_synthesizer'
]
