#!/usr/bin/env python3
"""
Generate Correlated Synthetic Data using SDV
=============================================
Uses Gaussian Copula and other models to generate
statistically correlated transaction data.

Usage:
    python sdv_correlations.py --sample transactions.csv --count 10000 --output synthetic.csv
    python sdv_correlations.py --sample transactions.csv --count 10000 --model ctgan --output synthetic.csv
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


# Available models
AVAILABLE_MODELS = ['gaussian_copula', 'ctgan', 'copulagan']
DEFAULT_MODEL = 'gaussian_copula'


def generate_correlated_transactions(
    sample_df: pd.DataFrame,
    num_samples: int,
    model_type: str = DEFAULT_MODEL,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic transactions preserving statistical correlations.
    
    Args:
        sample_df: Real transaction sample DataFrame
        num_samples: Number of synthetic samples to generate
        model_type: SDV model to use
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic transactions
    """
    try:
        from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
        from sdv.metadata import SingleTableMetadata
    except ImportError:
        raise ImportError(
            "SDV not installed. Install with: pip install sdv"
        )
    
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(sample_df)
    
    # Select model
    if model_type == 'gaussian_copula':
        synthesizer = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            numerical_distributions={
                'amount': 'truncnorm'
            }
        )
    elif model_type == 'ctgan':
        synthesizer = CTGANSynthesizer(
            metadata,
            enforce_min_max_values=True,
            epochs=300
        )
    elif model_type == 'copulagan':
        synthesizer = CopulaGANSynthesizer(
            metadata,
            enforce_min_max_values=True,
            epochs=300
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit model
    print(f"Fitting {model_type} model...")
    synthesizer.fit(sample_df)
    
    # Generate synthetic data
    print(f"Generating {num_samples} samples...")
    synthetic_df = synthesizer.sample(num_rows=num_samples)
    
    return synthetic_df


def create_sample_transaction_data(
    num_samples: int = 1000,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Create sample transaction data for model training.
    
    Args:
        num_samples: Number of samples to create
        seed: Random seed
    
    Returns:
        DataFrame with sample transactions
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate correlated data
    # Higher risk accounts have larger transactions on average
    risk_scores = np.random.beta(2, 5, num_samples)
    
    # Amount correlates with risk (higher risk = larger amounts)
    base_amounts = np.random.lognormal(mean=8.0, sigma=1.5, size=num_samples)
    amounts = base_amounts * (1 + risk_scores * 2)  # Risk multiplier
    
    # Transaction frequency correlates with amount
    frequencies = np.log1p(amounts) * np.random.uniform(0.5, 1.5, num_samples)
    
    # Generate timestamps over past year
    base_date = datetime.now()
    timestamps = [
        base_date - timedelta(days=np.random.randint(0, 365))
        for _ in range(num_samples)
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'amount': np.round(amounts, 2),
        'risk_score': np.round(risk_scores, 2),
        'frequency': np.round(frequencies, 2),
        'is_international': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'transaction_type': np.random.choice(
            ['wire', 'ach', 'cash', 'internal'],
            num_samples,
            p=[0.3, 0.4, 0.1, 0.2]
        ),
        'timestamp': timestamps
    })
    
    return df


def validate_synthetic_data(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame
) -> dict:
    """
    Validate that synthetic data preserves statistical properties.
    
    Args:
        original_df: Original sample data
        synthetic_df: Generated synthetic data
    
    Returns:
        Dictionary with validation metrics
    """
    import numpy as np
    
    metrics = {}
    
    # Compare numerical columns
    numerical_cols = original_df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if col in synthetic_df.columns:
            metrics[col] = {
                'original_mean': float(original_df[col].mean()),
                'synthetic_mean': float(synthetic_df[col].mean()),
                'original_std': float(original_df[col].std()),
                'synthetic_std': float(synthetic_df[col].std()),
                'mean_diff_pct': abs(
                    (original_df[col].mean() - synthetic_df[col].mean()) 
                    / original_df[col].mean() * 100
                ) if original_df[col].mean() != 0 else 0
            }
    
    # Overall quality score (lower is better)
    total_diff = sum(
        m['mean_diff_pct'] for m in metrics.values() 
        if 'mean_diff_pct' in m
    )
    metrics['quality_score'] = total_diff / len(metrics) if metrics else 0
    
    return metrics


def save_synthetic_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save synthetic data to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} synthetic records to {path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate correlated synthetic transaction data using SDV'
    )
    parser.add_argument(
        '--sample', '-s',
        type=str,
        default=None,
        help='Input sample CSV file (creates sample data if not provided)'
    )
    parser.add_argument(
        '--count', '-n',
        type=int,
        default=10000,
        help='Number of synthetic samples to generate (default: 10000)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=AVAILABLE_MODELS,
        default=DEFAULT_MODEL,
        help=f'SDV model to use (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate synthetic data quality'
    )
    
    args = parser.parse_args()
    
    # Load or create sample data
    if args.sample:
        print(f"Loading sample data from {args.sample}...")
        sample_df = pd.read_csv(args.sample)
    else:
        print("Creating sample transaction data...")
        sample_df = create_sample_transaction_data(
            num_samples=1000,
            seed=args.seed
        )
        # Save sample for reference
        sample_path = Path(args.output).parent / 'sample_transactions.csv'
        sample_df.to_csv(sample_path, index=False)
        print(f"Sample data saved to {sample_path}")
    
    print(f"Sample data shape: {sample_df.shape}")
    
    # Generate synthetic data
    try:
        synthetic_df = generate_correlated_transactions(
            sample_df,
            num_samples=args.count,
            model_type=args.model,
            seed=args.seed
        )
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nSDV is required for this script. Install with:")
        print("  pip install sdv")
        print("\nAlternatively, the sample data has been created and can be used directly.")
        return
    
    # Validate if requested
    if args.validate:
        print("\nValidating synthetic data...")
        metrics = validate_synthetic_data(sample_df, synthetic_df)
        print(f"Quality Score: {metrics['quality_score']:.2f}% average deviation")
        for col, m in metrics.items():
            if col != 'quality_score' and isinstance(m, dict):
                print(f"  {col}: {m['mean_diff_pct']:.2f}% mean difference")
    
    # Save results
    save_synthetic_data(synthetic_df, args.output)


if __name__ == '__main__':
    main()
