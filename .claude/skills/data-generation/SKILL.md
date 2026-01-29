---
name: data-generation
description: Expert in synthetic data generation using Faker and SDV for realistic financial profiles
version: 1.0.0
---

# Data Generation Skill

This skill handles realistic synthetic entity and transaction generation for the Green Financial Crime Agent.

## Core Technologies

### Faker
Python library for generating realistic fake data. Supports multiple locales for international diversity.

**Supported Locales**:
- `en_US` - United States
- `en_GB` - United Kingdom  
- `en_IN` - India

**Key Providers**:
- `fake.name()` - Person names
- `fake.company()` - Company names
- `fake.address()` - Full addresses
- `fake.swift()` - SWIFT/BIC codes
- `fake.iban()` - IBAN numbers
- `fake.bban()` - Basic Bank Account Numbers

### SDV (Synthetic Data Vault)
Library for generating statistically correlated synthetic data using machine learning.

**Key Models**:
- `GaussianCopula` - Preserves statistical relationships
- `CTGAN` - Deep learning for complex distributions
- `CopulaGAN` - Combines Copula and GAN approaches

## Tools

### faker_entities.py

Generate realistic entity profiles for graph nodes.

**Location**: `.claude/skills/data-generation/scripts/faker_entities.py`

**Usage**:
```bash
# Generate 1000 entities with US locale
python faker_entities.py --count 1000 --output entities.json

# Generate with multiple locales
python faker_entities.py --count 1000 --locales en_US en_GB en_IN --output entities.json
```

**Entity Schema**:
```python
{
    "id": str,              # Unique identifier
    "entity_type": str,     # "person" | "company" | "bank"
    "name": str,            # Generated name
    "company": str,         # Company name (if applicable)
    "address": str,         # Full address
    "phone": str,           # Phone number
    "email": str,           # Email address
    "swift": str,           # SWIFT/BIC code
    "iban": str,            # IBAN (European)
    "account_number": str,  # Basic bank account number
    "country": str,         # ISO country code
    "locale": str           # Source locale
}
```

### sdv_correlations.py

Generate statistically correlated transaction patterns.

**Location**: `.claude/skills/data-generation/scripts/sdv_correlations.py`

**Usage**:
```bash
# Generate correlated transactions from sample
python sdv_correlations.py --sample transactions.csv --count 10000 --output synthetic.csv

# With custom model
python sdv_correlations.py --sample transactions.csv --count 10000 --model ctgan --output synthetic.csv
```

**Correlation Patterns**:
- Amount-Risk: Higher risk entities have larger transactions
- Time-Volume: Transactions follow realistic business patterns
- Geographic: Cross-border transactions match country pairs

## Entity Types

### Person
Individual account holders with personal attributes.

```python
{
    "entity_type": "person",
    "name": "John Smith",
    "address": "123 Main St, London, UK",
    "email": "john.smith@email.com",
    "phone": "+44 20 7123 4567"
}
```

### Company
Business entities with corporate attributes.

```python
{
    "entity_type": "company",
    "name": "Acme Corporation Ltd",
    "company": "Acme Corporation Ltd",
    "address": "100 Business Park, Manchester, UK",
    "swift": "ACMEGB2L"
}
```

### Bank
Financial institutions with banking attributes.

```python
{
    "entity_type": "bank",
    "name": "First National Bank",
    "company": "First National Bank",
    "swift": "FNBKUS33",
    "iban": "US82FNBK12345698765432"
}
```

## Transaction Attributes

### Amount Distribution
```python
# Legitimate transactions follow log-normal distribution
amount = np.random.lognormal(mean=8.5, sigma=1.5)
# Typical range: $100 - $50,000

# Crime transactions are constrained
structuring_amount = random.uniform(9000, 9800)  # Below CTR
layering_amount = previous_amount * (1 - decay)  # With decay
```

### Time Patterns
```python
# Business hours more common
hour_weights = [0.5 if 9 <= h <= 17 else 0.2 for h in range(24)]

# Weekdays more common
day_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.2]  # Mon-Sun
```

### Transaction Types
- `wire` - Wire transfers (international)
- `ach` - ACH transfers (domestic US)
- `cash` - Cash deposits/withdrawals
- `internal` - Internal bank transfers

## SDV Model Configuration

### GaussianCopula (Recommended)
```python
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(sample_df)

synthesizer = GaussianCopulaSynthesizer(
    metadata,
    enforce_min_max_values=True,
    enforce_rounding=True,
    numerical_distributions={
        'amount': 'truncnorm',
        'risk_score': 'beta'
    }
)

synthesizer.fit(sample_df)
synthetic_df = synthesizer.sample(num_rows=10000)
```

### Preserving Correlations
```python
# Key correlations to maintain:
# 1. High-risk entities have larger average transactions
# 2. Companies have higher transaction volumes than individuals
# 3. Cross-border transactions are larger on average
# 4. Transaction frequency correlates with account age
```

## Best Practices

1. **Seed Everything**: Use consistent seeds for reproducibility
   ```python
   Faker.seed(42)
   random.seed(42)
   np.random.seed(42)
   ```

2. **Locale Diversity**: Mix locales for realistic international data
   ```python
   locales = ['en_US', 'en_GB', 'en_IN']
   fake = Faker(locales)
   ```

3. **Validate Outputs**: Check statistical properties after generation
   ```python
   assert synthetic_df['amount'].mean() > 0
   assert synthetic_df['amount'].std() > 100
   ```

4. **No PII**: Never use real data, always synthetic
   ```python
   # GOOD
   name = fake.name()
   
   # BAD - Never do this
   name = real_customer_database.get_name()
   ```

## Usage Examples

### Generate Entity Batch
```python
from faker import Faker

def generate_entities(count: int, locales: list) -> list:
    fake = Faker(locales)
    Faker.seed(42)
    
    entities = []
    for i in range(count):
        entity_type = random.choices(
            ['person', 'company', 'bank'],
            weights=[0.7, 0.25, 0.05]
        )[0]
        
        entity = {
            'id': f'entity_{i}',
            'entity_type': entity_type,
            'name': fake.name() if entity_type == 'person' else fake.company(),
            'address': fake.address(),
            'country': fake.country_code(),
            'swift': fake.swift(),
            'risk_score': round(random.uniform(0, 1), 2)
        }
        entities.append(entity)
    
    return entities
```

### Generate Correlated Transactions
```python
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer

def generate_transactions(sample_path: str, count: int) -> pd.DataFrame:
    # Load sample data
    sample_df = pd.read_csv(sample_path)
    
    # Create and fit model
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(sample_df)
    
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(sample_df)
    
    # Generate synthetic data
    synthetic_df = synthesizer.sample(num_rows=count)
    
    return synthetic_df
```
