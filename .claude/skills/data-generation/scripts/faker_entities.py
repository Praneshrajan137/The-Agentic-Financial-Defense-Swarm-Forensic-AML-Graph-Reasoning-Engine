#!/usr/bin/env python3
"""
Generate Synthetic Entity Profiles using Faker
===============================================
Creates realistic entity profiles (people, companies, banks)
with localized attributes for graph nodes.

Usage:
    python faker_entities.py --count 1000 --output entities.json
    python faker_entities.py --count 1000 --locales en_US en_GB en_IN --output entities.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from faker import Faker


# Default configuration
DEFAULT_COUNT = 1000
DEFAULT_LOCALES = ['en_US', 'en_GB', 'en_IN']
ENTITY_TYPES = ['person', 'company', 'bank']
ENTITY_WEIGHTS = [0.7, 0.25, 0.05]  # 70% people, 25% companies, 5% banks


def generate_entity_profile(
    fake: Faker,
    entity_id: str,
    entity_type: Optional[str] = None
) -> Dict:
    """
    Generate a single entity profile.
    
    Args:
        fake: Faker instance
        entity_id: Unique identifier for the entity
        entity_type: Type of entity (auto-selected if None)
    
    Returns:
        Dictionary with entity attributes
    """
    if entity_type is None:
        entity_type = random.choices(ENTITY_TYPES, weights=ENTITY_WEIGHTS)[0]
    
    profile = {
        'id': entity_id,
        'entity_type': entity_type,
        'locale': fake.locales[0] if hasattr(fake, 'locales') else 'en_US'
    }
    
    if entity_type == 'person':
        profile['name'] = fake.name()
        profile['company'] = None
        profile['email'] = fake.email()
        profile['phone'] = fake.phone_number()
    elif entity_type == 'company':
        profile['name'] = fake.company()
        profile['company'] = profile['name']
        profile['email'] = fake.company_email()
        profile['phone'] = fake.phone_number()
    else:  # bank
        profile['name'] = fake.company() + " Bank"
        profile['company'] = profile['name']
        profile['email'] = f"info@{fake.domain_name()}"
        profile['phone'] = fake.phone_number()
    
    # Common attributes
    profile['address'] = fake.address().replace('\n', ', ')
    profile['country'] = fake.country_code()
    profile['swift'] = fake.swift()
    profile['iban'] = fake.iban()
    profile['account_number'] = fake.bban()
    profile['risk_score'] = round(random.uniform(0.0, 1.0), 2)
    profile['verification_status'] = random.choices(
        ['verified', 'pending', 'failed'],
        weights=[0.85, 0.10, 0.05]
    )[0]
    profile['created_at'] = fake.date_time_between(
        start_date='-2y',
        end_date='now'
    ).isoformat()
    
    return profile


def generate_entity_batch(
    count: int,
    locales: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Generate batch of entity profiles.
    
    Args:
        count: Number of entities to generate
        locales: List of Faker locales
        seed: Random seed for reproducibility
    
    Returns:
        List of entity dictionaries
    """
    if locales is None:
        locales = DEFAULT_LOCALES
    
    # Set seeds
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    
    # Initialize Faker with locales
    fake = Faker(locales)
    
    entities = []
    for i in range(count):
        entity_id = f"entity_{i}"
        profile = generate_entity_profile(fake, entity_id)
        entities.append(profile)
    
    return entities


def generate_person(fake: Faker, entity_id: str) -> Dict:
    """
    Generate a person entity.
    
    Args:
        fake: Faker instance
        entity_id: Unique identifier
    
    Returns:
        Person entity dictionary
    """
    return generate_entity_profile(fake, entity_id, 'person')


def generate_company(fake: Faker, entity_id: str) -> Dict:
    """
    Generate a company entity.
    
    Args:
        fake: Faker instance
        entity_id: Unique identifier
    
    Returns:
        Company entity dictionary
    """
    return generate_entity_profile(fake, entity_id, 'company')


def generate_bank(fake: Faker, entity_id: str) -> Dict:
    """
    Generate a bank entity.
    
    Args:
        fake: Faker instance
        entity_id: Unique identifier
    
    Returns:
        Bank entity dictionary
    """
    return generate_entity_profile(fake, entity_id, 'bank')


def save_entities(entities: List[Dict], output_path: str) -> None:
    """
    Save entities to JSON file.
    
    Args:
        entities: List of entity dictionaries
        output_path: Path to output file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(entities, f, indent=2)
    
    print(f"Saved {len(entities)} entities to {path}")
    
    # Print summary
    type_counts = {}
    locale_counts = {}
    for entity in entities:
        etype = entity['entity_type']
        locale = entity.get('locale', 'unknown')
        type_counts[etype] = type_counts.get(etype, 0) + 1
        locale_counts[locale] = locale_counts.get(locale, 0) + 1
    
    print(f"\nEntity Types:")
    for etype, count in sorted(type_counts.items()):
        print(f"  {etype}: {count} ({count/len(entities)*100:.1f}%)")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic entity profiles using Faker'
    )
    parser.add_argument(
        '--count', '-n',
        type=int,
        default=DEFAULT_COUNT,
        help=f'Number of entities to generate (default: {DEFAULT_COUNT})'
    )
    parser.add_argument(
        '--locales', '-l',
        nargs='+',
        default=DEFAULT_LOCALES,
        help=f'Faker locales (default: {DEFAULT_LOCALES})'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    print(f"Generating {args.count} entities with locales: {args.locales}")
    
    entities = generate_entity_batch(
        count=args.count,
        locales=args.locales,
        seed=args.seed
    )
    
    save_entities(entities, args.output)


if __name__ == '__main__':
    main()
