# src/atramentum/data/validators.py
"""
Data validators - because even despair needs structure.
"""

import re
from typing import Dict, List, Optional
from datetime import datetime


def validate_entry(entry: Dict) -> bool:
    """Validate a journal entry has required fields."""
    required = ['date', 'text']
    return all(field in entry for field in required)


def validate_date_format(date_str: str) -> bool:
    """Check if date matches expected format MM/DD/YYYY."""
    pattern = re.compile(r'^\d{2}/\d{2}/\d{4}$')
    if not pattern.match(date_str):
        return False
    
    # Validate it's a real date
    try:
        datetime.strptime(date_str, '%m/%d/%Y')
        return True
    except ValueError:
        return False


def validate_messages(messages: List[Dict]) -> bool:
    """Validate message format for training."""
    if len(messages) != 3:
        return False
    
    roles = [msg.get('role') for msg in messages]
    expected = ['system', 'user', 'assistant']
    
    return roles == expected


def validate_dataset_entry(entry: Dict) -> bool:
    """Validate a complete dataset entry."""
    if 'messages' not in entry:
        return False
    
    if not validate_messages(entry['messages']):
        return False
    
    # Check assistant message starts with date
    assistant_msg = entry['messages'][2]['content']
    date_pattern = re.compile(r'^\d{2}/\d{2}/\d{4} —')
    
    return bool(date_pattern.match(assistant_msg))