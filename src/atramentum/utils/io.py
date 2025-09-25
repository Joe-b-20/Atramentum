# src/atramentum/utils/io.py
"""
I/O utilities. Reading and writing the void.
"""

import json
from pathlib import Path
from typing import List, Dict, Union, Generator


def read_jsonl(filepath: Union[str, Path]) -> List[Dict]:
    """Read JSONL file. Each line a small tragedy."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(filepath: Union[str, Path], data: List[Dict]) -> None:
    """Write JSONL file. Commit thoughts to disk."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def stream_jsonl(filepath: Union[str, Path]) -> Generator[Dict, None, None]:
    """Stream JSONL file. For when loading everything would break you."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_json(filepath: Union[str, Path], data: Dict) -> None:
    """Save JSON file with pretty printing. Make the pain readable."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)