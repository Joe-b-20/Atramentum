#!/usr/bin/env python3
"""Auto-label journal entries for RAG."""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

def main():
    input_path = Path("data/processed/atra_sft.jsonl")
    output_path = Path("data/processed/atra_labels.jsonl")
    
    if not input_path.exists():
        print(f"Input file {input_path} not found. Run make_dataset.py first.")
        return
    
    print("Creating labels for RAG indexing...")
    
    # Read SFT data
    with open(input_path) as f:
        data = [json.loads(line) for line in f]
    
    # Create labeled entries
    labeled_data = []
    for item in data:
        assistant_text = item['messages'][2]['content']
        date = item['meta']['date']
        
        # Simple labeling
        labeled_item = {
            'id': hashlib.md5(f"{date}{assistant_text[:64]}".encode()).hexdigest(),
            'date': date,
            'text': assistant_text,
            'type': item['meta']['type']
        }
        labeled_data.append(labeled_item)
    
    # Save
    with open(output_path, 'w') as f:
        for item in labeled_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(labeled_data)} labeled entries to {output_path}")

if __name__ == '__main__':
    main()
