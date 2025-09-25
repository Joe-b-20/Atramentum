#!/usr/bin/env python3
"""Build FAISS index for RAG."""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("Note: Full indexing requires sentence-transformers and faiss.")
    print("For now, creating placeholder index metadata.")
    
    input_path = Path("data/processed/atra_labels.jsonl")
    output_dir = Path("index/faiss/bge_small")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.exists():
        # Copy metadata for now
        with open(input_path) as f:
            data = [json.loads(line) for line in f]
        
        meta_path = output_dir / "meta.jsonl"
        with open(meta_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Metadata saved to {meta_path}")
        print("Full FAISS indexing will be available after installing sentence-transformers")
    else:
        print("No labeled data found. Run auto_label.py first.")

if __name__ == '__main__':
    main()
