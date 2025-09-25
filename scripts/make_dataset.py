#!/usr/bin/env python3
"""Create SFT dataset from journal export."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.atramentum.data.formatter import JournalFormatter

def main():
    parser = argparse.ArgumentParser(description="Format journal entries for training.")
    parser.add_argument('--config', type=str, default='configs/data_formatter.yaml')
    
    args = parser.parse_args()
    
    print("Starting dataset creation...")
    formatter = JournalFormatter(args.config)
    output_path = formatter.process_files()
    print(f"Dataset created: {output_path}")

if __name__ == '__main__':
    main()
