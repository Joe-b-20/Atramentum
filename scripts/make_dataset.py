# scripts/make_dataset.py
#!/usr/bin/env python3
"""
Create SFT dataset from journal export. Transform the raw into the refined.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.atramentum.data.formatter import JournalFormatter
from src.atramentum.utils import logging as alog


def main():
    parser = argparse.ArgumentParser(
        description="Format journal entries for training. Making order from chaos."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_formatter.yaml',
        help='Path to formatter config'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = alog.get_logger(__name__)
    logger.info("Starting dataset creation. Parsing the unparseable...")
    
    # Process
    formatter = JournalFormatter(args.config)
    output_path = formatter.process_files()
    
    logger.info(f"Dataset created: {output_path}")
    logger.info("Ready for training. May the models have mercy.")


if __name__ == '__main__':
    main()