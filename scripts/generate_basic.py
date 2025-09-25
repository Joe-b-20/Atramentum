#!/usr/bin/env python3
"""Basic text generation script."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("Basic generation test")
    print("Full generation requires model download and GPU setup")
    print("Run this after setting up your environment and downloading models")
    
    # Test import
    try:
        import torch
        print(f"PyTorch installed: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed. Run: pip install -r requirements.txt")

if __name__ == '__main__':
    main()
