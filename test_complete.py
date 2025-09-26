"""Complete test of Atramentum setup."""

import os
import sys
import json
from pathlib import Path

def check_setup():
    """Check if everything is set up correctly."""
    
    print("=== Atramentum Setup Check ===\n")
    
    # 1. Check directory structure
    required_dirs = ['data', 'configs', 'scripts', 'src/atramentum']
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory {dir_path} exists")
        else:
            print(f"✗ Missing directory: {dir_path}")
    
    # 2. Check for data
    if Path('data/Journal.txt').exists():
        print("✓ Journal data found")
    else:
        print("✗ No journal data. Add data/Journal.txt")
    
    # 3. Check processed data
    if Path('data/processed/atra_sft.jsonl').exists():
        with open('data/processed/atra_sft.jsonl') as f:
            count = sum(1 for _ in f)
        print(f"✓ Processed dataset with {count} examples")
    else:
        print("✗ No processed data. Run: python scripts/make_dataset.py")
    
    # 4. Check Python packages
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("! CUDA not available (CPU mode only)")
    except ImportError:
        print("✗ PyTorch not installed")
    
    try:
        import transformers
        print(f"✓ Transformers installed: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
    
    # 5. Check environment
    if Path('.env').exists():
        print("✓ Environment file exists")
        # Load env vars
        with open('.env') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        
        if os.environ.get('HF_TOKEN'):
            print("✓ HF_TOKEN is set")
        else:
            print("! HF_TOKEN not set (needed for some models)")
    else:
        print("✗ No .env file")
    
    print("\n=== Setup Check Complete ===")
    print("\nNext steps:")
    print("1. If data not processed: python scripts/make_dataset.py")
    print("2. To start API: python scripts/serve_simple.py")
    print("3. To train model: python scripts/train_sft.py --config configs/sft_llama3.yaml")

if __name__ == "__main__":
    check_setup()