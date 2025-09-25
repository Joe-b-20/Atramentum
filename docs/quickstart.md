# Quick Start Guide

*From zero to existential crisis in 10 minutes.*

## Prerequisites

- Python 3.10+
- CUDA-capable GPU
- Google Doc export of your journal
- Acceptance of computational suffering

## Installation
# Clone and setup
git clone https://github.com/Joe-b-20/Atramentum.git
cd atramentum

# Build environment
make build

# Process your journal export
make data

# Train the model (RunPod recommended)
python scripts/train_sft.py --config configs/sft_llama3.yaml

# Serve the API
make serve