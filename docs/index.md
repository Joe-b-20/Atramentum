# docs/index.md
# Atramentum Documentation

*Where technical documentation meets existential resignation.*

## Overview

Atramentum is a self-hosted language model specialized in replicating personal journal entries. It combines:

- **QLoRA fine-tuning** for efficient training on consumer hardware
- **Recency-weighted sampling** to prioritize recent entries
- **Local RAG retrieval** for "memory" augmentation
- **DPO preference learning** to maintain appropriate tone

The result: an AI that writes like you do at 3 AM, complete with date-first formatting and darkly sarcastic observations.

## Core Concepts

### Date-First Philosophy
Every entry begins with `MM/DD/YYYY —` because even chaos needs timestamps.

### Recency Weighting
Recent entries receive exponentially more weight during training (λ=0.6), because yesterday's problems matter more than last year's.

### Archive Memory
RAG-powered retrieval provides "distorted recollections" - snippets from past entries that may or may not be relevant.

### Voice Preservation
The model learns your specific flavor of exhaustion through:
- Rewrite pairs (70%): Teaching style transfer
- Seed generation (30%): Teaching continuation
- DPO hardening: Preventing drift toward toxic positivity

## System Requirements

- **Training**: 24GB+ VRAM (RTX 3090/4090 or better)
- **Inference**: 8GB+ VRAM (RTX 3060 or better)
- **Storage**: ~50GB for models and indices
- **Patience**: Infinite

## Navigation

- [Quick Start](quickstart.md) - Get running in minutes
- [Datasets](datasets.md) - Data format and preparation
- [Training](training.md) - QLoRA and DPO configuration
- [Evaluation](evaluation.md) - Measuring despair accurately
- [Deployment](deployment.md) - Serving your digital doppelganger
- [Style Guide](style.md) - Writing with appropriate darkness

## Philosophy

Atramentum operates on the principle that journaling AI should reflect authentic human expression, not sanitized positivity. It's trained to:

1. Preserve your voice, including its flaws
2. Maintain chronological grounding
3. Generate with appropriate cynicism
4. Remember imperfectly, like humans do

## Support

For issues, open a GitHub issue and wait. For security concerns, email joebachir20@gmail.com.

*"The unexamined life is not worth living, but the over-examined life needs good tooling."*