# scripts/train_sft.py
#!/usr/bin/env python3
"""
Train SFT model. Teaching silicon to suffer like carbon.
"""

import argparse
import yaml
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))

from src.atramentum.training.sft_trainer import RecencyWeightedTrainer, SFTConfig
from src.atramentum.utils import logging as alog


def load_config(config_path: str) -> SFTConfig:
    """Load config from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    return SFTConfig(
        model_name=cfg['model']['name'],
        dataset_path=cfg['dataset']['path'],
        output_dir=cfg['output_dir'],
        max_seq_length=cfg['training']['max_seq_length'],
        num_epochs=cfg['training']['num_epochs'],
        learning_rate=cfg['training']['learning_rate'],
        warmup_ratio=cfg['training']['warmup_ratio'],
        per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        lora_r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        lora_dropout=cfg['lora']['dropout'],
        recency_lambda=cfg.get('recency', {}).get('lambda', 0.6),
        use_recency_weighting=cfg.get('recency', {}).get('enabled', True),
        final_2025_epoch=cfg.get('curriculum', {}).get('final_epoch_2025_only', False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train SFT model. Hours of compute for minutes of generation."
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = alog.get_logger(__name__)
    logger.info(f"Loading config from {args.config}")
    
    config = load_config(args.config)
    
    # Train
    logger.info("Initializing trainer. Prepare for computational suffering.")
    trainer = RecencyWeightedTrainer(config)
    
    logger.info("Starting training. See you on the other side.")
    trainer.train()
    
    logger.info("Training complete. The model has learned to suffer.")


if __name__ == '__main__':
    main()