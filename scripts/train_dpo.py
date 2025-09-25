# scripts/train_dpo.py
#!/usr/bin/env python3
"""
Train DPO model. Teaching preferences through binary choices.
"""

import argparse
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.atramentum.training.dpo_trainer import PreferenceTrainer, DPOTrainingConfig
from src.atramentum.utils import logging as alog


def load_config(config_path: str) -> DPOTrainingConfig:
    """Load DPO config from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    return DPOTrainingConfig(
        model_path=cfg['model']['name'],
        dataset_path=cfg['dataset']['path'],
        output_dir=cfg['output_dir'],
        beta=cfg['dpo']['beta'],
        learning_rate=cfg['training']['learning_rate'],
        num_epochs=cfg['training']['num_epochs'],
        per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        lora_r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        max_seq_length=cfg['training']['max_seq_length']
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train DPO model. Choose your suffering wisely."
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to DPO config'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = alog.get_logger(__name__)
    logger.info(f"Loading DPO config from {args.config}")
    
    config = load_config(args.config)
    
    # Train
    logger.info("Initializing preference trainer.")
    trainer = PreferenceTrainer(config)
    
    logger.info("Starting DPO training. Teaching the model to prefer darkness.")
    trainer.train()
    
    logger.info("DPO training complete. Preferences aligned with the void.")


if __name__ == '__main__':
    main()