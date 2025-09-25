# src/atramentum/training/sft_trainer.py
"""
SFT trainer - teaching models to channel your specific flavor of despair.
"""

import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer
from torch.utils.data import WeightedRandomSampler

from ..utils import io as aio
from ..utils import logging as alog


@dataclass
class SFTConfig:
    """Configuration for SFT training. Like therapy settings, but cheaper."""
    model_name: str
    dataset_path: str
    output_dir: str
    max_seq_length: int = 4096
    num_epochs: int = 3
    learning_rate: float = 1.5e-4
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 32
    lora_r: int = 128
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    recency_lambda: float = 0.6
    use_recency_weighting: bool = True
    final_2025_epoch: bool = False


class RecencyWeightedTrainer:
    """Trainer with recency weighting. Recent trauma gets more attention."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.logger = alog.get_logger(__name__)
        
    def load_dataset(self) -> Dataset:
        """Load and prepare dataset with recency weights."""
        data = aio.read_jsonl(self.config.dataset_path)
        
        # Calculate weights if enabled
        if self.config.use_recency_weighting:
            weights = self._calculate_weights(data)
        else:
            weights = [1.0] * len(data)
        
        # Create HF dataset
        dataset = Dataset.from_list(data)
        dataset = dataset.add_column("weight", weights)
        
        return dataset
    
    def _calculate_weights(self, data: List[Dict]) -> List[float]:
        """Calculate recency weights. The past fades, but never disappears."""
        today = datetime.now()
        weights = []
        
        for entry in data:
            date_str = entry['meta']['date']
            entry_date = datetime.strptime(date_str, '%m/%d/%Y')
            
            # Calculate age in years
            age_years = (today - entry_date).days / 365.0
            
            # Exponential decay
            weight = np.exp(-self.config.recency_lambda * age_years)
            weights.append(weight)
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total * len(weights) for w in weights]
        
        return weights
    
    def setup_model(self):
        """Setup model with QLoRA. Maximum suffering, minimum VRAM."""
        # BitsAndBytes config for 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=os.environ.get('HF_TOKEN')
        )
        
        # Prepare for training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_auth_token=os.environ.get('HF_TOKEN')
        )
        
        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def train(self):
        """Execute training. Hours of compute for minutes of enjoyment."""
        # Setup
        model, tokenizer = self.setup_model()
        dataset = self.load_dataset()
        
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.05, seed=42)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=True,
            gradient_checkpointing=True,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="messages",  # Will be formatted by trainer
            packing=False,
        )
        
        # Train
        self.logger.info("Starting training. This will hurt (your GPU).")
        trainer.train()
        
        # Optional final 2025-only epoch
        if self.config.final_2025_epoch:
            self._train_final_epoch(trainer, dataset['train'])
        
        # Save
        trainer.save_model()
        self.logger.info(f"Model saved to {self.config.output_dir}")
        
        return trainer
    
    def _train_final_epoch(self, trainer, dataset):
        """Final epoch on 2025 entries only. Recency bias made manifest."""
        # Filter to 2025 entries
        def is_2025(example):
            date = example['meta']['date']
            return date.endswith('/2025')
        
        recent_data = dataset.filter(is_2025)
        
        if len(recent_data) == 0:
            self.logger.warning("No 2025 entries found. The future remains unwritten.")
            return
        
        # Update training args for final epoch
        trainer.args.num_train_epochs = 1
        trainer.args.learning_rate *= 0.3
        
        self.logger.info(f"Final epoch on {len(recent_data)} recent entries")
        trainer.train(resume_from_checkpoint=True)