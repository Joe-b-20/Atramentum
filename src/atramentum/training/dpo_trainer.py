# src/atramentum/training/dpo_trainer.py
"""
DPO trainer - teaching preferences through suffering.
"""

import os
import torch
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

from ..utils import io as aio
from ..utils import logging as alog


@dataclass
class DPOTrainingConfig:
    """DPO configuration. Choose your suffering carefully."""
    model_path: str  # Path to SFT checkpoint
    dataset_path: str
    output_dir: str
    beta: float = 0.2
    learning_rate: float = 1e-5
    num_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lora_r: int = 64
    lora_alpha: int = 16
    max_seq_length: int = 4096


class PreferenceTrainer:
    """Handles DPO training. Teaching models to prefer darkness over light."""
    
    def __init__(self, config: DPOTrainingConfig):
        self.config = config
        self.logger = alog.get_logger(__name__)
    
    def load_preference_dataset(self) -> Dataset:
        """Load preference pairs. Chosen: darkness. Rejected: toxic positivity."""
        data = aio.read_jsonl(self.config.dataset_path)
        
        # Validate preference format
        valid_data = []
        for item in data:
            if all(k in item for k in ['prompt', 'chosen', 'rejected']):
                valid_data.append(item)
        
        self.logger.info(f"Loaded {len(valid_data)} preference pairs")
        return Dataset.from_list(valid_data)
    
    def setup_model(self):
        """Load base model and prepare for DPO."""
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model
        base_model_name = self._get_base_model_name()
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=os.environ.get('HF_TOKEN')
        )
        
        # Load SFT adapter if exists
        if Path(self.config.model_path).exists():
            model = PeftModel.from_pretrained(model, self.config.model_path)
            model = model.merge_and_unload()  # Merge for DPO
        
        # New LoRA for DPO
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            use_auth_token=os.environ.get('HF_TOKEN')
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer, lora_config
    
    def _get_base_model_name(self) -> str:
        """Extract base model name from config or path."""
        # Check if it's a path to adapters
        adapter_config_path = Path(self.config.model_path) / "adapter_config.json"
        if adapter_config_path.exists():
            import json
            with open(adapter_config_path) as f:
                config = json.load(f)
                return config.get('base_model_name_or_path', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
        
        return 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    
    def train(self):
        """Run DPO training. Preference learning through computational masochism."""
        # Setup
        model, tokenizer, lora_config = self.setup_model()
        dataset = self.load_preference_dataset()
        
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        # DPO config
        dpo_config = DPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            beta=self.config.beta,
            max_length=self.config.max_seq_length,
            max_prompt_length=512,
            gradient_checkpointing=True,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        )
        
        # Create DPO trainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Will use model copy as reference
            args=dpo_config,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            tokenizer=tokenizer,
            peft_config=lora_config,
        )
        
        self.logger.info("Starting DPO training. Teaching preferences...")
        trainer.train()
        
        # Save
        trainer.save_model()
        self.logger.info(f"DPO model saved to {self.config.output_dir}")
        
        return trainer