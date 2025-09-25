# src/atramentum/inference/generator.py
"""
Text generation with RAG support. Manufacturing memories since 2024.
"""

import torch
from typing import List, Dict, Optional
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from . import prompts
from ..utils import logging as alog


class JournalGenerator:
    """Generates journal entries with optional memory retrieval."""
    
    def __init__(
        self,
        model_name: str,
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        load_in_4bit: bool = True
    ):
        self.logger = alog.get_logger(__name__)
        self.device = device
        
        # Setup model and tokenizer
        self.model, self.tokenizer = self._load_model(
            model_name, adapter_path, load_in_4bit
        )
    
    def _load_model(self, model_name: str, adapter_path: str, load_in_4bit: bool):
        """Load model with optional LoRA adapter."""
        # Quantization config if needed
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load adapter if provided
        if adapter_path and Path(adapter_path).exists():
            self.logger.info(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def generate(
        self,
        prompt: str,
        memory: Optional[str] = None,
        mode: str = "generate",
        max_new_tokens: int = 800,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        do_sample: bool = True
    ) -> str:
        """Generate a journal entry with optional memory context."""
        
        # Build full prompt with memory
        user_prompt = prompts.build_prompt(
            mode=mode,
            content=prompt,
            memory=memory or ""
        )
        
        # Format messages
        messages = prompts.format_messages(
            system=prompts.SYSTEM_PERSONA,
            user=user_prompt
        )
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback to simple format
            input_text = f"{prompts.SYSTEM_PERSONA}\n\nUser: {user_prompt}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096 - max_new_tokens
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated.strip()