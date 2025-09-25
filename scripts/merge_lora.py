# scripts/merge_lora.py
#!/usr/bin/env python3
"""
Merge LoRA adapters with base model. Making the temporary permanent.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.atramentum.utils import logging as alog


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters. Commitment issues resolved."
    )
    parser.add_argument('--base-model', type=str, required=True)
    parser.add_argument('--adapter-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--push-to-hub', action='store_true')
    parser.add_argument('--hub-model-id', type=str, default=None)
    
    args = parser.parse_args()
    
    logger = alog.get_logger(__name__)
    logger.info("Loading base model. This will take a moment of contemplation.")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load adapter
    logger.info("Loading adapter. Temporary becomes permanent.")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    # Merge
    logger.info("Merging adapter with base model.")
    model = model.merge_and_unload()
    
    # Save merged model
    logger.info(f"Saving merged model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    
    # Save tokenizer too
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_dir)
    
    # Optionally push to hub
    if args.push_to_hub:
        hub_id = args.hub_model_id or Path(args.output_dir).name
        logger.info(f"Pushing to hub: {hub_id}")
        model.push_to_hub(hub_id, private=True)
        tokenizer.push_to_hub(hub_id, private=True)
    
    logger.info("Merge complete. The transformation is permanent.")


if __name__ == '__main__':
    main()