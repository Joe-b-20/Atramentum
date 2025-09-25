# src/atramentum/__init__.py
"""
Atramentum: Where memories go to get distorted.
"""

__version__ = "0.1.2"
__author__ = "JB"

from .data import formatter, validators
from .training import sft_trainer, dpo_trainer
from .inference import generator, prompts
from .utils import io, logging, metrics

__all__ = [
    "formatter", "validators",
    "sft_trainer", "dpo_trainer", 
    "generator", "prompts",
    "io", "logging", "metrics"
]