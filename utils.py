"""Utility functions for the IndoBart training project."""

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
from transformers import HfArgumentParser, TrainingArguments

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model we are fine-tuning."""

    model_name: str = field(
        default="indobenchmark/indobart-v2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    dataset_name: str = field(
        default="SEACrowd/indosum",
        metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)"}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the downloaded datasets"}
    )
    text_column: str = field(
        default="document",
        metadata={"help": "The name of the column in the datasets containing the documents"}
    )
    summary_column: Optional[str] = field(
        default="summary",
        metadata={"help": "The name of the column in the datasets containing the summaries"}
    )
    max_input_length: int = field(
        default=1024,
        metadata={"help": "Max length of the document input to the model"}
    )
    max_target_length: int = field(
        default=128,
        metadata={"help": "Max length of the summary output from the model"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for preprocessing"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Arguments for training with additional options specific to our task."""

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run evaluation."})
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model for evaluation or pushing to hub (if not training)"}
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Model identifier for uploading to HF Hub"}
    )
    # Generation config parameters required by Seq2SeqTrainer
    generation_max_length: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum length for generation"}
    )
    generation_num_beams: Optional[int] = field(
        default=4,
        metadata={"help": "Number of beams for beam search during generation"}
    )
    generation_config: Optional[Dict] = field(
        default=None,
        metadata={"help": "Generation configuration for text generation"}
    )
    # Required for Seq2SeqTrainer prediction_step
    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)"}
    )


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility in PyTorch, NumPy, and Python's random module.
    
    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def postprocess_text(preds: List[str], labels: List[str]) -> tuple:
    """
    Postprocess the generated predictions and reference labels for metric calculation.
    
    Args:
        preds: List of generated prediction texts
        labels: List of reference label texts
        
    Returns:
        Processed predictions and labels
    """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    # Replace empty predictions with a single space
    preds = [pred if len(pred) > 0 else " " for pred in preds]
    labels = [label if len(label) > 0 else " " for label in labels]
    
    return preds, labels
