"""Model loading and configuration for IndoBart-v2 fine-tuning."""

from typing import Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
)

from utils import ModelArguments, logger


def load_model_and_tokenizer(
    model_args: ModelArguments,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load pretrained model and tokenizer from Hugging Face Hub.
    
    Args:
        model_args: Arguments for model loading
        device: Device to load the model onto
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_args.model_name}")
    
    # Load config and update parameters if needed
    config = AutoConfig.from_pretrained(
        model_args.model_name,
        cache_dir=model_args.cache_dir,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name,
        from_tf=bool(".ckpt" in model_args.model_name),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    # Handle tokenizer BOS/EOS tokens (Bart-specific)
    if isinstance(model, BartForConditionalGeneration):
        if model.config.decoder_start_token_id is None and isinstance(config, BartConfig):
            model.config.decoder_start_token_id = tokenizer.bos_token_id
        if (
            model.config.decoder_start_token_id is None
            and not isinstance(config, BartConfig)
        ):
            model.config.decoder_start_token_id = model.config.bos_token_id
    
    # Set device (if using GPU)
    model = model.to(device)
    
    logger.info(f"Model loaded: {model.__class__.__name__}")
    logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    return model, tokenizer


def get_model_for_evaluation(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load a fine-tuned model and tokenizer for evaluation.
    
    Args:
        model_path: Path to the fine-tuned model directory
        device: Device to load the model onto
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading fine-tuned model from: {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    
    return model, tokenizer
