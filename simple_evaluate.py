#!/usr/bin/env python3
"""
Simple evaluation script for IndoBart models that safely handles out-of-range token IDs.
"""

import os
import sys
import glob
import json
import logging
import argparse
from typing import Dict, List, Optional, Any

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import MBartConfig, MBartForConditionalGeneration
from datasets import load_dataset, Dataset, DatasetDict
from rouge_score import rouge_scorer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import the patched IndoNLGTokenizer to handle out-of-range token IDs
from model import IndoNLGTokenizer


def load_indosum_dataset(dataset_dir: str) -> Dataset:
    """
    Load only the test split of the IndoSum dataset from Arrow files.
    
    Args:
        dataset_dir: Directory containing the Arrow files
        
    Returns:
        Test dataset
    """
    logger.info(f"Loading indosum dataset from Arrow files in {dataset_dir}")
    
    try:
        # Load test dataset from Arrow file
        test_path = os.path.join(dataset_dir, "test")
        test_dataset = load_dataset("arrow", data_files=None, split="train", data_dir=test_path)
        logger.info(f"Loaded {len(test_dataset)} examples for test split")
        
        # Verify dataset format
        if "article" in test_dataset.column_names and "summary" in test_dataset.column_names:
            logger.info(f"Dataset features: {test_dataset.features}")
            return test_dataset
        else:
            # Try alternative column names
            if "document" in test_dataset.column_names:
                column_mapping = {"document": "article"}
                test_dataset = test_dataset.rename_columns(column_mapping)
            
            logger.info(f"Final dataset features: {test_dataset.features}")
            return test_dataset
            
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def safe_decode(tokenizer, token_ids, skip_special_tokens=True):
    """
    Safely decode token IDs, handling out-of-range IDs.
    
    Args:
        tokenizer: The tokenizer to use
        token_ids: Token IDs to decode
        skip_special_tokens: Whether to skip special tokens
        
    Returns:
        Decoded text
    """
    try:
        # Try standard decoding first
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    except Exception as e:
        logger.warning(f"Error in standard decoding: {e}")
        
        # Handle tensor inputs
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Get vocabulary size
        vocab_size = 32000  # Default fallback
        if hasattr(tokenizer, 'sp_model'):
            vocab_size = tokenizer.sp_model.get_piece_size()
        elif hasattr(tokenizer, 'vocab_size'):
            vocab_size = tokenizer.vocab_size
            
        # Filter out-of-range IDs
        filtered_ids = [id for id in token_ids if 0 <= id < vocab_size]
        logger.info(f"Filtered {len(token_ids) - len(filtered_ids)} out-of-range IDs")
        
        try:
            return tokenizer.decode(filtered_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e2:
            logger.error(f"Fallback decoding also failed: {e2}")
            
            # Last resort: decode token by token
            result = []
            for id in filtered_ids:
                try:
                    if hasattr(tokenizer, 'sp_model') and 0 <= id < tokenizer.sp_model.get_piece_size():
                        piece = tokenizer.sp_model.id_to_piece(id)
                        result.append(piece)
                except:
                    pass
            
            text = "".join(result)
            if hasattr(tokenizer, 'sp_model'):
                text = text.replace("▁", " ")  # Basic SentencePiece post-processing
                
            return text


def prepare_dataset(dataset: Dataset, tokenizer: IndoNLGTokenizer, max_input_length: int = 1024, max_target_length: int = 128):
    """
    Prepare dataset for evaluation by tokenizing inputs.
    
    Args:
        dataset: Dataset to prepare
        tokenizer: Tokenizer to use
        max_input_length: Maximum input length
        max_target_length: Maximum target length
        
    Returns:
        Prepared dataset
    """
    def preprocess_function(examples):
        # Extract inputs and targets
        inputs = examples["article"] if "article" in examples else examples["document"]
        targets = examples["summary"]
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs, 
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply preprocessing function
    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset for evaluation"
    )


def evaluate_model(
    model_path: str,
    tokenizer: IndoNLGTokenizer,
    dataset: Dataset,
    output_dir: str,
    num_beams: int = 4,
    max_length: int = 128,
    min_length: int = 10,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate a model on a dataset.
    
    Args:
        model_path: Path to the model checkpoint
        tokenizer: Tokenizer to use
        dataset: Dataset to evaluate on
        output_dir: Directory to save outputs
        num_beams: Number of beams for generation
        max_length: Maximum generation length
        min_length: Minimum generation length
        batch_size: Batch size for generation
        device: Device to run on
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating model: {model_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model safely
    try:
        # First try loading with config to control vocab size
        config = MBartConfig.from_pretrained(model_path)
        model = MBartForConditionalGeneration(config)
        
        # Load state dict manually
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")
            
            # Filter problematic embeddings
            if "model.encoder.embed_tokens.weight" in state_dict:
                embed_weight = state_dict["model.encoder.embed_tokens.weight"]
                if embed_weight.size(0) > config.vocab_size:
                    logger.warning(f"Truncating embedding matrix from {embed_weight.size(0)} to {config.vocab_size}")
                    state_dict["model.encoder.embed_tokens.weight"] = embed_weight[:config.vocab_size, :]
                    
                    if "model.decoder.embed_tokens.weight" in state_dict:
                        state_dict["model.decoder.embed_tokens.weight"] = state_dict["model.decoder.embed_tokens.weight"][:config.vocab_size, :]
                        
            # Load filtered state dict
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model loaded with safe approach")
        else:
            # Fallback to standard loading
            model = MBartForConditionalGeneration.from_pretrained(model_path)
            logger.info("Model loaded with standard approach")
            
    except Exception as e:
        logger.warning(f"Safe loading failed: {e}, trying standard loading")
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        
    # Move model to device
    model.to(device)
    model.eval()
    
    # Generate summaries
    generated_summaries = []
    reference_summaries = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating summaries"):
        batch = dataset[i:i+batch_size]
        
        # Prepare inputs for model
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        
        # Extract reference summaries
        references = []
        for label_ids in batch["labels"]:
            reference = safe_decode(tokenizer, label_ids, skip_special_tokens=True)
            references.append(reference)
        
        # Generate summaries
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        
        # Decode generated summaries
        predictions = []
        for ids in generated_ids:
            prediction = safe_decode(tokenizer, ids, skip_special_tokens=True)
            predictions.append(prediction)
            
        # Save decoded summaries
        generated_summaries.extend(predictions)
        reference_summaries.extend(references)
    
    # Save generated and reference summaries
    with open(os.path.join(output_dir, "generated.txt"), "w", encoding="utf-8") as f:
        for summary in generated_summaries:
            f.write(summary + "\n")
            
    with open(os.path.join(output_dir, "reference.txt"), "w", encoding="utf-8") as f:
        for summary in reference_summaries:
            f.write(summary + "\n")
    
    # Compute ROUGE scores
    logger.info("Computing ROUGE scores")
    metrics = compute_rouge(generated_summaries, reference_summaries)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores for summarization evaluation.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        
    Returns:
        Dictionary of ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    r1_precision, r1_recall, r1_f1 = 0, 0, 0
    r2_precision, r2_recall, r2_f1 = 0, 0, 0
    rl_precision, rl_recall, rl_f1 = 0, 0, 0
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        
        r1_precision += scores['rouge1'].precision
        r1_recall += scores['rouge1'].recall
        r1_f1 += scores['rouge1'].fmeasure
        
        r2_precision += scores['rouge2'].precision
        r2_recall += scores['rouge2'].recall
        r2_f1 += scores['rouge2'].fmeasure
        
        rl_precision += scores['rougeL'].precision
        rl_recall += scores['rougeL'].recall
        rl_f1 += scores['rougeL'].fmeasure
    
    n = len(predictions)
    return {
        "rouge1_precision": r1_precision / n,
        "rouge1_recall": r1_recall / n,
        "rouge1_f1": r1_f1 / n,
        
        "rouge2_precision": r2_precision / n,
        "rouge2_recall": r2_recall / n,
        "rouge2_f1": r2_f1 / n,
        
        "rougeL_precision": rl_precision / n,
        "rougeL_recall": rl_recall / n,
        "rougeL_f1": rl_f1 / n,
        
        # Legacy names for compatibility with original script
        "rouge1": r1_f1 / n,
        "rouge2": r2_f1 / n,
        "rougeL": rl_f1 / n,
    }


def find_checkpoints(output_dir: str) -> List[str]:
    """
    Find checkpoints in a directory.
    
    Args:
        output_dir: Directory to search for checkpoints
        
    Returns:
        List of checkpoint paths
    """
    # Look for checkpoint directories
    checkpoint_dirs = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    
    # Add main directory if it has a model
    if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
        checkpoint_dirs.append(output_dir)
        
    logger.info(f"Found {len(checkpoint_dirs)} checkpoints")
    for checkpoint in checkpoint_dirs:
        logger.info(f"  - {os.path.basename(checkpoint)}")
        
    return checkpoint_dirs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple IndoBart Model Evaluation")
    parser.add_argument("--model_dir", required=True, help="Directory with model checkpoints")
    parser.add_argument("--dataset_dir", default="/home/jupyter-23522029/dataset/indosum", help="Dataset directory")
    parser.add_argument("--output_dir", default="./simple_evaluation", help="Output directory for results")
    parser.add_argument("--checkpoint", help="Specific checkpoint to evaluate (optional)")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for generation")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--min_length", type=int, default=10, help="Minimum generation length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    try:
        logger.info("Loading IndoNLGTokenizer")
        import huggingface_hub
        
        # Download sentencepiece model explicitly
        spm_path = huggingface_hub.hf_hub_download(
            repo_id="indobenchmark/indobart-v2",
            filename="sentencepiece.bpe.model",
            cache_dir=None
        )
        logger.info(f"Found sentencepiece model at: {spm_path}")
        
        # Initialize with explicit model path
        tokenizer = IndoNLGTokenizer(vocab_file=spm_path)
        logger.info("Successfully loaded IndoNLGTokenizer")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return
    
    # Load dataset
    try:
        test_dataset = load_indosum_dataset(args.dataset_dir)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Prepare dataset
    try:
        logger.info("Preparing dataset for evaluation")
        prepared_dataset = prepare_dataset(
            test_dataset,
            tokenizer,
            max_input_length=1024,
            max_target_length=args.max_length
        )
        logger.info("Dataset preparation complete")
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        return
    
    # Find checkpoints to evaluate
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = find_checkpoints(args.model_dir)
    
    # Evaluate each checkpoint
    results = []
    for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
        checkpoint_name = os.path.basename(checkpoint)
        checkpoint_output_dir = os.path.join(args.output_dir, checkpoint_name)
        
        try:
            metrics = evaluate_model(
                model_path=checkpoint,
                tokenizer=tokenizer,
                dataset=prepared_dataset,
                output_dir=checkpoint_output_dir,
                num_beams=args.num_beams,
                max_length=args.max_length,
                min_length=args.min_length,
                batch_size=args.batch_size
            )
            
            # Add checkpoint info
            metrics["checkpoint"] = checkpoint_name
            results.append(metrics)
            
            logger.info(f"Evaluation metrics for {checkpoint_name}:")
            logger.info(f"  ROUGE-1: {metrics['rouge1']:.4f}")
            logger.info(f"  ROUGE-2: {metrics['rouge2']:.4f}")
            logger.info(f"  ROUGE-L: {metrics['rougeL']:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {checkpoint_name}: {e}")
            results.append({"checkpoint": checkpoint_name, "error": str(e)})
    
    # Create summary report
    if results:
        df = pd.DataFrame(results)
        summary_file = os.path.join(args.output_dir, "summary_report.csv")
        df.to_csv(summary_file, index=False)
        
        logger.info(f"Saved summary report to {summary_file}")
        logger.info(f"Successfully evaluated {len(results)} checkpoints")


if __name__ == "__main__":
    main()
