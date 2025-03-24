#!/usr/bin/env python3
"""
Script to evaluate all checkpoints from a directory and save results to a report folder.
This utility allows batch evaluation of multiple model checkpoints at once.
"""

import os
import glob
import json
import logging
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import torch
import pandas as pd
from tqdm import tqdm
from transformers import HfArgumentParser

# Import project modules
from main import get_model_for_evaluation, evaluate_model
from data_processing import load_indosum_dataset, prepare_dataset
from utils import DataArguments, ModelArguments, CustomTrainingArguments

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    """Arguments for batch checkpoint evaluation."""
    
    checkpoints_dir: str = field(
        metadata={"help": "Directory containing model checkpoints to evaluate"}
    )
    report_dir: str = field(
        default="./reports",
        metadata={"help": "Directory to save evaluation reports"}
    )
    dataset_use_mock: bool = field(
        default=False,
        metadata={"help": "Whether to use mock dataset instead of real dataset"}
    )
    num_beams: int = field(
        default=4,
        metadata={"help": "Number of beams for beam search during generation"}
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run evaluation on (cuda or cpu)"}
    )


def find_checkpoints(checkpoints_dir: str) -> List[str]:
    """
    Find all checkpoint directories in the given directory.
    
    Args:
        checkpoints_dir: Path to directory containing checkpoints
        
    Returns:
        List of paths to checkpoint directories
    """
    # Look for checkpoint-* directories
    checkpoint_dirs = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
    checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]
    
    # Also consider the main directory if it has a model file
    if os.path.exists(os.path.join(checkpoints_dir, "pytorch_model.bin")):
        checkpoint_dirs.append(checkpoints_dir)
    
    logger.info(f"Found {len(checkpoint_dirs)} checkpoints to evaluate")
    return sorted(checkpoint_dirs)


def prepare_report_dir(report_dir: str) -> str:
    """
    Create the report directory if it doesn't exist.
    
    Args:
        report_dir: Path to report directory
        
    Returns:
        Path to report directory
    """
    os.makedirs(report_dir, exist_ok=True)
    return report_dir


def save_metrics(metrics: Dict[str, float], checkpoint_path: str, report_dir: str) -> str:
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        checkpoint_path: Path to the evaluated checkpoint
        report_dir: Directory to save reports
        
    Returns:
        Path to saved report file
    """
    # Extract checkpoint name from path
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Create report filename
    report_file = os.path.join(report_dir, f"{checkpoint_name}_metrics.json")
    
    # Save metrics to JSON file
    with open(report_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {report_file}")
    return report_file


def create_summary_report(all_metrics: List[Dict[str, Any]], report_dir: str) -> str:
    """
    Create a summary report of all evaluated checkpoints.
    
    Args:
        all_metrics: List of dictionaries with metrics and checkpoint info
        report_dir: Directory to save reports
        
    Returns:
        Path to summary report
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_metrics)
    
    # Sort by ROUGE-2 score (descending)
    if "rouge2" in df.columns:
        df = df.sort_values(by="rouge2", ascending=False)
    
    # Save to CSV
    summary_file = os.path.join(report_dir, "summary_report.csv")
    df.to_csv(summary_file, index=False)
    
    # Also save as HTML for better visualization
    html_file = os.path.join(report_dir, "summary_report.html")
    df.to_html(html_file, index=False)
    
    logger.info(f"Saved summary report to {summary_file} and {html_file}")
    return summary_file


def main():
    """Main function to evaluate multiple checkpoints."""
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, EvalArguments))
    
    # Set system arguments for required fields
    import sys
    if "--output_dir" not in sys.argv:
        sys.argv.extend(["--output_dir", "./temp_output"])
    
    model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses()
    
    # Override output_dir with report_dir if needed
    if training_args.output_dir == "./temp_output":
        training_args.output_dir = eval_args.report_dir
    
    # Ensure report directory exists
    report_dir = prepare_report_dir(eval_args.report_dir)
    
    # Find all checkpoints to evaluate
    checkpoint_paths = find_checkpoints(eval_args.checkpoints_dir)
    if not checkpoint_paths:
        logger.error(f"No checkpoints found in {eval_args.checkpoints_dir}")
        return
    
    # Load the tokenizer first from the main model directory
    logger.info(f"Loading tokenizer from main model directory: {eval_args.checkpoints_dir}")
    
    # Variable to hold the base model class we'll use for loading checkpoints
    base_model = None
    
    try:
        # Try to load tokenizer directly from main directory
        from model import IndoNLGTokenizer, load_model_and_tokenizer
        
        # Try loading from the output directory first
        logger.info("Attempting to load model and tokenizer from original training...")
        model_path = eval_args.checkpoints_dir
        model_args.model_name = model_path  # Set model path to load from
        model, tokenizer = load_model_and_tokenizer(model_args)
        # Save the model class for later checkpoint loading
        base_model = model.__class__
        logger.info(f"Successfully loaded tokenizer and model architecture from training output (model type: {model.__class__.__name__})")
    except Exception as e:
        logger.warning(f"Could not load tokenizer from training output: {e}")
        logger.info("Trying to load tokenizer from pretrained model...")
        try:
            # Try loading the pretrained tokenizer directly
            # Use our custom IndoNLGTokenizer which has the fixed decode method
            from model import IndoNLGTokenizer
            from transformers import MBartForConditionalGeneration
            
            # Load the original pretrained model and tokenizer
            tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2", trust_remote_code=True)
            # Just load this to get model structure, we'll replace weights later
            temp_model = MBartForConditionalGeneration.from_pretrained("indobenchmark/indobart-v2")
            base_model = MBartForConditionalGeneration  # Save the class for later checkpoint loading
            logger.info("Successfully loaded tokenizer from pretrained model")
        except Exception as e2:
            logger.error(f"Failed to load tokenizer with IndoNLGTokenizer: {e2}")
            logger.info("Falling back to MBartTokenizer...")
            try:
                from transformers import MBartTokenizer, MBartForConditionalGeneration
                # Last resort - try standard MBartTokenizer
                tokenizer = MBartTokenizer.from_pretrained("indobenchmark/indobart-v2", trust_remote_code=True)
                base_model = MBartForConditionalGeneration
                logger.info("Successfully loaded MBartTokenizer")
            except Exception as e3:
                logger.error(f"Failed to load tokenizer: {e3}")
                return
    
    # Load dataset once
    try:
        logger.info("Loading dataset...")
        raw_dataset = load_indosum_dataset(
            data_args,
            cache_dir=model_args.cache_dir,
            force_download=False,
            use_mock=eval_args.dataset_use_mock
        )
        logger.info("Dataset loaded successfully")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Prepare dataset once with our tokenizer
    processed_dataset = prepare_dataset(
        raw_dataset,
        tokenizer, 
        data_args,
        preprocessing_num_workers=data_args.preprocessing_num_workers
    )
    
    # Track all metrics for summary report
    all_metrics = []
    
    # Evaluate each checkpoint
    for checkpoint_path in tqdm(checkpoint_paths, desc="Evaluating checkpoints"):
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        
        try:
            # Load just the model from checkpoint
            logger.info(f"Loading model from: {checkpoint_path}")
            
            if base_model is not None:
                model = base_model.from_pretrained(checkpoint_path)
            else:
                # Fallback to MBartForConditionalGeneration if we couldn't determine the model class
                from transformers import MBartForConditionalGeneration
                model = MBartForConditionalGeneration.from_pretrained(checkpoint_path)
                
            model.to(eval_args.device)
            
            # Evaluate model
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,  # Use the tokenizer we loaded earlier
                eval_dataset=processed_dataset["validation"],
                data_args=data_args,
                output_path=os.path.join(report_dir, os.path.basename(checkpoint_path)),
                num_beams=eval_args.num_beams,
                max_length=data_args.max_target_length,
            )
            
            # Save individual metrics
            save_metrics(metrics, checkpoint_path, report_dir)
            
            # Add to all metrics for summary
            metrics["checkpoint"] = checkpoint_path
            all_metrics.append(metrics)
            
            # Free up memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Error evaluating {checkpoint_path}: {e}")
    
    # Create summary report
    if all_metrics:
        create_summary_report(all_metrics, report_dir)
        logger.info(f"Evaluation complete. Reports saved to {report_dir}")
    else:
        logger.warning("No checkpoints were successfully evaluated")


if __name__ == "__main__":
    main()
