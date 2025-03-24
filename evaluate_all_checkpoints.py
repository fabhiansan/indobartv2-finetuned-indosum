#!/usr/bin/env python3
"""
Simplified script to evaluate all checkpoints using IndoNLGTokenizer.
This script focuses solely on using the correct tokenizer to avoid "piece id is out of range" errors.
"""

import os
import sys
import glob
import json
import logging
import argparse
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

import torch
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizer

# Import project modules
from main import evaluate_model
from data_processing import load_indosum_dataset, prepare_dataset
from utils import DataArguments, set_seed
from model import IndoNLGTokenizer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def find_checkpoints(output_dir: str) -> List[str]:
    """Find all checkpoint directories in the output directory."""
    # Look for checkpoint directories (checkpoint-*)
    checkpoint_dirs = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    
    # Add main directory if it has a model
    if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
        checkpoint_dirs.append(output_dir)
    
    logger.info(f"Found {len(checkpoint_dirs)} checkpoints to evaluate")
    for checkpoint in checkpoint_dirs:
        logger.info(f"  - {os.path.basename(checkpoint)}")
    
    return checkpoint_dirs


def load_tokenizer() -> IndoNLGTokenizer:
    """Load the IndoNLGTokenizer without relying on auto-detection."""
    logger.info("Loading IndoNLGTokenizer directly...")
    
    try:
        tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2", trust_remote_code=True)
        logger.info("Successfully loaded IndoNLGTokenizer from indobenchmark/indobart-v2")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer from pretrained: {e}")
        raise


def save_metrics(metrics: Dict[str, float], checkpoint_path: str, report_dir: str) -> str:
    """Save evaluation metrics to a JSON file."""
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
    """Create a summary report of all evaluated checkpoints."""
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
    """Main function to evaluate checkpoints."""
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a directory")
    parser.add_argument("--output_dir", required=True, help="Directory containing model checkpoints")
    parser.add_argument("--report_dir", default="./reports", help="Directory to save evaluation reports")
    parser.add_argument("--use_mock", action="store_true", help="Use mock dataset instead of real dataset")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for generation")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum output length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run evaluation on")
    args = parser.parse_args()
    
    # Create report directory
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Set fixed seed for reproducibility
    set_seed(42)
    
    # Find all checkpoints
    checkpoint_paths = find_checkpoints(args.output_dir)
    
    if not checkpoint_paths:
        logger.error(f"No checkpoints found in {args.output_dir}")
        return
    
    # Load tokenizer - this is the key part to fix the "piece id is out of range" error
    try:
        tokenizer = load_tokenizer()
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return
    
    # Create data arguments
    data_args = DataArguments(
        text_column="document",
        summary_column="summary",
        max_input_length=1024,
        max_target_length=args.max_length
    )
    
    # Load dataset
    try:
        logger.info("Loading indosum dataset...")
        raw_dataset = load_indosum_dataset(
            data_args,
            cache_dir=None,
            force_download=False,
            use_mock=args.use_mock
        )
        logger.info("Successfully loaded dataset")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Process dataset only once with our tokenizer
    try:
        logger.info("Preprocessing dataset with tokenizer...")
        processed_dataset = prepare_dataset(
            raw_dataset,
            tokenizer,
            data_args,
            preprocessing_num_workers=None
        )
        logger.info("Dataset preprocessing complete")
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        return
    
    # Track all metrics
    all_metrics = []
    
    # Evaluate each checkpoint
    successful_evaluations = 0
    for checkpoint_path in tqdm(checkpoint_paths, desc="Evaluating checkpoints"):
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        
        try:
            # Load model weights using MBartForConditionalGeneration
            from transformers import MBartForConditionalGeneration
            
            logger.info(f"Loading model from {checkpoint_path}...")
            model = MBartForConditionalGeneration.from_pretrained(checkpoint_path)
            model.to(args.device)
            
            # Evaluate model
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=processed_dataset["validation"],
                data_args=data_args,
                output_path=os.path.join(args.report_dir, os.path.basename(checkpoint_path)),
                num_beams=args.num_beams,
                max_length=args.max_length,
            )
            
            # Save metrics
            save_metrics(metrics, checkpoint_path, args.report_dir)
            
            # Add checkpoint info
            metrics["checkpoint"] = os.path.basename(checkpoint_path)
            all_metrics.append(metrics)
            
            successful_evaluations += 1
            
            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Error evaluating {checkpoint_path}: {e}")
    
    # Create summary if we have successful evaluations
    if all_metrics:
        create_summary_report(all_metrics, args.report_dir)
        logger.info(f"Successfully evaluated {successful_evaluations} out of {len(checkpoint_paths)} checkpoints")
    else:
        logger.warning("No checkpoints were successfully evaluated")


if __name__ == "__main__":
    main()
