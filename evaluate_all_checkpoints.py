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
from datasets import DatasetDict

# Import project modules
from main import evaluate_model
from data_processing import load_indosum_arrow, prepare_dataset
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
        # First, try to get the actual SentencePiece model file
        try:
            import huggingface_hub
            # Download sentencepiece model explicitly
            spm_path = huggingface_hub.hf_hub_download(
                repo_id="indobenchmark/indobart-v2",
                filename="sentencepiece.bpe.model",
                cache_dir=None
            )
            logger.info(f"Found sentencepiece model at: {spm_path}")
            
            # Initialize with the explicit sentencepiece model path
            tokenizer = IndoNLGTokenizer(
                vocab_file=spm_path,
                trust_remote_code=True
            )
            logger.info("Successfully loaded IndoNLGTokenizer with explicit SentencePiece model")
        except Exception as e:
            logger.warning(f"Error downloading SentencePiece model: {e}")
            
            # Fallback to loading from the model class directly
            logger.info("Attempting to load tokenizer from indobart-v2 model...")
            tokenizer = IndoNLGTokenizer.from_pretrained(
                "indobenchmark/indobart-v2", 
                trust_remote_code=True
            )
            logger.info("Successfully loaded IndoNLGTokenizer from pretrained model")
        
        # Verify tokenizer has decode method with out-of-range protection
        if hasattr(tokenizer, 'decode'):
            test_ids = [0, 1, 2, 3, 40000, 999999]  # Intentionally include an out-of-range ID
            try:
                tokenizer.decode(test_ids)
                logger.info("Tokenizer decode method successfully handles out-of-range token IDs")
            except Exception as e:
                logger.warning(f"Tokenizer decode method test failed: {e}")
                logger.info("This is expected if using an older version - our custom implementation should handle this")
        
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
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


def evaluate_checkpoint(
    checkpoint_path: str,
    tokenizer: IndoNLGTokenizer,
    eval_dataset: Any,
    report_dir: str,
    device: torch.device,
    data_args: Optional[DataArguments] = None,
    num_beams: int = 4,
    max_length: int = 512
) -> Dict[str, float]:
    """Evaluate a single checkpoint."""
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    
    try:
        # Load the model with a custom safe approach
        from transformers import MBartForConditionalGeneration
        
        # First, attempt to catch the "piece id out of range" error with a dummy forward pass
        try:
            logger.info("Loading model with safe approach...")
            # Load model config first
            from transformers import MBartConfig
            config = MBartConfig.from_pretrained(checkpoint_path)
            
            # Initialize model with config only
            model = MBartForConditionalGeneration(config)
            
            # Load state dict manually
            import os
            state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(state_dict_path):
                # Load state dict
                state_dict = torch.load(state_dict_path, map_location="cpu")
                
                # Filter out any problematic embeddings entries
                if "model.encoder.embed_tokens.weight" in state_dict:
                    embed_weight = state_dict["model.encoder.embed_tokens.weight"]
                    vocab_size = config.vocab_size
                    
                    # Ensure embeddings match vocab size
                    if embed_weight.size(0) > vocab_size:
                        logger.warning(f"Embeddings size {embed_weight.size(0)} exceeds vocab size {vocab_size}, truncating")
                        state_dict["model.encoder.embed_tokens.weight"] = embed_weight[:vocab_size, :]
                        
                        # Also truncate decoder embeddings if needed
                        if "model.decoder.embed_tokens.weight" in state_dict:
                            state_dict["model.decoder.embed_tokens.weight"] = state_dict["model.decoder.embed_tokens.weight"][:vocab_size, :]
                
                # Load filtered state dict
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"Missing keys when loading state dict: {missing[:5]} (showing up to 5)")
                if unexpected:
                    logger.warning(f"Unexpected keys when loading state dict: {unexpected[:5]} (showing up to 5)")
            else:
                # Fallback to standard loading if no separate state dict file
                model = MBartForConditionalGeneration.from_pretrained(checkpoint_path)
                
            logger.info("Model loaded successfully with safe approach")
            
        except Exception as e:
            logger.warning(f"Safe loading approach failed: {e}")
            logger.info("Falling back to standard model loading")
            # Fallback to standard loading
            model = MBartForConditionalGeneration.from_pretrained(checkpoint_path)
        
        # Move model to device
        model.to(device)
        model.eval()
        
        # Create a report directory for this checkpoint
        checkpoint_report_dir = os.path.join(report_dir, os.path.basename(checkpoint_path))
        os.makedirs(checkpoint_report_dir, exist_ok=True)
        
        # Evaluate model using the already processed dataset
        try:
            # Try monkey patching the tokenizer's sp_model to catch the exact error location
            original_decode = tokenizer.sp_model.decode
            def safe_decode(piece_ids):
                try:
                    return original_decode(piece_ids)
                except Exception as e:
                    logger.error(f"SentencePiece decode error with piece IDs: {piece_ids}")
                    logger.error(f"Error: {e}")
                    raise
            
            # Replace with our safe version that logs details
            tokenizer.sp_model.decode = safe_decode
            
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                data_args=data_args,
                output_path=checkpoint_report_dir,
                num_beams=num_beams,
                max_length=max_length
            )
            
            # Restore original function
            tokenizer.sp_model.decode = original_decode
        
        except Exception as eval_error:
            import traceback
            logger.error(f"Full traceback for {checkpoint_path}:")
            logger.error(traceback.format_exc())
            
            # Try to extract line-by-line info
            tb_parts = traceback.format_exception(type(eval_error), eval_error, eval_error.__traceback__)
            for part in tb_parts:
                logger.error(part.strip())
            
            raise eval_error
        
        # Save metrics
        save_metrics(metrics, checkpoint_path, report_dir)
        
        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating checkpoint {checkpoint_path}: {e}")
        # Add the error to the metrics dictionary for the summary report
        return {"checkpoint": os.path.basename(checkpoint_path), "error": str(e)}


def main():
    """Main function to evaluate checkpoints."""
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a directory")
    parser.add_argument("--output_dir", required=True, help="Directory containing model checkpoints")
    parser.add_argument("--report_dir", default="./reports", help="Directory to save evaluation reports")
    parser.add_argument("--dataset_dir", default="/home/jupyter-23522029/dataset/indosum", 
                       help="Directory containing the dataset Arrow files")
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
        logger.info("Loading indosum test dataset...")
        # Use the load_indosum_arrow function to load only the test dataset
        all_datasets = load_indosum_arrow(
            base_dir=args.dataset_dir
        )
        # Extract only the test split
        raw_dataset = all_datasets["test"]
        logger.info(f"Successfully loaded test dataset with {len(raw_dataset)} examples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Process dataset only once with our tokenizer
    try:
        logger.info("Preprocessing test dataset with tokenizer...")
        # Convert the test dataset to a DatasetDict with both train and validation splits
        # This is needed because prepare_dataset expects a DatasetDict with both splits
        dataset_dict = DatasetDict({
            "train": raw_dataset,  # We use the test dataset for both
            "validation": raw_dataset
        })
        
        processed_dataset = prepare_dataset(
            dataset_dict,
            tokenizer,
            data_args,
            preprocessing_num_workers=None
        )
        
        # We only need the validation split for evaluation
        processed_dataset = processed_dataset["validation"]
        logger.info("Test dataset preprocessing complete")
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
            # Evaluate checkpoint
            metrics = evaluate_checkpoint(
                checkpoint_path,
                tokenizer,
                processed_dataset,
                args.report_dir,
                torch.device(args.device),
                data_args=data_args,
                num_beams=args.num_beams,
                max_length=args.max_length
            )
            
            # Add checkpoint info
            metrics["checkpoint"] = os.path.basename(checkpoint_path)
            all_metrics.append(metrics)
            
            successful_evaluations += 1
            
            # Free memory
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
