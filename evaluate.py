"""Evaluation utilities for IndoBart summarization model."""

from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizer

from utils import DataArguments, logger, postprocess_text


def generate_summaries(
    model: AutoModelForSeq2SeqLM,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    data_args: DataArguments,
    output_path: Optional[str] = None,
    num_beams: int = 4,
    max_length: int = 128,
    min_length: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8,
) -> tuple[List[str], List[str]]:
    """
    Generate summaries for the evaluation dataset and compute metrics.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_dataset: Dataset for evaluation
        data_args: Data configuration arguments
        output_path: Optional path to save generated summaries
        num_beams: Number of beams for generation
        max_length: Maximum length of generated summaries
        min_length: Minimum length of generated summaries
        device: Device to run generation on
        batch_size: Batch size for generation
        
    Returns:
        Tuple of (generated summaries, reference summaries)
    """
    logger.info("Generating summaries for evaluation...")
    
    # Get columns from dataset
    input_column = data_args.text_column
    target_column = data_args.summary_column
    
    # Create a raw version of the dataset with only input and target columns
    raw_dataset = eval_dataset.map(
        lambda x: {
            input_column: tokenizer.decode(x["input_ids"], skip_special_tokens=True),
            target_column: tokenizer.decode(x["labels"], skip_special_tokens=True),
        }
    )
    
    generated_summaries = []
    reference_summaries = []
    
    # Generate in batches to avoid OOM
    for i in range(0, len(raw_dataset), batch_size):
        batch = raw_dataset[i:i+batch_size]
        inputs = batch[input_column]
        targets = batch[target_column]
        
        # Tokenize inputs
        inputs_tokenized = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=data_args.max_input_length,
            return_tensors="pt",
        ).to(device)
        
        # Generate summaries
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs_tokenized["input_ids"],
                attention_mask=inputs_tokenized["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        
        # Decode generated summaries
        decoded_summaries = []
        for batch_idx, ids in enumerate(generated_ids):
            try:
                # Log the token IDs for debugging
                logger.info(f"Decoding batch item {batch_idx}, token IDs: {ids[:10]}... (showing first 10)")
                
                # Check vocab size
                if hasattr(tokenizer, 'sp_model'):
                    vocab_size = tokenizer.sp_model.get_piece_size()
                    logger.info(f"Tokenizer vocabulary size: {vocab_size}")
                    
                    # Check for out-of-range IDs
                    invalid_ids = [id.item() for id in ids if id.item() >= vocab_size or id.item() < 0]
                    if invalid_ids:
                        logger.warning(f"Found {len(invalid_ids)} invalid token IDs: {invalid_ids[:5]}... (showing up to 5)")
                
                # Use our safer decode implementation to handle out-of-range token IDs
                text = tokenizer.decode(ids, skip_special_tokens=True)
                logger.info(f"Successfully decoded batch item {batch_idx}: '{text[:50]}...' (showing first 50 chars)")
                decoded_summaries.append(text)
            except Exception as e:
                logger.warning(f"Error during decoding batch item {batch_idx}: {e}")
                # Fallback to a simpler approach if there's an error
                try:
                    # Convert tensor to list if needed
                    if torch.is_tensor(ids):
                        ids_list = ids.tolist()
                    else:
                        ids_list = list(ids)
                    
                    # Get vocab size
                    vocab_size = 32000  # Default fallback
                    if hasattr(tokenizer, 'sp_model'):
                        vocab_size = tokenizer.sp_model.get_piece_size()
                    elif hasattr(tokenizer, 'vocab_size'):
                        vocab_size = tokenizer.vocab_size
                    
                    logger.info(f"Using fallback decoding with vocab size: {vocab_size}")
                    
                    # Filter out-of-range IDs
                    filtered_ids = [id for id in ids_list if 0 <= id < vocab_size]
                    logger.info(f"Filtered {len(ids_list) - len(filtered_ids)} invalid token IDs")
                    
                    # Decode with filtered IDs
                    text = tokenizer.decode(filtered_ids, skip_special_tokens=True)
                    logger.info(f"Fallback decoding successful: '{text[:50]}...' (showing first 50 chars)")
                    decoded_summaries.append(text)
                except Exception as fallback_error:
                    logger.error(f"Fallback decoding also failed: {fallback_error}")
                    # If all else fails, add an empty string to maintain alignment
                    decoded_summaries.append("")
        
        generated_summaries.extend(decoded_summaries)
        reference_summaries.extend(targets)
        
        logger.info(f"Generated summaries for batch {i//batch_size+1}/{len(raw_dataset)//batch_size+1}")
    
    # Postprocess summaries
    generated_summaries, reference_summaries = postprocess_text(
        generated_summaries, reference_summaries
    )
    
    # Optionally save the results
    if output_path:
        with open(f"{output_path}/generated_summaries.txt", "w", encoding="utf-8") as f:
            for summary in generated_summaries:
                f.write(summary + "\n")
        
        with open(f"{output_path}/reference_summaries.txt", "w", encoding="utf-8") as f:
            for summary in reference_summaries:
                f.write(summary + "\n")
    
    return generated_summaries, reference_summaries


def compute_rouge_scores(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE scores for summarization evaluation.
    
    Args:
        predictions: List of generated summaries
        references: List of reference summaries
        
    Returns:
        Dictionary of ROUGE scores
    """
    import evaluate
    
    logger.info("Computing ROUGE scores...")
    
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    
    # Format scores
    results = {k: round(v * 100, 2) for k, v in results.items()}
    
    logger.info(f"ROUGE scores: {results}")
    
    return results


def evaluate_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    data_args: DataArguments,
    output_path: Optional[str] = None,
    num_beams: int = 4,
    max_length: int = 128,
    min_length: int = 10,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Complete evaluation pipeline: generate summaries and compute metrics.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_dataset: Dataset for evaluation
        data_args: Data configuration arguments
        output_path: Optional path to save generated summaries
        num_beams: Number of beams for generation
        max_length: Maximum length of generated summaries
        min_length: Minimum length of generated summaries
        batch_size: Batch size for generation
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Generate summaries
    predictions, references = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        data_args=data_args,
        output_path=output_path,
        num_beams=num_beams,
        max_length=max_length,
        min_length=min_length,
        batch_size=batch_size,
    )
    
    # Compute ROUGE scores
    metrics = compute_rouge_scores(predictions, references)
    
    # Save metrics to file if output path provided
    if output_path:
        import json
        with open(f"{output_path}/eval_results.json", "w") as f:
            json.dump(metrics, f, indent=4)
    
    return metrics
