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
        for ids in generated_ids:
            try:
                # Use our safer decode implementation to handle out-of-range token IDs
                text = tokenizer.decode(ids, skip_special_tokens=True)
                decoded_summaries.append(text)
            except Exception as e:
                logger.warning(f"Error during decoding: {e}")
                # Fallback to a simpler approach if there's an error
                filtered_ids = [id for id in ids if 0 <= id < tokenizer.vocab_size]
                text = tokenizer.decode(filtered_ids, skip_special_tokens=True)
                decoded_summaries.append(text)
        
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
