#!/usr/bin/env python3
"""
Main script for fine-tuning IndoBart-v2 on the indosum dataset.

This script handles the complete pipeline:
1. Data loading and preprocessing
2. Model loading and setup
3. Training
4. Evaluation
5. Pushing to Hugging Face Hub
"""

import argparse
import os
import sys
from typing import Dict, Optional

import torch
from transformers import HfArgumentParser, set_seed

from data_processing import (
    load_indosum_dataset,
    prepare_dataset,
    get_data_collator,
)
from evaluate import evaluate_model
from model import load_model_and_tokenizer, get_model_for_evaluation
from trainer import SummarizationTrainer, get_compute_metrics_fn
from utils import (
    ModelArguments,
    DataArguments,
    CustomTrainingArguments,
    logger,
    set_seed as set_random_seed,
)


def parse_args() -> tuple:
    """
    Parse command line arguments.
    
    Returns:
        Tuple of parsed arguments: (model_args, data_args, training_args)
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If a single JSON file is provided, parse it
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        # Otherwise parse command line arguments
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Log some info about the arguments
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training arguments: {training_args}")
    
    return model_args, data_args, training_args


def main() -> None:
    """Main function to run the training and evaluation pipeline."""
    # Parse arguments
    model_args, data_args, training_args = parse_args()
    
    # Set seed for reproducibility
    set_random_seed(training_args.seed)
    
    # Make sure output directory exists
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Check if we need to train or just evaluate
    do_train = training_args.do_train
    do_eval = training_args.do_eval
    push_to_hub = training_args.push_to_hub
    
    # Initialize model and tokenizer
    if do_train:
        model, tokenizer = load_model_and_tokenizer(model_args)
        
        # Load and preprocess dataset
        logger.info("Loading dataset...")
        processed_dataset = None
        cache_dir = data_args.cache_dir
        preprocessing_num_workers = data_args.preprocessing_num_workers
        
        try:
            # Load from the real Arrow dataset by default (use_mock=False)
            raw_dataset = load_indosum_dataset(data_args, cache_dir, use_mock=False)
            
            processed_dataset = prepare_dataset(
                raw_dataset,
                tokenizer,
                data_args,
                preprocessing_num_workers
            )
            logger.info("Dataset loaded and preprocessed successfully")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Trying to load mock dataset as fallback...")
            try:
                # Try using mock dataset as fallback
                raw_dataset = load_indosum_dataset(data_args, cache_dir, use_mock=True)
                processed_dataset = prepare_dataset(
                    raw_dataset,
                    tokenizer,
                    data_args,
                    preprocessing_num_workers
                )
                logger.info("Mock dataset loaded successfully as fallback")
            except Exception as e2:
                logger.error(f"Error loading mock dataset: {e2}")
                raise RuntimeError("Failed to load both real and mock datasets") from e2
        
        # Get data collator
        data_collator = get_data_collator(tokenizer)
        
        # Get compute metrics function
        compute_metrics_fn = get_compute_metrics_fn(tokenizer)
        
        # Initialize trainer
        trainer = SummarizationTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            data_collator=data_collator,
            training_args=training_args,
            compute_metrics_fn=compute_metrics_fn,
        )
        
        # Train the model
        trainer.train()
        
        # Evaluate after training
        if do_eval:
            logger.info("Evaluating model after training...")
            eval_metrics = trainer.evaluate()
            logger.info(f"Evaluation metrics: {eval_metrics}")
        
        # Push to Hub if requested
        if push_to_hub and training_args.hub_model_id:
            trainer.push_to_hub(training_args.hub_model_id)
    
    elif training_args.model_path:
        # Load model from path (for evaluation or pushing to hub only)
        model, tokenizer = get_model_for_evaluation(training_args.model_path)
        
        if do_eval:
            # Load and preprocess dataset for evaluation
            logger.info("Loading dataset...")
            processed_dataset = None
            cache_dir = data_args.cache_dir
            
            try:
                # Load from the real Arrow dataset by default (use_mock=False)
                raw_dataset = load_indosum_dataset(data_args, cache_dir, use_mock=False)
                
                processed_dataset = prepare_dataset(
                    raw_dataset,
                    tokenizer,
                    data_args,
                )
                logger.info("Dataset loaded and preprocessed successfully")
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                logger.info("Trying to load mock dataset as fallback...")
                try:
                    # Try using mock dataset as fallback
                    raw_dataset = load_indosum_dataset(data_args, cache_dir, use_mock=True)
                    processed_dataset = prepare_dataset(
                        raw_dataset,
                        tokenizer,
                        data_args,
                    )
                    logger.info("Mock dataset loaded successfully as fallback")
                except Exception as e2:
                    logger.error(f"Error loading mock dataset: {e2}")
                    raise RuntimeError("Failed to load both real and mock datasets") from e2
            
            # Evaluate the model using separate evaluation script
            eval_metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=processed_dataset["validation"],
                data_args=data_args,
                output_path=training_args.output_dir,
                num_beams=training_args.generation_num_beams,
                max_length=data_args.max_target_length,
            )
            
            logger.info(f"Evaluation metrics: {eval_metrics}")
        
        if push_to_hub and training_args.hub_model_id:
            logger.info(f"Pushing model to Hub as: {training_args.hub_model_id}")
            model.push_to_hub(training_args.hub_model_id)
            tokenizer.push_to_hub(training_args.hub_model_id)
            logger.info("Model successfully pushed to Hub!")
    
    else:
        logger.error(
            "Either --do_train must be true or --model_path must be provided "
            "for evaluation or pushing to hub."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
