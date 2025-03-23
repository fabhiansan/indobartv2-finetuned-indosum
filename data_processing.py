"""Data processing utilities for IndoBart training on indosum dataset."""

from typing import Dict, List, Optional, Union, Any, Callable
import os
import shutil
import logging

import datasets
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

from utils import DataArguments, logger


def load_indosum_dataset(
    data_args: DataArguments,
    cache_dir: Optional[str] = None,
    force_download: bool = True
) -> DatasetDict:
    """
    Load the indosum dataset from Hugging Face.
    
    Args:
        data_args: Configuration for data loading
        cache_dir: Directory to cache the dataset
        force_download: Whether to force a fresh download
        
    Returns:
        Dataset dictionary with train, validation, and test splits
    """
    logger.info("Loading indosum dataset from Hugging Face...")
    
    # Clean the dataset cache if forcing download
    if force_download:
        try:
            # Get cache directory
            from datasets.config import HF_DATASETS_CACHE
            cache_path = os.path.join(HF_DATASETS_CACHE, "downloads")
            
            if os.path.exists(cache_path):
                logger.info(f"Clearing dataset cache at {cache_path}")
                # Remove all 'indosum' related files in cache
                for root, dirs, files in os.walk(cache_path):
                    if "indosum" in root.lower():
                        logger.info(f"Removing {root}")
                        shutil.rmtree(root, ignore_errors=True)
            
            # Also try to clear the extracted datasets
            extracted_path = os.path.join(HF_DATASETS_CACHE, "extracted")
            if os.path.exists(extracted_path):
                for root, dirs, files in os.walk(extracted_path):
                    if "indosum" in root.lower():
                        logger.info(f"Removing {root}")
                        shutil.rmtree(root, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error clearing dataset cache: {e}")
    
    # Set the download mode
    download_mode = "force_redownload" if force_download else None
    
    try:
        # Load the dataset using the Hugging Face datasets library
        dataset = load_dataset(
            "SEACrowd/indosum",  # Use the SEACrowd/indosum dataset
            trust_remote_code=True,  # Required for SEACrowd datasets
            cache_dir=cache_dir,
            download_mode=download_mode,
        )
        
        logger.info(f"Dataset loaded with splits: {dataset.keys()}")
        
        # Rename columns if needed to match expected format
        if "text" in dataset["train"].column_names and data_args.text_column != "text":
            dataset = dataset.rename_column("text", data_args.text_column)
        if "summary" in dataset["train"].column_names and data_args.summary_column != "summary":
            dataset = dataset.rename_column("summary", data_args.summary_column)
            
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face: {e}")
        
        try:
            # Fallback to using the datasets library directly on the GitHub URL
            logger.info("Trying to load directly using the datasets library...")
            
            # Try loading directly using the datasets library's native functionality
            dataset = load_dataset(
                "seacrowd/indosum",  # Lowercase variant
                trust_remote_code=True,
                cache_dir=cache_dir,
                download_mode=download_mode,
            )
            
            logger.info(f"Successfully loaded dataset: {dataset.keys()}")
            
            # Rename columns if needed
            if "text" in dataset["train"].column_names and data_args.text_column != "text":
                dataset = dataset.rename_column("text", data_args.text_column)
            if "summary" in dataset["train"].column_names and data_args.summary_column != "summary":
                dataset = dataset.rename_column("summary", data_args.summary_column)
                
            return dataset
            
        except Exception as e2:
            logger.error(f"Failed to load dataset from seacrowd: {e2}")
            raise RuntimeError(f"Failed to load the dataset: {e}, then {e2}")


def preprocess_indosum_examples(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    padding: str = "max_length",
    is_training: bool = True
) -> Dict[str, List]:
    """
    Preprocess indosum examples for the model (tokenize inputs and targets).
    
    Args:
        examples: Dictionary containing document and summary examples
        tokenizer: Tokenizer for the model
        data_args: Configuration for data processing
        padding: Padding strategy to use
        is_training: Whether this is for training
        
    Returns:
        Processed examples with model inputs
    """
    # Get the column names for inputs and targets
    input_column = data_args.text_column
    target_column = data_args.summary_column
    
    # Temporarily set max_target_length for validation
    max_target_length = data_args.max_target_length
    
    # Prepare inputs
    inputs = examples[input_column]
    targets = examples[target_column]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=data_args.max_input_length,
        padding=padding,
        truncation=True,
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )
    
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100
    # to ignore padding in the loss
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def prepare_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    preprocessing_num_workers: Optional[int] = None
) -> DatasetDict:
    """
    Prepare the dataset for training by preprocessing examples.
    
    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer for the model
        data_args: Configuration for data processing
        preprocessing_num_workers: Number of workers for preprocessing
        
    Returns:
        Processed dataset ready for training
    """
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"] if "validation" in dataset else dataset["test"]
    
    # Define preprocessing function for mapping
    def preprocess_function(examples: Dict[str, List]) -> Dict[str, List]:
        return preprocess_indosum_examples(
            examples=examples,
            tokenizer=tokenizer,
            data_args=data_args,
            padding="max_length",
            is_training=True
        )
    
    # Apply preprocessing to train dataset
    processed_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on train dataset",
    )
    
    # Apply preprocessing to eval dataset
    processed_eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
        desc="Running tokenizer on validation dataset",
    )
    
    return DatasetDict({
        "train": processed_train_dataset,
        "validation": processed_eval_dataset
    })


def get_data_collator(tokenizer: PreTrainedTokenizer) -> Callable:
    """
    Get the appropriate data collator for the model.
    
    Args:
        tokenizer: Tokenizer for the model
        
    Returns:
        Data collator function
    """
    from transformers import DataCollatorForSeq2Seq
    
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
