"""Data processing utilities for IndoBart training on indosum dataset."""

from typing import Dict, List, Optional, Union, Any, Callable
import os
import shutil

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
    Load the indosum dataset from the Hugging Face Hub.
    
    Args:
        data_args: Configuration for data loading
        cache_dir: Directory to cache the dataset
        force_download: Whether to force a fresh download
        
    Returns:
        Dataset dictionary with train, validation, and test splits
    """
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    
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
                    if "indosum" in root:
                        logger.info(f"Removing {root}")
                        shutil.rmtree(root, ignore_errors=True)
            
            # Also try to clear the extracted datasets
            extracted_path = os.path.join(HF_DATASETS_CACHE, "extracted")
            if os.path.exists(extracted_path):
                for root, dirs, files in os.walk(extracted_path):
                    if "indosum" in root:
                        logger.info(f"Removing {root}")
                        shutil.rmtree(root, ignore_errors=True)
                        
        except Exception as e:
            logger.warning(f"Error clearing dataset cache: {e}")
    
    # Try to load directly as a packaged dataset
    try:
        logger.info("Attempting to load dataset from IndoNLP package (backup method)...")
        from indobenchmark.datasets import IndoSum
        indobenchmark_dataset = IndoSum(tokenizer=None)
        train = Dataset.from_dict(indobenchmark_dataset.data["train"])
        validation = Dataset.from_dict(indobenchmark_dataset.data["valid"])
        test = Dataset.from_dict(indobenchmark_dataset.data["test"])
        
        return DatasetDict({
            "train": train,
            "validation": validation,
            "test": test
        })
    except Exception as e:
        logger.warning(f"Failed to load dataset from IndoNLP package: {e}")
    
    # Try direct download from GitHub repository
    try:
        logger.info("Trying to load dataset from GitHub repository...")
        # Alternative approach: directly download from GitHub
        dataset = load_dataset(
            "json",
            data_files={
                "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/indosum/train_preprocess.jsonl",
                "validation": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/indosum/valid_preprocess.jsonl",
                "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/indosum/test_preprocess.jsonl"
            },
            cache_dir=cache_dir,
        )
        
        # Rename columns to match expected format
        if "Document" in dataset["train"].column_names:
            dataset = dataset.rename_column("Document", data_args.text_column)
        if "Summary" in dataset["train"].column_names:
            dataset = dataset.rename_column("Summary", data_args.summary_column)
            
        logger.info(f"Successfully loaded dataset from GitHub: {dataset}")
        return dataset
    except Exception as e:
        logger.warning(f"Failed to load dataset from GitHub: {e}")
    
    # Fall back to original SEACrowd dataset as last resort
    logger.info("Trying to load dataset from SEACrowd as last resort...")
    try:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=cache_dir,
            trust_remote_code=True,  # Required for SEACrowd datasets
            download_mode="force_redownload" if force_download else None,
        )
        
        logger.info(f"Dataset loaded with splits: {dataset.keys()}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


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
    eval_dataset = dataset["validation"]
    
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
