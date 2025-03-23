"""Data processing utilities for IndoBart fine-tuning."""

from typing import Dict, List, Optional, Union, Any, Callable
import os
import json
import logging
from pathlib import Path

import datasets
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

from utils import DataArguments, logger


def load_indosum_dataset(
    data_args: DataArguments,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    use_mock: bool = True,
    base_dir: str = "/home/jupyter-23522029/dataset/indosum"
) -> DatasetDict:
    """
    Load the indosum dataset from local files.
    
    Args:
        data_args: Configuration for data loading
        cache_dir: Directory to cache the dataset (can be None, will use data_args.dataset_cache_dir if provided)
        force_download: Whether to force a fresh download (not used for local loading)
        use_mock: Whether to use the mock dataset or the real dataset
        base_dir: Directory containing the dataset Arrow files
        
    Returns:
        Dataset dictionary with train, validation, and test splits
    """
    # Use dataset_cache_dir from data_args if provided and cache_dir is None
    if cache_dir is None and hasattr(data_args, 'dataset_cache_dir'):
        cache_dir = data_args.dataset_cache_dir
        
    if use_mock:
        return load_indosum_jsonl(data_args, cache_dir)
    else:
        return load_indosum_arrow(base_dir)


def load_indosum_arrow(
    base_dir: str = "/home/jupyter-23522029/dataset/indosum"
) -> DatasetDict:
    """
    Load the indosum dataset from Arrow files.
    
    Args:
        base_dir: Directory containing the dataset Arrow files
        
    Returns:
        DatasetDict containing train, validation and test splits
    """
    logger.info(f"Loading indosum dataset from Arrow files in {base_dir}")
    
    train_dir = os.path.join(base_dir, "traindataset")
    dev_dir = os.path.join(base_dir, "devdataset")
    test_dir = os.path.join(base_dir, "testdataset")
    
    # Load each split
    train_dataset = load_from_disk(train_dir)
    logger.info(f"Loaded {len(train_dataset)} examples for train split")
    
    validation_dataset = load_from_disk(dev_dir)
    logger.info(f"Loaded {len(validation_dataset)} examples for validation split")
    
    test_dataset = load_from_disk(test_dir)
    logger.info(f"Loaded {len(test_dataset)} examples for test split")
    
    # Combine into a DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })
    
    # Check column names and rename if needed
    # Inspect the first example to see column names
    logger.info(f"Dataset features: {dataset['train'].features}")
    
    # Make sure we have the expected document/summary columns
    if "document" not in dataset["train"].column_names:
        # If common alternative names are present, rename them
        if "article" in dataset["train"].column_names:
            for split in dataset:
                dataset[split] = dataset[split].rename_column("article", "document")
        elif "text" in dataset["train"].column_names:
            for split in dataset:
                dataset[split] = dataset[split].rename_column("text", "document")
    
    if "summary" not in dataset["train"].column_names:
        # If common alternative names are present, rename them
        if "abstract" in dataset["train"].column_names:
            for split in dataset:
                dataset[split] = dataset[split].rename_column("abstract", "summary")
        elif "headline" in dataset["train"].column_names:
            for split in dataset:
                dataset[split] = dataset[split].rename_column("headline", "summary")
    
    logger.info(f"Dataset loaded with splits: {dataset.keys()}")
    return dataset


def load_indosum_jsonl(
    data_args: DataArguments,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load the indosum dataset from local JSONL files.
    
    Args:
        data_args: Configuration for data loading
        cache_dir: Cache directory for huggingface datasets
        
    Returns:
        DatasetDict containing train, validation and test splits
    """
    logger.info(f"Loading indosum dataset from local files in data/indosum")
    
    # Define the path to the dataset
    data_dir = Path("data/indosum")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory {data_dir} does not exist. Run create_mock_dataset.py first.")
        
    # Define paths to each split
    file_paths = {
        "train": data_dir / "train.jsonl",
        "validation": data_dir / "validation.jsonl",
        "test": data_dir / "test.jsonl"
    }
    
    # Check if all files exist
    missing_files = [str(path) for path, file_path in file_paths.items() if not file_path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing_files)}")
    
    try:
        # Load the dataset using datasets library from local files
        dataset_dict = {}
        
        for split, file_path in file_paths.items():
            logger.info(f"Loading {split} split from {file_path}")
            
            # Use the datasets library to load from jsonl files
            dataset = load_dataset('json', data_files=str(file_path), split='train')
            
            logger.info(f"Loaded {len(dataset)} examples for {split} split")
            dataset_dict[split] = dataset
        
        # Create a DatasetDict
        dataset = DatasetDict(dataset_dict)
        logger.info(f"Dataset loaded with splits: {dataset.keys()}")
        
        # Ensure the column names match what we expect
        for split in dataset.keys():
            # Check if we need to rename columns to match expected format
            if "document" in dataset[split].column_names and data_args.text_column != "document":
                dataset[split] = dataset[split].rename_column("document", data_args.text_column)
            if "summary" in dataset[split].column_names and data_args.summary_column != "summary":
                dataset[split] = dataset[split].rename_column("summary", data_args.summary_column)
            
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset from local files: {e}")
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
