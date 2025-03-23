#!/usr/bin/env python3
"""Test script to verify loading of real indosum dataset from Arrow files."""

import logging
import sys
from typing import Dict, Any
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import our data processing modules
from data_processing import load_indosum_arrow, load_indosum_dataset
from utils import DataArguments, set_seed


def test_arrow_dataset() -> Dict[str, Any]:
    """Test loading the indosum dataset from Arrow files."""
    try:
        logger.info("Testing loading of indosum dataset from Arrow files...")
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Create data arguments
        data_args = DataArguments(
            dataset_name="indosum",
            dataset_config_name=None,
            text_column="document",
            summary_column="summary"
        )
        
        # Load the dataset directly from Arrow files
        arrow_dataset = load_indosum_arrow(base_dir="/home/jupyter-23522029/dataset/indosum")
        
        # Print dataset statistics
        logger.info(f"Dataset splits: {arrow_dataset.keys()}")
        for split in arrow_dataset:
            logger.info(f"{split} split has {len(arrow_dataset[split])} examples")
        
        # Print column names for each split
        for split in arrow_dataset:
            logger.info(f"{split} split columns: {arrow_dataset[split].column_names}")
        
        # Print a sample example from each split
        for split in arrow_dataset:
            logger.info(f"Sample from {split} split:")
            sample = arrow_dataset[split][0]
            for key, value in sample.items():
                # Truncate long values for display
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                logger.info(f"  {key}: {value}")
        
        # Load also using the main loading function
        logger.info("\nAlso testing load_indosum_dataset function with use_mock=False...")
        dataset = load_indosum_dataset(
            data_args=data_args,
            cache_dir=None,
            force_download=False,
            use_mock=False,
            base_dir="/home/jupyter-23522029/dataset/indosum"
        )
        
        # Verify it loaded correctly
        logger.info(f"Dataset loaded through load_indosum_dataset function has splits: {dataset.keys()}")
        for split in dataset:
            logger.info(f"{split} split has {len(dataset[split])} examples")
        
        return arrow_dataset
    
    except Exception as e:
        logger.error(f"Error loading Arrow dataset: {e}")
        raise e


if __name__ == "__main__":
    test_arrow_dataset()
