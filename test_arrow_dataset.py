#!/usr/bin/env python3
"""Test script to verify loading of real indosum dataset from Arrow files."""

import logging
import sys
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import our data processing modules
from data_processing import load_indosum_arrow
from utils import DataArguments, set_seed


def test_arrow_dataset() -> Dict[str, Any]:
    """Test loading the indosum dataset from Arrow files."""
    try:
        logger.info("Testing loading of indosum dataset from Arrow files...")
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Load the dataset from Arrow files
        dataset = load_indosum_arrow()
        
        # Print dataset statistics
        logger.info(f"Dataset splits: {dataset.keys()}")
        for split in dataset:
            logger.info(f"{split} split has {len(dataset[split])} examples")
        
        # Print column names for each split
        for split in dataset:
            logger.info(f"{split} split columns: {dataset[split].column_names}")
        
        # Print a sample example from each split
        for split in dataset:
            logger.info(f"Sample from {split} split:")
            sample = dataset[split][0]
            for key, value in sample.items():
                # Truncate long values for display
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                logger.info(f"  {key}: {value}")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Error loading Arrow dataset: {e}")
        raise e


if __name__ == "__main__":
    test_arrow_dataset()
