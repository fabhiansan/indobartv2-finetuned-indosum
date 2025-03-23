"""Test script to verify loading of local indosum dataset."""

import logging
import sys
from dataclasses import dataclass
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import our data processing modules
from data_processing import load_indosum_dataset


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    dataset_name: str = "indosum"
    dataset_config_name: str = None
    text_column: str = "document"
    summary_column: str = "summary"
    max_input_length: int = 1024
    max_target_length: int = 256


def test_local_dataset() -> Dict[str, Any]:
    """Test loading the indosum dataset from local files."""
    try:
        logger.info("Testing loading of local indosum dataset...")
        
        # Create data arguments
        data_args = DataArguments()
        
        # Load the dataset using our local loading method
        dataset = load_indosum_dataset(data_args=data_args)
        
        # Log information about the loaded dataset
        logger.info("Dataset loaded successfully with splits: %s", dataset.keys())
        
        for split in dataset:
            logger.info("Split '%s' has %d examples", split, len(dataset[split]))
            logger.info("Column names in '%s': %s", split, dataset[split].column_names)
            
        # Show a sample from the training data
        if "train" in dataset:
            sample = dataset["train"][0]
            # Log a truncated version of the document to avoid excessive output
            document = sample["document"][:200] + "..." if len(sample["document"]) > 200 else sample["document"]
            logger.info("Sample document (truncated): %s", document)
            logger.info("Sample summary: %s", sample["summary"])
        
        return {
            "status": "success",
            "splits": list(dataset.keys()),
            "num_examples": {split: len(dataset[split]) for split in dataset},
            "column_names": {split: dataset[split].column_names for split in dataset}
        }
    except Exception as e:
        logger.error("Error loading local dataset: %s", e)
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    result = test_local_dataset()
    logger.info("Test completed with status: %s", result["status"])
