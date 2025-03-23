"""Test script to verify direct downloading of indosum dataset."""

import logging
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

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
    dataset_name: str = "seacrowd/indosum"
    dataset_config_name: str = None
    text_column: str = "document"
    summary_column: str = "summary"
    max_input_length: int = 1024
    max_target_length: int = 256


def test_direct_download() -> Dict[str, Any]:
    """Test the direct download function for the indosum dataset."""
    try:
        logger.info("Testing direct download of indosum dataset...")
        
        # Create data arguments
        data_args = DataArguments()
        
        # Load the dataset using our direct download method
        dataset = load_indosum_dataset(data_args=data_args, force_download=True)
        
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
    except ImportError as e:
        logger.error("Import error: %s", e)
        return {"status": "error", "error": f"ImportError: {str(e)}"}
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return {"status": "error", "error": f"FileNotFoundError: {str(e)}"}
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return {"status": "error", "error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    result = test_direct_download()
    logger.info("Test completed with status: %s", result["status"])
