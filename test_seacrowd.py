"""Test script to verify SEACrowd integration with indosum dataset."""

import logging
import sys
from typing import Dict, Any, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_seacrowd_loading() -> Dict[str, Any]:
    """Test loading indosum dataset with SEACrowd."""
    try:
        # Import seacrowd
        import seacrowd as sc
        logger.info("SEACrowd imported successfully")
        
        # List available configs for indosum
        configs = sc.available_config_names("indosum")
        logger.info("Available indosum configurations: %s", configs)
        
        # Get the first available config if any exist
        if configs:
            config_name = configs[0]
            logger.info("Using config: %s", config_name)
            
            # Load using the specific config instead of schema
            logger.info("Loading dataset with specific config name...")
            dataset = sc.load_dataset_by_config_name(config_name=config_name, trust_remote_code=True)
            
            logger.info("Dataset loaded successfully with keys: %s", dataset.keys())
            logger.info("Train split size: %d", len(dataset['train']))
            
            # Check column names
            column_names = dataset["train"].column_names
            logger.info("Column names in train split: %s", column_names)
            
            # Sample from dataset
            sample = dataset["train"][0]
            logger.info("Sample from dataset: %s", sample)
            
            return {
                "status": "success",
                "config_used": config_name,
                "splits": list(dataset.keys()),
                "column_names": column_names,
                "sample": sample
            }
        else:
            logger.error("No configurations found for indosum dataset")
            return {
                "status": "error",
                "error": "No configurations available"
            }
    except ImportError as e:
        logger.error("Error importing SEACrowd: %s", e)
        return {
            "status": "error",
            "error": f"ImportError: {str(e)}"
        }
    except ValueError as e:
        logger.error("Error with dataset values: %s", e)
        return {
            "status": "error",
            "error": f"ValueError: {str(e)}"
        }
    except Exception as e:
        logger.error("Unexpected error loading dataset with SEACrowd: %s", e)
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}"
        }

if __name__ == "__main__":
    result = test_seacrowd_loading()
    logger.info("Test completed with status: %s", result['status'])
