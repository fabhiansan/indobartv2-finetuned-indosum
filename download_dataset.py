"""Script to download the indosum dataset directly to the data directory."""

import os
import logging
import sys
from typing import Dict, List, Optional, Union
import json
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define the dataset URLs - from SeaCrowd/indosum dataset info
DATASET_URLS = {
    "train": "https://raw.githubusercontent.com/SEACrowd/seacrowd-datahub/main/seacrowd/sea_datasets/indosum/IndoSum_train.jsonl",
    "validation": "https://raw.githubusercontent.com/SEACrowd/seacrowd-datahub/main/seacrowd/sea_datasets/indosum/IndoSum_valid.jsonl",
    "test": "https://raw.githubusercontent.com/SEACrowd/seacrowd-datahub/main/seacrowd/sea_datasets/indosum/IndoSum_test.jsonl"
}

# Define the data directory
DATA_DIR = Path("data")


def download_file(url: str, output_path: Path) -> None:
    """
    Download a file from the given URL and save it to the specified path.
    
    Args:
        url: URL to download the file from
        output_path: Path to save the downloaded file
    """
    try:
        logger.info(f"Downloading {url} to {output_path}")
        
        # Make HTTP request
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the content to the file
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        logger.info(f"Successfully downloaded to {output_path}")
    except Exception as e:
        logger.error(f"Error downloading from {url}: {e}")
        raise


def validate_jsonl(file_path: Path) -> bool:
    """
    Validate that the downloaded file is a valid JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        True if the file is valid, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to parse each line as JSON
            line_count = 0
            for i, line in enumerate(f):
                json.loads(line.strip())
                line_count += 1
                
                # Just check the first few lines to avoid checking the entire file
                if i >= 5:
                    break
        
        logger.info(f"Validated {file_path} as a valid JSONL file")
        return True
    except Exception as e:
        logger.error(f"Failed to validate {file_path}: {e}")
        return False


def download_dataset() -> bool:
    """
    Download the indosum dataset.
    
    Returns:
        True if download was successful, False otherwise
    """
    success = True
    for split, url in DATASET_URLS.items():
        output_path = DATA_DIR / f"indosum_{split}.jsonl"
        
        try:
            # Skip if file already exists and is valid
            if output_path.exists() and validate_jsonl(output_path):
                logger.info(f"File {output_path} already exists and is valid. Skipping download.")
                continue
                
            # Download the file
            download_file(url, output_path)
            
            # Validate the downloaded file
            if not validate_jsonl(output_path):
                logger.error(f"Downloaded file {output_path} is not a valid JSONL file")
                success = False
                
        except Exception as e:
            logger.error(f"Failed to download {split} split: {e}")
            success = False
    
    return success


if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    # Download the dataset
    if download_dataset():
        logger.info("Dataset download completed successfully")
    else:
        logger.error("Dataset download failed")
        sys.exit(1)
