"""
Create a mock indosum dataset for testing purposes.
This script generates synthetic data following the indosum format.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Sample Indonesian sentences for generating mock data
SAMPLE_SENTENCES = [
    "Indonesia adalah negara kepulauan terbesar di dunia dengan lebih dari 17.000 pulau.",
    "Jakarta merupakan ibu kota Indonesia dan kota terbesar di negara ini.",
    "Bahasa Indonesia adalah bahasa resmi negara Republik Indonesia.",
    "Borobudur adalah candi Buddha terbesar di dunia yang terletak di Magelang, Jawa Tengah.",
    "Bali merupakan salah satu tujuan wisata paling populer di Indonesia.",
    "Batik adalah kain tradisional Indonesia yang telah diakui UNESCO sebagai warisan budaya dunia.",
    "Penduduk Indonesia terdiri dari berbagai suku bangsa dengan keberagaman budaya yang kaya.",
    "Komodo adalah kadal terbesar di dunia yang hanya dapat ditemukan di Indonesia.",
    "Wayang kulit adalah seni pertunjukan tradisional Indonesia yang menampilkan boneka bayangan.",
    "Rendang adalah masakan tradisional Minangkabau yang telah dinobatkan sebagai makanan terlezat di dunia.",
    "Kawah Ijen di Jawa Timur terkenal dengan fenomena api biru yang langka.",
    "Raja Ampat di Papua Barat memiliki keanekaragaman hayati laut tertinggi di dunia.",
    "Gamelan adalah ansambel musik tradisional dari Indonesia yang terdiri dari alat musik perkusi.",
    "Tari Saman dari Aceh dikenal dengan gerakan tangan yang cepat dan sinkronisasi yang sempurna.",
    "Danau Toba di Sumatera Utara adalah danau vulkanik terbesar di dunia."
]


def generate_document(num_sentences: int = 10) -> str:
    """Generate a mock document by randomly combining sample sentences.
    
    Args:
        num_sentences: Number of sentences to include in the document
        
    Returns:
        A string containing the generated document
    """
    sentences = random.sample(SAMPLE_SENTENCES, min(num_sentences, len(SAMPLE_SENTENCES)))
    # If we need more sentences than we have samples, repeat some
    if num_sentences > len(SAMPLE_SENTENCES):
        additional_sentences = random.choices(
            SAMPLE_SENTENCES, 
            k=num_sentences - len(SAMPLE_SENTENCES)
        )
        sentences.extend(additional_sentences)
    
    # Shuffle to avoid having the same order
    random.shuffle(sentences)
    
    return " ".join(sentences)


def generate_summary(document: str, max_sentences: int = 3) -> str:
    """Generate a summary by selecting a few sentences from the document.
    
    Args:
        document: Source document to summarize
        max_sentences: Maximum number of sentences to include in summary
        
    Returns:
        A string containing the generated summary
    """
    sentences = document.split(". ")
    num_summary_sentences = min(max_sentences, len(sentences))
    
    # Select random sentences for the summary
    summary_sentences = random.sample(sentences, num_summary_sentences)
    
    return ". ".join(summary_sentences)


def generate_dataset_entry() -> Dict[str, str]:
    """Generate a single dataset entry with document and summary.
    
    Returns:
        Dictionary with document and summary fields
    """
    document = generate_document(random.randint(5, 15))
    summary = generate_summary(document, random.randint(1, 3))
    
    return {
        "document": document,
        "summary": summary
    }


def create_mock_dataset(output_dir: Path, num_examples: Dict[str, int]) -> None:
    """Create a mock indosum dataset with specified number of examples.
    
    Args:
        output_dir: Directory to write the dataset files
        num_examples: Dictionary with split names and number of examples
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split, count in num_examples.items():
        output_file = output_dir / f"{split}.jsonl"
        logger.info(f"Generating {count} examples for '{split}' split")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for _ in range(count):
                entry = generate_dataset_entry()
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
        logger.info(f"Saved {split} dataset to {output_file}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define output directory
    output_dir = Path("data/indosum")
    
    # Define number of examples for each split
    num_examples = {
        "train": 100,
        "validation": 20,
        "test": 20
    }
    
    # Create the mock dataset
    create_mock_dataset(output_dir, num_examples)
    logger.info("Mock dataset creation completed!")
