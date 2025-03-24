"""Model loading and configuration for IndoBart-v2 fine-tuning."""

from typing import Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_base import BatchEncoding

from utils import ModelArguments, logger

try:
    # Try to import IndoNLGTokenizer from indobenchmark
    from indobenchmark import IndoNLGTokenizer
except ImportError:
    # If not available, create a simple wrapper class that uses SentencePiece
    logger.info("IndoNLGTokenizer not found, using a custom implementation...")
    from pathlib import Path
    import os
    import sentencepiece as spm
    from transformers import BartTokenizer
    
    class IndoNLGTokenizer(PreTrainedTokenizer):
        """
        Custom implementation of the IndoNLGTokenizer for IndoBart models.
        Uses SentencePiece for tokenization, with a fallback to BartTokenizer.
        """
        
        vocab_files_names = {"vocab_file": "sentencepiece.bpe.model"}
        
        def __init__(
            self,
            vocab_file=None,
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            **kwargs
        ):
            super().__init__(
                bos_token=bos_token,
                eos_token=eos_token,
                sep_token=sep_token,
                cls_token=cls_token,
                unk_token=unk_token,
                pad_token=pad_token,
                mask_token=mask_token,
                **kwargs,
            )
            
            # Try to find the SentencePiece model file
            if vocab_file:
                self.vocab_file = vocab_file
            else:
                # Fallback to BartTokenizer
                self._tokenizer = BartTokenizer.from_pretrained(
                    "facebook/bart-base", 
                    bos_token=bos_token,
                    eos_token=eos_token,
                    sep_token=sep_token,
                    cls_token=cls_token,
                    unk_token=unk_token,
                    pad_token=pad_token,
                    mask_token=mask_token
                )
                
                # Try to download the model file
                try:
                    import huggingface_hub
                    # Download sentencepiece model
                    spm_path = huggingface_hub.hf_hub_download(
                        repo_id="indobenchmark/indobart-v2",
                        filename="sentencepiece.bpe.model",
                        cache_dir=kwargs.get("cache_dir", None)
                    )
                    self.vocab_file = spm_path
                except Exception as e:
                    logger.warning(f"Failed to download sentencepiece model: {e}")
                    self.vocab_file = None
            
            # Initialize SentencePiece if the model file is available
            if self.vocab_file and os.path.exists(self.vocab_file):
                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.Load(self.vocab_file)
            else:
                self.sp_model = None
                logger.warning(
                    "No SentencePiece model file found. Using BartTokenizer as fallback."
                )
        
        def get_vocab(self):
            if self.sp_model:
                vocab = {self.sp_model.id_to_piece(i): i for i in range(self.sp_model.get_piece_size())}
                return vocab
            else:
                return self._tokenizer.get_vocab()
                
        @property
        def vocab_size(self):
            if self.sp_model:
                return self.sp_model.get_piece_size()
            else:
                return self._tokenizer.vocab_size
        
        def _tokenize(self, text):
            if self.sp_model:
                return self.sp_model.encode(text, out_type=str)
            else:
                return self._tokenizer.tokenize(text)
        
        def _convert_token_to_id(self, token):
            if self.sp_model:
                return self.sp_model.piece_to_id(token)
            else:
                return self._tokenizer.convert_tokens_to_ids(token)
        
        def _convert_id_to_token(self, index):
            if self.sp_model:
                return self.sp_model.id_to_piece(index)
            else:
                return self._tokenizer.convert_ids_to_tokens(index)
        
        def convert_tokens_to_string(self, tokens):
            if self.sp_model:
                return self.sp_model.decode(tokens)
            else:
                return self._tokenizer.convert_tokens_to_string(tokens)
        
        def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True, **kwargs):
            """
            Converts a sequence of ids in a string, using the tokenizer and vocabulary
            with options to remove special tokens and clean up tokenization spaces.
            
            Args:
                token_ids: List of tokenized input ids
                skip_special_tokens: Whether to remove special tokens from decoded string
                clean_up_tokenization_spaces: Whether to clean up spaces from tokenization
                
            Returns:
                Decoded string
            """
            # Handle the fallback case first - use the underlying tokenizer
            if not self.sp_model:
                return self._tokenizer.decode(
                    token_ids, 
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    **kwargs
                )
                
            # For SentencePiece-based tokenization:
            if not isinstance(token_ids, list):
                token_ids = token_ids.tolist()
            
            # Filter out any token IDs that are out of range to avoid "piece id is out of range" error
            vocab_size = self.sp_model.get_piece_size()
            filtered_ids = []
            for id in token_ids:
                # Skip any token IDs that are out of range
                if 0 <= id < vocab_size:
                    filtered_ids.append(id)
                else:
                    logger.debug(f"Skipping out-of-range token id: {id} (vocab size: {vocab_size})")
            
            # If all tokens were filtered out, return empty string
            if not filtered_ids:
                return ""
                
            # Convert token ids to tokens
            tokens = [self._convert_id_to_token(i) for i in filtered_ids]
            
            # Filter special tokens if requested
            if skip_special_tokens:
                tokens = [token for token in tokens if token not in self.all_special_tokens]
                
            # Convert tokens to string
            text = self.convert_tokens_to_string(tokens)
            
            # Clean up tokenization spaces if requested
            if clean_up_tokenization_spaces:
                text = text.replace(" ##", "")
                text = text.replace("##", "")
                text = text.strip()
                
            return text
        
        def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
            """Save the tokenizer vocabulary to a directory."""
            if not self.sp_model:
                if hasattr(self._tokenizer, 'save_vocabulary'):
                    return self._tokenizer.save_vocabulary(save_directory, filename_prefix)
                else:
                    logger.warning("Fallback tokenizer does not implement save_vocabulary")
                    # Return empty tuple to avoid NotImplementedError
                    return ()
                
            if not os.path.isdir(save_directory):
                logger.error(f"Vocabulary path ({save_directory}) should be a directory")
                return ()
            
            out_vocab_file = os.path.join(
                save_directory, 
                (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
            )
            
            if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.exists(self.vocab_file):
                import shutil
                shutil.copyfile(self.vocab_file, out_vocab_file)
                logger.info(f"SentencePiece model saved to {out_vocab_file}")
            elif not os.path.exists(self.vocab_file):
                logger.warning(f"Cannot save vocabulary: source vocab file {self.vocab_file} does not exist")
                # Return empty tuple to avoid NotImplementedError
                return ()
            
            return (out_vocab_file,)


def load_model_and_tokenizer(
    model_args: ModelArguments,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[AutoModelForSeq2SeqLM, PreTrainedTokenizer]:
    """
    Load pretrained model and tokenizer from Hugging Face Hub.
    
    Args:
        model_args: Arguments for model loading
        device: Device to load the model onto
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_args.model_name}")
    
    # Load config and update parameters if needed
    config = AutoConfig.from_pretrained(
        model_args.model_name,
        cache_dir=model_args.cache_dir,
    )
    
    # Try to load tokenizer with AutoTokenizer first
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
        logger.info(f"Successfully loaded tokenizer with AutoTokenizer: {tokenizer.__class__.__name__}")
    except (ValueError, OSError) as e:
        logger.warning(f"Error loading tokenizer with AutoTokenizer: {e}")
        logger.info("Trying to load tokenizer using IndoNLGTokenizer...")
        
        # If AutoTokenizer fails, try directly with IndoNLGTokenizer
        tokenizer = IndoNLGTokenizer.from_pretrained(
            model_args.model_name,
            cache_dir=model_args.cache_dir,
        )
        logger.info("Successfully loaded tokenizer with IndoNLGTokenizer")
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name,
        from_tf=bool(".ckpt" in model_args.model_name),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    # Handle tokenizer BOS/EOS tokens (Bart-specific)
    if isinstance(model, BartForConditionalGeneration):
        if model.config.decoder_start_token_id is None and isinstance(config, BartConfig):
            model.config.decoder_start_token_id = tokenizer.bos_token_id
        if (
            model.config.decoder_start_token_id is None
            and not isinstance(config, BartConfig)
        ):
            model.config.decoder_start_token_id = model.config.bos_token_id
    
    # Set device (if using GPU)
    model = model.to(device)
    
    logger.info(f"Model loaded: {model.__class__.__name__}")
    logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    return model, tokenizer


def get_model_for_evaluation(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[AutoModelForSeq2SeqLM, PreTrainedTokenizer]:
    """
    Load a fine-tuned model and tokenizer for evaluation.
    
    Args:
        model_path: Path to the fine-tuned model directory
        device: Device to load the model onto
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading fine-tuned model from: {model_path}")
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (ValueError, OSError):
        logger.info("Falling back to IndoNLGTokenizer for loading tokenizer...")
        tokenizer = IndoNLGTokenizer.from_pretrained(model_path)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    
    return model, tokenizer
