"""Model loading and configuration for IndoBart-v2 fine-tuning."""

from typing import Optional, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import (
    PaddingStrategy,
    TensorType,
    is_tf_available,
    is_torch_available,
    logging,
    to_py_obj,
)

from utils import ModelArguments, logger

try:
    # Try to import IndoNLGTokenizer from indobenchmark
    from indobenchmark import IndoNLGTokenizer
except ImportError:
    # If not available, create a custom implementation based on the official indobenchmark-toolkit
    logger.info("IndoNLGTokenizer not found, using a custom implementation based on indobenchmark-toolkit...")
    from pathlib import Path
    import os
    import sentencepiece as spm
    from typing import Dict, List, Optional, Tuple, Union
    
    VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

    PRETRAINED_VOCAB_FILES_MAP = {
        "vocab_file": {
            "indobenchmark/indobart": "https://huggingface.co/indobenchmark/indobart/resolve/main/sentencepiece.bpe.model",
            "indobenchmark/indogpt": "https://huggingface.co/indobenchmark/indogpt/resolve/main/sentencepiece.bpe.model",
            "indobenchmark/indobart-v2": "https://huggingface.co/indobenchmark/indobart-v2/resolve/main/sentencepiece.bpe.model"
        }
    }

    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
        "indobenchmark/indobart": 768,
        "indobenchmark/indogpt": 768,
        "indobenchmark/indobart-v2": 768
    }

    SPIECE_UNDERLINE = "▁"

    # Define type aliases
    TextInput = str
    PreTokenizedInput = List[str]
    EncodedInput = List[int]
    TextInputPair = Tuple[str, str]
    PreTokenizedInputPair = Tuple[List[str], List[str]]
    EncodedInputPair = Tuple[List[int], List[int]]
    
    class IndoNLGTokenizer(PreTrainedTokenizer):
        """
        IndoNLGTokenizer for IndoBart and IndoGPT models.
        Based on the implementation from the indobenchmark-toolkit repository with additional
        safety checks for out-of-range token IDs.
        """
        vocab_files_names = VOCAB_FILES_NAMES
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        model_input_names = ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels']
        input_error_message = "text input must of type `str` (single example), `List[str]` (batch of examples)."

        def __init__(
            self,
            vocab_file,
            decode_special_token=True,
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            additional_special_tokens=[],
            **kwargs
        ):
            # Load the SentencePiece model
            self.sp_model = spm.SentencePieceProcessor()
            try:
                self.sp_model.Load(str(vocab_file))
            except Exception as e:
                logger.error(f"Error loading SentencePiece model: {e}")
                logger.info("Trying to download the SentencePiece model from HuggingFace...")
                try:
                    import huggingface_hub
                    repo_id = "indobenchmark/indobart-v2"
                    vocab_file = huggingface_hub.hf_hub_download(repo_id, filename="sentencepiece.bpe.model")
                    self.sp_model.Load(str(vocab_file))
                except Exception as download_err:
                    logger.error(f"Failed to download SentencePiece model: {download_err}")
                    raise ValueError("Could not load or download SentencePiece model")
            
            self.vocab_file = vocab_file
            self.decode_special_token = decode_special_token
            self.model_max_length = 1024
            
            # HACK: These tokens were added by fairseq but don't seem to be actually used when duplicated in the actual
            # sentencepiece vocabulary (this is the case for <s> and </s>
            self.special_tokens_to_ids = {
                "[javanese]": 40000, 
                "[sundanese]": 40001, 
                "[indonesian]": 40002,
                "<mask>": 40003
            }
            self.special_ids_to_tokens = {v: k for k, v in self.special_tokens_to_ids.items()}
            
            # Store Language token ID
            self.javanese_token = '[javanese]'
            self.javanese_token_id = 40000
            self.sundanese_token = '[sundanese]'
            self.sundanese_token_id = 40001
            self.indonesian_token = '[indonesian]'
            self.indonesian_token_id = 40002
            
            super().__init__(
                vocab_file=vocab_file,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                sep_token=sep_token,
                cls_token=cls_token,
                pad_token=pad_token,
                mask_token=mask_token,
                additional_special_tokens=additional_special_tokens,
                **kwargs,
            )
            self.special_token_ids = [
                self.bos_token_id, self.eos_token_id, self.sep_token_id, self.cls_token_id, 
                self.unk_token_id, self.pad_token_id, self.mask_token_id,
                self.javanese_token_id, self.sundanese_token_id, self.indonesian_token_id
            ]

        def __len__(self):
            return max(self.special_ids_to_tokens) + 1
        
        def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
        ) -> List[int]:
            """
            Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
            special tokens using the tokenizer ``prepare_for_model`` method.
            """
            if already_has_special_tokens:
                return super().get_special_tokens_mask(
                    token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                )

            if token_ids_1 is None:
                return [1] + ([0] * len(token_ids_0)) + [1]
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

        @property
        def vocab_size(self):
            return 4 + len(self.sp_model)

        def get_vocab(self):
            vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
            vocab.update(self.added_tokens_encoder)
            return vocab

        def _tokenize(self, text: str) -> List[str]:
            return self.sp_model.encode(text.lower(), out_type=str)
        
        def convert_ids_to_tokens(
            self, ids: Union[int, List[int]], skip_special_tokens: bool = False
        ) -> Union[str, List[str]]:
            """
            Converts a single index or a sequence of indices to tokens.
            Args:
                ids: The token id or ids to convert
                skip_special_tokens: Whether to skip special tokens
            Returns:
                The decoded token(s)
            """
            if isinstance(ids, int):
                return self._convert_id_to_token(ids, skip_special_tokens=skip_special_tokens)
            
            tokens = []
            for index in ids:
                if skip_special_tokens and index in self.all_special_ids:
                    continue
                if index not in self.added_tokens_decoder or index in self.special_tokens_to_ids:
                    tokens.append(self._convert_id_to_token(index, skip_special_tokens=skip_special_tokens))                
                else:
                    tokens.append(self.added_tokens_decoder[index])
            return tokens
        
        def _convert_token_to_id(self, token):
            """ Converts a token (str) in an id using the vocab. """
            if token in self.special_tokens_to_ids:
                return self.special_tokens_to_ids[token]
            return self.sp_model.PieceToId(token)
        
        def _convert_id_to_token(self, index, skip_special_tokens=False):
            """Converts an index (integer) in a token (str) using the vocab."""
            if skip_special_tokens and index in self.special_token_ids:
                return ''
                
            if index in self.special_ids_to_tokens:
                return self.special_ids_to_tokens[index]
            
            # Check if the index is within the valid range for the SentencePiece model
            vocab_size = self.sp_model.get_piece_size()
            if index < 0 or index >= vocab_size:
                logger.warning(f"Token ID {index} is out of range (0, {vocab_size}). Using UNK token instead.")
                return self.unk_token
            
            # Convert valid index to token
            try:
                token = self.sp_model.IdToPiece(index)
                if '<0x' in token:
                    char_rep = chr(int(token[1:-1], 0))
                    if char_rep.isprintable():
                        return char_rep
                return token
            except Exception as e:
                logger.warning(f"Error converting token ID {index} to token: {e}. Using UNK token instead.")
                return self.unk_token
        
        def __getstate__(self):
            state = self.__dict__.copy()
            state["sp_model"] = None
            return state

        def __setstate__(self, d):
            self.__dict__ = d

            # for backward compatibility
            if not hasattr(self, "sp_model_kwargs"):
                self.sp_model_kwargs = {}

            self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
            self.sp_model.Load(self.vocab_file)

        def decode(self, token_ids, skip_special_tokens=False, **kwargs):
            """Decode the given token IDs to a string, handling out-of-range token IDs."""
            try:
                if token_ids is None:
                    return ""
                
                # Convert to list if we got a tensor
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                
                # Get the vocab size
                vocab_size = self.sp_model.get_piece_size()
                
                # Check for invalid IDs
                invalid_ids = [id for id in token_ids if id >= vocab_size or id < 0]
                if invalid_ids:
                    logger.warning(f"Found {len(invalid_ids)} invalid token IDs out of {len(token_ids)}")
                    logger.warning(f"First few invalid IDs: {invalid_ids[:5]}")
                
                # Filter out token IDs that are out of range
                filtered_ids = []
                for id in token_ids:
                    if 0 <= id < vocab_size:
                        filtered_ids.append(id)
                    else:
                        logger.warning(f"Filtered out token ID {id} (vocab size is {vocab_size})")
                
                # If all tokens were filtered, use the UNK token
                if not filtered_ids and token_ids:
                    logger.warning("All token IDs were invalid, using UNK token instead")
                    filtered_ids = [self.unk_token_id]
                
                # Log summary of filtering
                if len(filtered_ids) != len(token_ids):
                    logger.info(f"Filtered {len(token_ids) - len(filtered_ids)} out of {len(token_ids)} token IDs")
                
                # Decode the filtered token IDs
                text = self._decode_with_spmodel(filtered_ids, skip_special_tokens)
                logger.info(f"Successfully decoded to text: '{text[:50]}...' (first 50 chars)")
                return text
            except Exception as e:
                logger.error(f"Error in decode method: {e}")
                if not token_ids:
                    return ""
                try:
                    # Last-resort fallback: decode token by token
                    logger.warning("Attempting fallback token-by-token decoding")
                    result = []
                    for id in token_ids:
                        try:
                            if 0 <= id < self.sp_model.get_piece_size():
                                piece = self.sp_model.id_to_piece(id)
                                result.append(piece)
                            else:
                                logger.warning(f"Skipping invalid token ID {id}")
                        except:
                            logger.warning(f"Error decoding token ID {id}, skipping")
                    
                    text = "".join(result)
                    logger.info(f"Fallback decoding successful: '{text[:50]}...' (first 50 chars)")
                    return text
                except Exception as fallback_error:
                    logger.error(f"Fallback decoding also failed: {fallback_error}")
                    return ""  # Return empty string as last resort
        
        def _decode_with_spmodel(self, token_ids, skip_special_tokens):
            # Convert token ids to tokens
            tokens = [self._convert_id_to_token(i) for i in token_ids]
            
            # Filter special tokens if requested
            if skip_special_tokens:
                tokens = [token for token in tokens if token not in self.all_special_tokens]
                
            # Convert tokens to string
            text = self.convert_tokens_to_string(tokens)
            
            # Clean up tokenization spaces if requested
            text = text.replace(" ", "").replace(SPIECE_UNDERLINE, " ")
            
            return text
        
        def convert_tokens_to_string(self, tokens):
            if self.sp_model:
                return self.sp_model.decode(tokens)
            else:
                return self._tokenizer.convert_tokens_to_string(tokens)
        
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
