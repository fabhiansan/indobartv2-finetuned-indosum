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
            """
            Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary.
            We filter out of range token IDs which could cause piece id errors in SentencePiece.
            """
            if isinstance(token_ids, int):
                token_ids = [token_ids]
                
            if isinstance(token_ids, list):
                # Filter out of range token IDs to avoid "piece id out of range" errors
                vocab_size = self.sp_model.get_piece_size()
                filtered_ids = []
                for id in token_ids:
                    # Keep special tokens and valid SentencePiece tokens
                    if id in self.special_ids_to_tokens or (0 <= id < vocab_size):
                        filtered_ids.append(id)
                    else:
                        logger.debug(f"Filtering out token ID {id} which is out of range (0, {vocab_size})")
                
                # Convert filtered IDs to tokens
                tokens = self.convert_ids_to_tokens(filtered_ids, skip_special_tokens=skip_special_tokens)
                
                # Join tokens to form the output string
                text = "".join(tokens)
                
                # Apply SentencePiece post-processing (replace underscore with space, etc.)
                text = text.replace(" ", "").replace(SPIECE_UNDERLINE, " ")
                
                return text
            else:
                # Handle other input types by using the parent's decode method with our filtered token IDs
                outputs = super().decode(token_ids, skip_special_tokens=skip_special_tokens)
                return outputs.replace(' ', '').replace('▁', ' ')
    
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
                self._tokenizer = AutoTokenizer.from_pretrained(
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
