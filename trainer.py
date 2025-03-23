"""Training utilities for IndoBart model fine-tuning."""

import os
from typing import Dict, List, Optional, Any, Callable, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import EvalPrediction

from utils import CustomTrainingArguments, logger, postprocess_text


class SummarizationTrainer:
    """Trainer class for sequence-to-sequence summarization tasks."""
    
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        data_collator: Callable,
        training_args: CustomTrainingArguments,
        compute_metrics_fn: Callable[[EvalPrediction], Dict[str, float]],
    ):
        """
        Initialize the trainer with model, datasets, and training arguments.
        
        Args:
            model: Model to be trained
            tokenizer: Tokenizer for the model
            train_dataset: Dataset for training
            eval_dataset: Dataset for evaluation
            data_collator: Function to collate batch data
            training_args: Arguments for training setup
            compute_metrics_fn: Function to compute evaluation metrics
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.training_args = training_args
        self.compute_metrics_fn = compute_metrics_fn
        
        # Create HF Trainer instance
        self.trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )
    
    def train(self) -> Optional[Dict[str, float]]:
        """
        Run model training.
        
        Returns:
            Training results
        """
        logger.info("Starting training...")
        
        train_result = self.trainer.train()
        metrics = train_result.metrics
        self.trainer.save_model()
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        
        metrics = self.trainer.evaluate(
            max_length=self.training_args.generation_max_length,
            num_beams=self.training_args.generation_num_beams,
            metric_key_prefix="eval"
        )
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        return metrics
    
    def push_to_hub(self, model_id: Optional[str] = None) -> None:
        """
        Push the model to Hugging Face Hub.
        
        Args:
            model_id: Model identifier for the Hub
        """
        hub_model_id = model_id or self.training_args.hub_model_id
        
        if not hub_model_id:
            raise ValueError(
                "To push to the Hub, you need to specify a hub_model_id via"
                " --hub_model_id or when calling push_to_hub()"
            )
        
        logger.info(f"Pushing model to Hugging Face Hub as: {hub_model_id}")
        self.trainer.push_to_hub(hub_model_id=hub_model_id)
        logger.info("Model successfully pushed to Hub!")


def get_compute_metrics_fn(
    tokenizer: PreTrainedTokenizer,
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """
    Get the compute_metrics function for evaluation.
    
    Args:
        tokenizer: Tokenizer for decoding predictions
        
    Returns:
        Function that computes metrics from model predictions
    """
    import evaluate
    
    rouge_metric = evaluate.load("rouge")
    
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute ROUGE metrics for summarization.
        
        Args:
            eval_pred: Model predictions and labels
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        
        # Decode generated summaries
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Postprocess text for metric calculation
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        # Calculate ROUGE scores
        result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        
        # Extract the median scores
        result = {k: round(v * 100, 2) for k, v in result.items()}
        
        return result
    
    return compute_metrics
