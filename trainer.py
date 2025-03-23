"""Training utilities for IndoBart model fine-tuning."""

import os
from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np
import logging

from datasets import Dataset
from transformers import (
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments as TrainingArguments,
    AutoModelForSeq2SeqLM,
    EvalPrediction,
)

from rouge_score import rouge_scorer

from utils import CustomTrainingArguments


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
    
    def train(self):
        """
        Run model training.
        
        Returns:
            Training results
        """
        train_result = None
        if self.training_args.do_train:
            train_result = self.trainer.train()
            # Save the model
            self.trainer.save_model()
            
            # Save metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
            # Save state
            self.trainer.save_state()
        
        return train_result
    
    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Evaluation metrics
        """
        eval_results = None
        if self.training_args.do_eval:
            eval_results = self.trainer.evaluate()
            
            # Log and save metrics
            self.trainer.log_metrics("eval", eval_results)
            self.trainer.save_metrics("eval", eval_results)
        
        return eval_results
    
    def push_to_hub(self, model_id: Optional[str] = None):
        """
        Push the model to Hugging Face Hub.
        
        Args:
            model_id: Model identifier for the Hub
        """
        self.trainer.push_to_hub(model_id=model_id)


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
    # Initialize rouge scorer directly instead of using evaluate.load
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute ROUGE metrics for summarization.
        
        Args:
            eval_pred: Model predictions and labels
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Calculate ROUGE scores
        rouge_results = {}
        for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            scores = scorer.score(label, pred)
            for metric, score in scores.items():
                if f"{metric}_precision" not in rouge_results:
                    rouge_results[f"{metric}_precision"] = []
                    rouge_results[f"{metric}_recall"] = []
                    rouge_results[f"{metric}_fmeasure"] = []
                
                rouge_results[f"{metric}_precision"].append(score.precision)
                rouge_results[f"{metric}_recall"].append(score.recall)
                rouge_results[f"{metric}_fmeasure"].append(score.fmeasure)
        
        # Compute averages
        result = {}
        for key, values in rouge_results.items():
            result[key] = np.mean(values)
        
        return result
    
    return compute_metrics
