# IndoBart-v2 Fine-tuning for Indonesian Summarization

This project fine-tunes the [indobenchmark/indobart-v2](https://huggingface.co/indobenchmark/indobart-v2) model on the [SEACrowd/indosum](https://huggingface.co/datasets/SEACrowd/indosum) dataset for Indonesian text summarization.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── data_processing.py
├── model.py
├── trainer.py
├── evaluate.py
├── evaluate_checkpoints.py
├── utils.py
└── main.py
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Hugging Face token (needed to push model to Hub):
```bash
export HF_TOKEN=your_huggingface_token
```

## Usage

Run the complete pipeline:
```bash
python main.py --output_dir ./output --num_train_epochs 3 --push_to_hub --hub_model_id your-username/indobart-indosum
```

Or run individual steps:
```bash
# Train only
python main.py --do_train --output_dir ./output

# Evaluate only
python main.py --do_eval --model_path ./output

# Push to Hub
python main.py --model_path ./output --push_to_hub --hub_model_id your-username/indobart-indosum
```

### Batch Checkpoint Evaluation

To evaluate all checkpoints in a directory and generate comprehensive reports:
```bash
python evaluate_checkpoints.py \
  --checkpoints_dir ./output \
  --report_dir ./reports \
  --output_dir ./reports \
  --dataset_use_mock False
```

This utility will:
1. Find all checkpoint directories in the specified folder
2. Evaluate each checkpoint against the validation dataset
3. Generate individual metrics JSON files for each checkpoint
4. Create a summary report (CSV and HTML) comparing all checkpoints

## Configuration

Edit the parameters in `main.py` or pass them as command-line arguments:

- `--model_name`: HF model ID (default: "indobenchmark/indobart-v2")
- `--dataset_name`: HF dataset name (default: "SEACrowd/indosum")
- `--output_dir`: Directory to save model checkpoints
- `--learning_rate`: Learning rate (default: 5e-5)
- `--batch_size`: Batch size for training (default: 4)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--max_input_length`: Maximum input sequence length (default: 512)
- `--max_target_length`: Maximum target sequence length (default: 128)
- `--push_to_hub`: Whether to push model to Hugging Face Hub
- `--hub_model_id`: Hugging Face Hub model ID for uploading

## Performance Metrics

The model is evaluated using ROUGE scores, which measure the overlap between generated summaries and reference summaries.

## License

This project is available under the MIT License.
