# BERT Fine-Tuning with LoRA for Sentiment Analysis

This project demonstrates how to fine-tune BERT using LoRA (Low-Rank Adaptation) for sentiment analysis on the IMDB movie reviews dataset. LoRA enables efficient fine-tuning by adding small adapter layers while keeping the original model weights frozen.

## 🎯 Project Overview

- **Model**: BERT-base-cased (110M parameters)
- **Task**: Binary sentiment classification (Positive/Negative)
- **Dataset**: IMDB movie reviews (currently using 2,000 samples)
- **Method**: LoRA (Low-Rank Adaptation) fine-tuning
- **Framework**: Hugging Face Transformers + PEFT

## 🏗️ Project Structure

```
/
├── 0-cache_model.py          # Cache BERT model and tokenizer
├── 1-prepare_data.py         # Data preprocessing and tokenization
├── 2-train_model.py          # LoRA fine-tuning training
├── 3-inference_base.py       # Base BERT inference
├── 3-inference_lora.py       # LoRA fine-tuned model inference
├── 4-evaluation.py           # Model comparison and evaluation
├── run_training.sh           # Training script with warning suppression
├── run_code.sh              # Basic run script with warning suppression
├── requirements.txt          # Python dependencies
├── data/                     # Datasets
│   ├── imdb_dataset.csv     # Raw IMDB dataset (63MB)
│   └── tokenized_data/
│       ├── train/           # Training set (70%)
│       ├── validation/      # Validation set (15%)
│       └── test/           # Test set (15%)
├── adapters/                # LoRA adapter weights
│   └── bert-base-cased-sentiment-lora/
│       └── final/
├── training_output/         # Training checkpoints and logs
├── .cache/                  # Cached models and tokenizers
└── utils/                   # Utility scripts
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Step 0: Cache the model (optional, speeds up subsequent runs)
python 0-cache_model.py

# Step 1: Prepare and tokenize data
python 1-prepare_data.py

# Step 2: Train the model with LoRA
python 2-train_model.py

# Step 3: Try basic inference with base and LoRA adapted models (optional)
python 3-inference_base.py
python 3-inference_lora.py

# Step 4: Evaluate both models
python 4-evaluate.py
```

## 📊 What You'll Get

The evaluation will compare:
- **Base BERT**: Pretrained model without fine-tuning
- **BERT + LoRA**: Fine-tuned model with LoRA adapters

Expected output:
```
Base BERT Accuracy:     0.4967 (49.67%)
BERT + LoRA Accuracy:   0.8833 (88.33%)
Improvement:            38.67 percentage points
Relative Improvement:   77.89%
```

## 🔧 Detailed Usage

### Data Preparation
```bash
python 1-prepare_data.py
```
- Downloads IMDB dataset
- Creates 70/15/15 train/validation/test splits
- Tokenizes text with BERT tokenizer
- Saves processed data to `data/tokenized_data/`

### Training
```bash
python 2-train_model.py
```
- Loads BERT-base-cased model
- Applies LoRA configuration (r=1, alpha=1, dropout=0.1)
- Trains for 25 epochs
- Saves adapter weights to `adapters/`

## ⚙️ Configuration

### LoRA Parameters
```python
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=1,                    # Rank of adaptation
    lora_alpha=1,          # Scaling factor
    lora_dropout=0.1       # Dropout rate
)
```

### Training Parameters
```python
training_args = TrainingArguments(
    num_train_epochs=25,
    eval_strategy="epoch",
    output_dir="training_output/bert-base-cased-sentiment-lora"
)
```

## 📈 Key Benefits of LoRA

1. **Efficiency**: Only ~150KB adapter weights vs ~400MB full model
2. **Speed**: Faster training with fewer parameters to update
3. **Memory**: Lower memory requirements during training
4. **Modularity**: Easy to swap different adapters for different tasks

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: BERT-base-cased (12 layers, 768 hidden size)
- **Task Head**: Linear classifier for binary classification
- **LoRA Layers**: Applied to query and value projection matrices

### Data Processing
- **Tokenization**: BERT WordPiece tokenizer
- **Max Length**: 512 tokens
- **Padding**: Dynamic padding to max length in batch
- **Labels**: 0 (Negative), 1 (Positive)

### Training Process
1. Load pretrained BERT model
2. Add LoRA adapters to attention layers
3. Freeze original model weights
4. Train only adapter parameters
5. Evaluate on validation set each epoch
6. Save best adapter weights

## 🔍 Evaluation Metrics

The evaluation script provides:
- **Accuracy**: Percentage of correct predictions
- **Absolute Improvement**: Percentage point difference
- **Relative Improvement**: Percentage change from baseline

## 📋 Requirements

See `requirements.txt` for exact versions:
- torch==2.7.0
- transformers==4.51.3
- datasets==3.6.0
- peft==0.15.2
- evaluate==0.4.3
- numpy==2.2.5
- scikit-learn==1.6.1
- pandas==2.2.3