from datasets import load_dataset
from transformers import AutoTokenizer
import os

# Define the model name for the tokenizer
model_name = "bert-base-cased"

# Define the number of samples and split ratios
num_samples = 2000
train_ratio = 0.7   # 70% for training
val_ratio = 0.15    # 15% for validation (used during training)
test_ratio = 0.15   # 15% for final testing (never seen during training)

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Select random samples from the training set
full_dataset = dataset["train"].shuffle(seed=42).select(range(num_samples))

# Calculate split indices
train_end = int(num_samples * train_ratio)
val_end = int(num_samples * (train_ratio + val_ratio))

# Create the three splits
train_data = full_dataset.select(range(train_end))
val_data = full_dataset.select(range(train_end, val_end))
test_data = full_dataset.select(range(val_end, num_samples))

print(f"Dataset splits:")
print(f"Training: {len(train_data)} samples")
print(f"Validation: {len(val_data)} samples") 
print(f"Test: {len(test_data)} samples")

# Initialize tokenizer from cache
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=".cache/models"
)

# Tokenize function
def tokenize_function(examples):
    # Tokenize the text and keep the labels
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,  # Limit sequence length
    )
    # Add labels to the tokenized output
    tokenized["labels"] = examples["label"]
    return tokenized

# Tokenize all datasets with smaller batch size
train_tokenized = train_data.map(
    tokenize_function,
    batched=True,
    batch_size=8,  # Reduced batch size
    remove_columns=["text"]  # Only remove text column, keep labels
)
val_tokenized = val_data.map(
    tokenize_function,
    batched=True,
    batch_size=8,  # Reduced batch size
    remove_columns=["text"]  # Only remove text column, keep labels
)
test_tokenized = test_data.map(
    tokenize_function,
    batched=True,
    batch_size=8,  # Reduced batch size
    remove_columns=["text"]  # Only remove text column, keep labels
)

# Create output directory if it doesn't exist
output_dir = "./data/tokenized_data"
os.makedirs(output_dir, exist_ok=True)

# Save the processed datasets
train_tokenized.save_to_disk(os.path.join(output_dir, "train"))
val_tokenized.save_to_disk(os.path.join(output_dir, "validation"))
test_tokenized.save_to_disk(os.path.join(output_dir, "test"))

print(f"\nDatasets saved to {output_dir}")