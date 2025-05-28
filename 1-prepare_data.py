import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import os

# Define the model name for the tokenizer
model_name = "bert-base-cased"

# Define the number of samples and split ratios
num_samples = 2000
train_ratio = 0.7   # 70% for training
val_ratio = 0.15    # 15% for validation (used during training)
test_ratio = 0.15   # 15% for final testing (never seen during training)

# Load the IMDB dataset from local CSV file
df = pd.read_csv("data/imdb_dataset.csv")

print(f"Total samples in CSV: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")

# Shuffle and select samples
df_shuffled = df.sample(n=min(num_samples, len(df)), random_state=42).reset_index(drop=True)

print(f"Selected {len(df_shuffled)} samples for processing")

# Calculate split indices
train_end = int(len(df_shuffled) * train_ratio)
val_end = int(len(df_shuffled) * (train_ratio + val_ratio))

# Create the three splits
train_df = df_shuffled.iloc[:train_end]
val_df = df_shuffled.iloc[train_end:val_end]
test_df = df_shuffled.iloc[val_end:]

print(f"Dataset splits:")
print(f"Training: {len(train_df)} samples")
print(f"Validation: {len(val_df)} samples") 
print(f"Test: {len(test_df)} samples")

# Convert pandas DataFrames to Hugging Face Datasets
train_data = Dataset.from_pandas(train_df)
val_data = Dataset.from_pandas(val_df)
test_data = Dataset.from_pandas(test_df)

# Initialize tokenizer from cache
tokenizer = AutoTokenizer.from_pretrained(
    model_name
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
print("Tokenizing datasets...")
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
print("Saving tokenized datasets...")
train_tokenized.save_to_disk(os.path.join(output_dir, "train"))
val_tokenized.save_to_disk(os.path.join(output_dir, "validation"))
test_tokenized.save_to_disk(os.path.join(output_dir, "test"))

print(f"\nDatasets saved to {output_dir}")
print("Data preparation complete!") 