import evaluate
import numpy as np
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Load the tokenized datasets
train_dataset = load_from_disk("data/tokenized_data/train")
val_dataset = load_from_disk("data/tokenized_data/validation")  # Use validation set for training evaluation

# Define the output directory
final_adapter_dir = "adapters/bert-base-cased-sentiment-lora/final"     # Final adapter location

# Load the pre-trained BERT model for sequence classification with caching
model = BertForSequenceClassification.from_pretrained(
    'bert-base-cased', 
    num_labels=2
)

# Define the LoRA configuration with:
# - task_type: The type of task the model is trained for (sequence classification)
# - r: The rank of the LoRA matrices
# - lora_alpha: The scaling factor for the LoRA matrices
# - lora_dropout: The dropout rate for the LoRA matrices
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=1,
    lora_alpha=1,
    lora_dropout=0.1
)

# Create a LoRA-enabled version of the model
model = get_peft_model(model, lora_config)

# Print the model's trainable parameters
model.print_trainable_parameters()

# Load the accuracy metric
metric = evaluate.load("accuracy")


# Define the compute_metrics function to evaluate the model's performance
def compute_metrics(eval_pred):
    # Extract the logits and labels from the evaluation predictions
    logits, labels = eval_pred
    # Convert the logits to predictions by taking the argmax of the last dimension
    predictions = np.argmax(logits, axis=-1)
    # Compute the accuracy metric using the predictions and labels
    return metric.compute(predictions=predictions, references=labels)


# Define the training arguments with:
# - eval_strategy: The evaluation strategy (epochs)
# - num_train_epochs: The number of training epochs
training_args = TrainingArguments(
    eval_strategy="epoch",
    num_train_epochs=25,
    label_names=["labels"]  # Add label names to prevent warning
)

# Define the Trainer object with:
# - model: The model to train
# - args: The training arguments
# - train_dataset: The training dataset
# - eval_dataset: The validation dataset (NOT the test set)
# - compute_metrics: The function to compute the metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Use validation set for evaluation during training
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the LoRA adapter weights to final location
trainer.save_model(final_adapter_dir)

# Print the saved adapter path
print(f"LoRA adapter saved to: {final_adapter_dir}")