from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_from_disk
import torch
from sklearn.metrics import accuracy_score


def load_model(adapter_path=None):
    """
    Load BERT model with optional LoRA fine-tuning.
    
    Args:
        use_lora (bool): Whether to apply LoRA configuration
        adapter_path (str): Path to LoRA adapter weights (required if use_lora=True)
    
    Returns:
        tuple: (model, tokenizer)
    """
    model_name = 'bert-base-cased'
    cache_dir = ".cache/models"
    
    # Load model configuration
    config = BertConfig.from_pretrained(
        model_name,
        num_labels=2,
        cache_dir=cache_dir
    )
    
    # Load base model with config
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    # Apply LoRA if an adapter path is provided
    if adapter_path:
        # Define the same LoRA configuration used in training
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=1,
            lora_alpha=1,
            lora_dropout=0.1
        )
        
        # Create LoRA model
        model = get_peft_model(model, lora_config)
        
        # Load the adapter weights
        model.load_adapter(adapter_path, adapter_name="default")

    # Return the model
    return model


def evaluate_model(model, test_dataset, model_name, device='cpu'):
    """
    Evaluate a model on the test dataset.
    
    Args:
        model: The model to evaluate
        test_dataset: The test dataset
        model_name: Name for display purposes
        device: Device to run on
    
    Returns:
        dict: Evaluation results with accuracy
    """

    # Set model to evaluation mode
    model.eval()

    # Move model to device
    model.to(device)
    
    # Initialize lists to store predictions and true labels
    predictions = []
    true_labels = []
    
    # Process each example in the test set
    for i, example in enumerate(test_dataset):     
        if i % 50 == 0:
            print(f"Processing example {i} of {len(test_dataset)}")

        # Prepare model inputs from the tokenized example
        inputs = {k: torch.tensor(example[k]).unsqueeze(0).to(device) for k in ['input_ids', 'attention_mask', 'token_type_ids'] if k in example}
        # Get the true labels
        true_label = example['labels']
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Get prediction
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # Store predictions and true labels
        predictions.append(predicted_class)
        true_labels.append(true_label)
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Return results
    results = {
        'model_name': model_name,
        'accuracy': accuracy
    }
    
    return results


def main():
    # Define the adapter directory
    adapter_path = "adapters/bert-base-cased-sentiment-lora/final"
    
    # Load the test dataset
    test_dataset = load_from_disk("data/tokenized_data/test")
    
    # Evaluate base model
    base_model = load_model()
    base_results = evaluate_model(base_model, test_dataset, "Base BERT")
    
    # Evaluate LoRA model
    lora_model = load_model(adapter_path=adapter_path)
    lora_results = evaluate_model(lora_model, test_dataset, "BERT + LoRA")
    
    # Print results
    print(f"Base BERT Accuracy:     {base_results['accuracy']:.4f} ({base_results['accuracy']*100:.2f}%)")
    print(f"BERT + LoRA Accuracy:   {lora_results['accuracy']:.4f} ({lora_results['accuracy']*100:.2f}%)")
    print(f"Improvement:            {(lora_results['accuracy'] - base_results['accuracy'])*100:.2f} percentage points")
    print(f"Relative Improvement:   {((lora_results['accuracy'] / base_results['accuracy']) - 1)*100:.2f}%")


if __name__ == "__main__":
    main() 