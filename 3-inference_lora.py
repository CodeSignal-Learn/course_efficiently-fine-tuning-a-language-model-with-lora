from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch


def load_model():
    """Load the cached BERT model and tokenizer with LoRA configuration."""
    model_name = 'bert-base-cased'
    cache_dir = ".cache/models"
    adapter_path = "adapters/bert-base-cased-sentiment-lora/final"
    
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
    
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

    # Return the model and tokenizer
    return model, tokenizer


def predict(text, model, tokenizer, device='cpu'):
    """
    Perform inference on a single text input.
    
    Args:
        text (str): Input text to classify
        model: Loaded BERT model
        tokenizer: Loaded BERT tokenizer
        device (str): Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        tuple: (predicted_class, confidence_score, class_label)
    """
    # Prepare the input
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set model to evaluation mode
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        
    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Get the confidence score
    confidence = probabilities[0][predicted_class].item()
    
    # Map class to label (assuming 0 is negative, 1 is positive)
    class_label = "Negative" if predicted_class == 0 else "Positive"
    
    # Return the predicted class, confidence, and class label
    return predicted_class, confidence, class_label


def main():
    # Load the model and tokenizer
    model, tokenizer = load_model()
    
    # Example usage with the provided review
    test_text = """This stalk and slash turkey manages to bring nothing new to an increasingly stale genre. A masked killer stalks young, pert girls and slaughters them in a variety of gruesome ways, none of which are particularly inventive. It's not scary, it's not clever, and it's not funny. So what was the point of it?"""
    
    # Predict the class of the text
    predicted_class, confidence, class_label = predict(test_text, model, tokenizer)
    
    # Print the results
    print("\nReview Classification Results:")
    print(f"Input text:\n{test_text}\n")
    print(f"Prediction: {class_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Class ID: {predicted_class}")


if __name__ == "__main__":
    main() 