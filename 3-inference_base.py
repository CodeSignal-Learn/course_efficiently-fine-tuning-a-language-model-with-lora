from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import torch


def load_model():
    """Load the base BERT model and tokenizer without LoRA."""
    model_name = 'bert-base-cased'
    cache_dir = ".cache/models"
    
    # Load model configuration
    config = BertConfig.from_pretrained(
        model_name,
        num_labels=2,
        cache_dir=cache_dir
    )
    
    # Load base model with config (no LoRA)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
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
    # Load the base model and tokenizer (no fine-tuning)
    model, tokenizer = load_model()
    
    # Example usage with the provided review
    test_text = """This stalk and slash turkey manages to bring nothing new to an increasingly stale genre. A masked killer stalks young, pert girls and slaughters them in a variety of gruesome ways, none of which are particularly inventive. It's not scary, it's not clever, and it's not funny. So what was the point of it?"""
    
    # Predict the class of the text
    predicted_class, confidence, class_label = predict(test_text, model, tokenizer)
    
    # Print the results
    print("\nBase BERT Classification Results (No Fine-tuning):")
    print(f"Input text:\n{test_text}\n")
    print(f"Prediction: {class_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Class ID: {predicted_class}")
    print("\nNote: This is using the base BERT model without any fine-tuning.")
    print("The predictions may not be meaningful for sentiment analysis.")


if __name__ == "__main__":
    main() 