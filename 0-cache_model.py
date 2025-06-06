from transformers import BertForSequenceClassification, BertTokenizer

# Define the model name and cache directory
model_name = "bert-base-cased"

# Cache the model and tokenizer
def cache_model_components():
    print(f"Caching {model_name} model and tokenizer...")
    # Load and cache the pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    # Load and cache the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        model_name
    )
    print(f"{model_name} model and tokenizer successfully cached!")


if __name__ == "__main__":
    cache_model_components()