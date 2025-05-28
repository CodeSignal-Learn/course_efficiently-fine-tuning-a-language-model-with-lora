import os
from datasets import load_dataset
import pandas as pd

def load_and_save_imdb_dataset(cache_dir="./cache/datasets", output_dir="./data"):
    """
    Load the IMDB dataset and save it as a single CSV file.
    
    Args:
        cache_dir (str): Directory to cache the dataset
        output_dir (str): Directory to save the CSV file
    
    Returns:
        pd.DataFrame: Combined dataset with train/test split information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset with caching
    dataset = load_dataset("imdb", cache_dir=cache_dir)
    
    # Convert to pandas DataFrames and add split information
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    
    # Combine datasets
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Save as a single CSV file
    output_path = os.path.join(output_dir, "imdb_dataset.csv")
    combined_df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Total dataset size: {len(combined_df)}")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return combined_df

if __name__ == "__main__":
    # Example usage
    dataset_df = load_and_save_imdb_dataset() 