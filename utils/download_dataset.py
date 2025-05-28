import os
from datasets import load_dataset
import pandas as pd

def load_and_save_imdb_dataset(num_samples=2000, output_dir="./data"):
    """
    Load the IMDB dataset and save a subset as a CSV file.
    
    Args:
        num_samples (int): Number of samples to include in the dataset
        output_dir (str): Directory to save the CSV file
    
    Returns:
        pd.DataFrame: Sampled dataset with balanced labels
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading IMDB dataset...")
    # Load the dataset with caching
    dataset = load_dataset("imdb")
    
    # Convert to pandas DataFrames
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    
    # Combine datasets
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"Full dataset size: {len(combined_df)}")
    print(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
    
    # Sample the dataset to get the desired number of samples
    # Use stratified sampling to maintain label balance
    if num_samples < len(combined_df):
        # Calculate samples per class to maintain balance
        samples_per_class = num_samples // 2
        
        # Sample equal amounts from each class
        positive_samples = combined_df[combined_df['label'] == 1].sample(
            n=samples_per_class, 
            random_state=42
        )
        negative_samples = combined_df[combined_df['label'] == 0].sample(
            n=samples_per_class, 
            random_state=42
        )
        
        # Combine and shuffle
        sampled_df = pd.concat([positive_samples, negative_samples], ignore_index=True)
        sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Sampled {len(sampled_df)} instances")
        print(f"Sampled label distribution: {sampled_df['label'].value_counts().to_dict()}")
    else:
        sampled_df = combined_df
        print(f"Using full dataset ({len(sampled_df)} instances)")
    
    # Save as a single CSV file
    output_path = os.path.join(output_dir, "imdb_dataset.csv")
    sampled_df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Final dataset size: {len(sampled_df)}")
    
    return sampled_df

if __name__ == "__main__":
    # Download and save 2000 instances of the IMDB dataset
    dataset_df = load_and_save_imdb_dataset(num_samples=2000) 