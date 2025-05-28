#!/bin/bash

# Set Hugging Face cache directories for this project
# These environment variables tell Hugging Face libraries where to cache downloaded models and datasets

# HF_HOME: Main Hugging Face cache directory (parent directory for all HF cache)
export HF_HOME="$(pwd)/.cache"

# HF_HUB_CACHE: Directory for caching models and tokenizers from Hugging Face Hub
# Used by transformers library when loading models with from_pretrained()
export HF_HUB_CACHE="$(pwd)/.cache/models"

# HF_DATASETS_CACHE: Directory for caching datasets from Hugging Face Datasets
# Used by datasets library when loading datasets with load_dataset()
export HF_DATASETS_CACHE="$(pwd)/.cache/datasets"

echo "Environment variables set:"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE" 