#!/bin/bash

# Python file to run
SCRIPT_NAME="2-train_model.py"

# Set environment variables to suppress warnings at the source
export PYTHONWARNINGS="ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false

# Run Python with unbuffered output for real-time display
python -u $SCRIPT_NAME