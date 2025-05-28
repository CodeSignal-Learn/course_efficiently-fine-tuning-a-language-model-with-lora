#!/bin/bash

# Python file to run
SCRIPT_NAME="4-evaluation.py"

# Run the script with unbuffered output and filter out specific warnings
python $SCRIPT_NAME 2>&1 | \
    grep -v "Some weights of BertForSequenceClassification were not initialized" | \
    grep -v "bitsandbytes was compiled without GPU support" | \
    grep -v "cadam32bit_grad_fp32" | \
    grep -v "You should probably TRAIN this model" | \
    grep -v "pin_memory.*not supported on MPS" 