#!/bin/bash

# Activate the virtual environment
source ./venv/bin/activate

cd scripts

if [ ! -f ../data/gene_dict.json ]; then
    echo "Extracting protein features"
    python protein_feature_extraction.py
fi
