#!/bin/bash
set -e

# Create environment
conda create --name NL2CODE --file spec-file.txt
source activate NL2CODE

# Go to folder for data
cd ./dataset/data_conala

# Get the data
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
unzip conala-corpus-v1.1.zip
rm -r conala-corpus-v1.1.zip

# Preprocess data
cd ..
python -u preprocess_dataset.py
python -u json_to_seq2seq.py
