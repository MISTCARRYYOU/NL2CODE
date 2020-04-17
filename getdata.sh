#!/bin/bash
set -e

# Create environment
conda env create --name NL2CODE --file spec-file.txt
conda init bash
conda activate NL2CODE

# Go to folder for data
cd ./dataset/data_conala

# Get the data
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
unzip conala-corpus-v1.1.zip
rm -r conala-corpus-v1.1.zip

# Preprocess data
cd ..
python preprocess_dataset.py
python json_to_seq2seq.py
