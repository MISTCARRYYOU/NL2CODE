#!/bin/bash
set -e

# Set absolute path
export PYTHONPATH="$PWD/dataset:$PWD/dataset/data_conala:$PWD/model"
echo "$PYTHONPATH"

# Create environment
conda env create -f NL2CODE.yml
conda activate NL2CODE

# Get the data
echo "download dataset"
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
unzip ./dataset/data_conala/conala-corpus-v1.1.zip
rm -r conala-corpus-v1.1.zip
echo "done"

# Preprocess data
python -m preprocess_dataset.py
python -m json_to_seq2seq.py
