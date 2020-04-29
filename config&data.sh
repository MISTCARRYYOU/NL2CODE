#!/bin/bash
set -e

# Set absolute path
export PYTHONPATH="$PWD/dataset:$PWD/dataset/data_conala:$PWD/model"
echo "$PYTHONPATH"

# Create environment
conda env create -f NL2CODE.yml
eval "$(conda shell.bash hook)"
conda activate NL2CODE
conda info --envs

# Get the data
echo "download dataset"
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
unzip conala-corpus-v1.1.zip -d ./dataset/data_conala
rm -r conala-corpus-v1.1.zip
echo "done"

# Preprocess data
python -m get_data \
    --mode train
