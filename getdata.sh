#!/bin/bash
set -e

# Create environment
conda create --name myenv --file spec-file.txt


# Go to folder for data
cd ./dataset/data_conala

# Get the data
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
unzip conala-corpus-v1.1.zip
rm -r conala-corpus-v1.1.zip