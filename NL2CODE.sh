#!/bin/bash
set -e

git clone https://github.com/Jonor127-OP/NL2CODE.git

cd Desktop/NL2CODE/dataset/data_conala


# Get the data
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
unzip conala-corpus-v1.1.zip