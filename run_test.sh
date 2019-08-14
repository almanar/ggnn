#!/bin/bash


# Usage
# ./run_test.sh APIReplacement Mkdir

# For OWASP
python3 ggnn_sparse.py --predict --data_dir data/embed/$1/$2 --valid_file test_$1_$2.json --train_file train_$1_$2.json
