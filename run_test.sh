#!/bin/bash


# Usage
# ./run_test.sh APIReplacement Mkdir

if [ ! $# == 2 ]; then 
	echo "Usage: $0 APIReplacement Mkdir"
	exit
fi

python3 ggnn_sparse.py --predict --data_dir data/embed/$1/$2 --valid_file test_$1_$2.json --train_file train_$1_$2.json
