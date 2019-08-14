#!/bin/bash

# ./run_exp.sh APIReplacement Mkdir 0

if [ ! $# == 3 ]; then 
	echo "Usage: $0 APIReplacement Mkdir 0"
	exit
fi

python3 ggnn_sparse.py --data_dir data/embed/$1/$2 --valid_file valid_$1_$2_$3.json --train_file train_$1_$2_$3.json
