#!/bin/bash

python3 ggnn_sparse.py --data_dir data/embed/$1/$2 --valid_file valid_$1_$2_$3.json --train_file train_$1_$2_$3.json
