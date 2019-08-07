#!/bin/bash


# Usage
# ./run_test.sh splitID dataPointFileName

# For OWASP
python3 ggnn_sparse.py --predict --data_dir data/embed/APIReplacement/DateGetTime --restore logs/train_$1_undertest.pickle --valid_file $2 --train_file $2
