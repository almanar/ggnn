#!/bin/bash


# Usage
# ./run_test.sh splitID dataPointFileName

# For OWASP
python3 ggnn_sparse.py --data_dir data/singles/owasp --restore logs/owasp-$1-undertest.pickle --valid_file $2 --train_file $2

# For RW-Rand
# python3 ggnn_sparse.py --data_dir data/singles --restore logs/rw-slice-rand-1-undertest.pickle --valid_file $1 --train_file $1
# python3 ggnn_sparse.py --data_dir data/singles --restore logs/rw-slice-rand-2-undertest.pickle --valid_file $1 --train_file $1
# python3 ggnn_sparse.py --data_dir data/singles --restore logs/rw-slice-rand-3-undertest.pickle --valid_file $1 --train_file $1
# python3 ggnn_sparse.py --data_dir data/singles --restore logs/rw-slice-rand-4-undertest.pickle --valid_file $1 --train_file $1
# python3 ggnn_sparse.py --data_dir data/singles --restore logs/rw-slice-rand-5-undertest.pickle --valid_file $1 --train_file $1

# For RW-PW
# python3 ggnn_sparse.py --data_dir data/singles --restore logs/rw-slice-pw-1-undertest.pickle --valid_file $1 --train_file $1
# python3 ggnn_sparse.py --data_dir data/singles --restore logs/rw-slice-pw-2-undertest.pickle --valid_file $1 --train_file $1
# python3 ggnn_sparse.py --data_dir data/singles --restore logs/rw-slice-pw-3-undertest.pickle --valid_file $1 --train_file $1
# python3 ggnn_sparse.py --data_dir data/singles --restore logs/rw-slice-pw-4-undertest.pickle --valid_file $1 --train_file $1
# python3 ggnn_sparse.py --data_dir data/singles --restore logs/rw-slice-pw-5-undertest.pickle --valid_file $1 --train_file $1