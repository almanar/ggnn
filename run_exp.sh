#!/bin/bash

python3 ggnn_sparse.py --data_dir data/embed/APIReplacement/DateGetTime --valid_file valid_0.json --train_file train_0.json
python3 ggnn_sparse.py --data_dir data/embed/APIReplacement/DateGetTime --valid_file valid_1.json --train_file train_1.json
python3 ggnn_sparse.py --data_dir data/embed/APIReplacement/DateGetTime --valid_file valid_2.json --train_file train_2.json
python3 ggnn_sparse.py --data_dir data/embed/APIReplacement/DateGetTime --valid_file valid_3.json --train_file train_3.json
python3 ggnn_sparse.py --data_dir data/embed/APIReplacement/DateGetTime --valid_file valid_4.json --train_file train_4.json
