#!/bin/bash

python3 ggnn_sparse.py --data_dir data/embed/APIReplacement/DateGetTime --valid_file valid_APIReplacement_DateGetTime_0.json --train_file train_APIReplacement_DateGetTime_0.json
python3 ggnn_sparse.py --data_dir data/embed/APIReplacement/DateGetTime --valid_file valid_APIReplacement_DateGetTime_1.json --train_file train_APIReplacement_DateGetTime_1.json
python3 ggnn_sparse.py --data_dir data/embed/APIReplacement/DateGetTime --valid_file valid_APIReplacement_DateGetTime_2.json --train_file train_APIReplacement_DateGetTime_2.json
python3 ggnn_sparse.py --data_dir data/embed/APIReplacement/DateGetTime --valid_file valid_APIReplacement_DateGetTime_3.json --train_file train_APIReplacement_DateGetTime_3.json
python3 ggnn_sparse.py --data_dir data/embed/APIReplacement/DateGetTime --valid_file valid_APIReplacement_DateGetTime_4.json --train_file train_APIReplacement_DateGetTime_4.json
