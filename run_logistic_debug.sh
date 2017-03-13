#!/bin/bash

python run.py \
--action train \
--data-file data_train_sample.csv \
--param-file params.logistic \
--exp-dir exp/logistic_basic_test
echo "Start testing model"
python run.py \
--action test \
--data-file data_test_sample.csv \
--exp-dir exp/logistic_basic_test
echo "Testing finished"
bash