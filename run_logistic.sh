#!/bin/bash

python run.py \
--action train \
--data-file data_train.csv \
--param-file params.logistic \
--exp-dir exp/logistic_7f_1
echo "Start testing model"
python run.py \
--action test \
--data-file data_dev.csv \
--exp-dir exp/logistic_7f_1
echo "Testing finished"
bash