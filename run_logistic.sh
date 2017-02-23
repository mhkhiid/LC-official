#!/bin/bash

python run.py \
--action train \
--data-file data_train_sample.csv \
--param-file params.logistic \
--exp-dir exp/logistic

echo"Start testing model"
python run.py \
--action test \
--data-file data_dev_sample.csv \
--exp-dir exp/logistic

echo "Testing finished"