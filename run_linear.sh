#!/bin/bash

python run.py \
--action train \
--data-file data_train.csv \
--param-file params.linear \
--exp-dir exp/grade_linear_allmarked
echo "Start testing model"
python run.py \
--action test \
--data-file data_dev.csv \
--exp-dir exp/grade_linear_allmarked
echo "Testing finished"

read -n 1
