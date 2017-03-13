#!/bin/bash


python run.py \
--action train \
--data-file data_train.csv \
--param-file params.ridge \
--exp-dir exp/ridge_7f_0.1
echo "Start testing."
python run.py \
--action test \
--data-file data_dev.csv \
--exp-dir exp/ridge_7f_0.1
echo "Testing finished."
bash