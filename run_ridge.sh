#!/bin/bash


python run.py \
--action train \
--data-file data_train.csv \
--param-file params.ridge \
--exp-dir exp/status_ridge_allmarked_10
echo "Start testing."
python run.py \
--action test \
--data-file data_dev.csv \
--exp-dir exp/status_ridge_allmarked_10
echo "Testing finished."
bash