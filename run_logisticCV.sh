#!/bin/bash


python run.py \
--action train \
--data-file data_train.csv \
--param-file params.logisticCV \
--exp-dir exp/grade_logisticCV_allmarked
echo "Start testing."
python run.py \
--action test \
--data-file data_dev.csv \
--exp-dir exp/grade_logisticCV_allmarked
echo "Testing finished."

bash