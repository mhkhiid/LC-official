#!/bin/bash


python run.py \
--action train \
--data-file data_train.csv \
--param-file params.ridgeCV \
--exp-dir exp/ridgeCV_7f
echo "Start testing model."

python run.py \
--action test \
--data-file data_dev.csv \
--exp-dir exp/ridgeCV_7f

echo "Testing finished."
bash
