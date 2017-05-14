#!/bin/bash

python run.py \
--action train \
--data-file data_train.csv \
--param-file params.svc \
--exp-dir exp/status_svc_linear_allmarked_100
echo "Start testing model"
python run.py \
--action test \
--data-file data_dev.csv \
--exp-dir exp/status_svc_linear_allmarked_100
echo "Testing finished"
bash