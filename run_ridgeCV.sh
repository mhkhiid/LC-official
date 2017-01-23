#!/bin/bash

python run.py --action train --data-file data_train_sample.csv --param-file params.ridgeCV --exp-dir exp/ridgeCV
python run.py --action test --data-file data_dev_sample.csv --exp-dir exp/ridgeCV
