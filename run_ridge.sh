#!/bin/bash

python run.py --action train --data-file data_train_sample.csv --param-file params.ridge --exp-dir exp/ridge
python run.py --action test --data-file data_dev_sample.csv --exp-dir exp/ridge
