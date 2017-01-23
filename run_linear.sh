#!/bin/bash

python run.py --action train --data-file data_train_sample.csv --param-file params.linear --exp-dir exp/linear

python run.py --action test --data-file data_dev_sample.csv --exp-dir exp/linear
