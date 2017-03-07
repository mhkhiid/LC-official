#!/bin/bash

for alpha in 0.01 0.1 1 10 100 1000 10000; do

  python overwrite.py params.logisticRidge alpha $alpha

  python run.py --action train --data-file data_train_sample.csv --param-file params.logisticRidge --exp-dir exp/logistic_ridge_$alpha

  python run.py --action test --data-file data_test_sample.csv --exp-dir exp/logistic_ridge_$alpha

done
