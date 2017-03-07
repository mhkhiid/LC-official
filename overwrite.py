#!/usr/bin/python

import sys
import json

if len(sys.argv) != 4:
    raise RuntimeError("Usage: overwrite.py <param-file> <param-field> <param-value>")

param_file = sys.argv[1]
param_field = sys.argv[2]
param_value = sys.argv[3]

if param_field == 'alpha':
    param_value = float(param_value)

params = json.load(open(param_file, 'r'))
params['model_params'].update({param_field: param_value})
json.dump(params, open(param_file, 'w'))
