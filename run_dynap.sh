#!/bin/bash

data_root='/path/to/data' # your data path
dataset='A'  # I/A/R/V/S
logdate='250129'  # name of the log file
num_p=12  # number of online prompts
selection_p=0.1
lr=0.005
ntx=4
seed=6
arch='ViT-B/16'  #e.g., 'RN50' or 'ViT-B/16'

sh ./scripts/test_dynap.sh ${data_root} ${dataset} ${logdate} ${num_p} ${selection_p} ${lr} ${ntx} ${seed} ${arch}
