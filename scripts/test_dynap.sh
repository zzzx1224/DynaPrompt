#!/bin/bash

data_root=$1
testsets=$2
date=$3
num_prompts=$4
selection_p=$5
lr=$6
nctx=$7
seed=$8
arch=$9
bs=64
ctx_init=a_photo_of_a

python ./dynap_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--lr ${lr} \
--seed ${seed} \
--n_ctx ${nctx} \
--tpt --ctx_init ${ctx_init} --log_date ${date} \
--num_prompts ${num_prompts} \
--onlinetpt \
--selection_p ${selection_p}