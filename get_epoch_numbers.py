import pdb

import numpy as np

import os

# textname = 'logs/otpt_0711_tpt_ViT-B-16_A_nump1_selectp1_0.005.txt'
textname = 'logs/otpt_0725_pred_consis_tpt_ViT-B-16_A_nump8_mean_selectid8_0.1%augs_threshold-100.0_0.005.txt'
textname = 'logs/otpt_0712_tpt_ViT-B-16_A_nump1_selectp1_0.005_0.txt'
# textname = 'logs/tpt_0725_tpt_ViT-B-16_A_nump12_select_selectid12_0.1%augs_threshold-100.0_0.005.txt'
textname = 'logs/otpt_0928_ensemptr_0.0_s6_50_tpt_4ctx_updatestep1_ViT-B-16_A_nump12_mc_selectid12_0.1%augs_threshold-100.0_varweight0.5_0.005.txt'
epoch_acc = [0]
all_corr = 0
with open(textname, 'r') as f:
    lines = f.readlines()
    for line in lines:
        # pdb.set_trace()
        if line[:5] == 'Test:':
            # pdb.set_trace()
            num_sam = line.split('[')[1].split('/')[0]
            # num_sam = content[2].split('/')[0]
            num_sam = int(num_sam) + 1
            acc = line.split('(')[2].split(')')[0]
            # pdb.set_trace()
            acc = float(acc)

            # pdb.set_trace()

            current_acc = abs(int(acc * num_sam / 100) - all_corr) / 200
            all_corr = int(acc * num_sam / 100)
            epoch_acc.append(current_acc)

    # pdb.set_trace()
    print(epoch_acc)
