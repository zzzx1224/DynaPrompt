import os
import pdb
import numpy as np
# path = 'logs/0713_tpt_ViT-B-16_A_nump1_selectp1_0.005.txt'

# path = 'logs/otpt_0717_tpt_ViT-B-16_A_nump12_firstselectid5_0.005.txt'
path = 'logs/otpt_0731_doublep0_1_tpt_ViT-B-16_A_nump12_selectmc_selectid12_0.1%augs_threshold-100.0_varweight0.5_0.005.txt'

outputfile = 'logs/statistic_analyses' + path.split('/')[1]
pdb.set_trace()

f = open(path, 'r')
lines = f.readlines()
f.close()

fw = open(outputfile, 'w')

ent_lower = 0
adapt_c = 0
adapt_f = 0
ent_higher = 0
samples = 0

ent_lower_adapt_c = 0
ent_lower_adapt_f = 0
ent_higher_adapt_c = 0
ent_higher_adapt_f = 0
ent_lower_consis_f = 0
ent_lower_consis_c = 0
ent_higher_consis_f = 0
ent_higher_consis_c = 0

original_ent = []
augmented_ent = []
first6_ent = []
# original_conf = []
augmented_conf_dif = []
first6_conf_dif = []
tt_o_ent = []
tt_a_ent = []
tt_a_conf_dif = []
tt_f_conf_dif = []

tf_o_ent = []
tf_a_ent = []
tf_a_conf_dif = []
tf_f_conf_dif = []

ft_o_ent = []
ft_a_ent = []
ft_a_conf_dif = []
ft_f_conf_dif = []

ff_o_ent = []
ff_a_ent = []
ff_a_conf_dif = []
ff_f_conf_dif = []


for line in lines:
    # pdb.set_trace()
    line_com = line.strip('\n').split(' ')
    if line_com[0] == "Original" and line_com[1] == "entropy:":
        original_ent = float(line_com[2].strip(','))
        augmented_ent = float(line_com[5].strip(','))
    elif line_com[0] == "confidence":
        augmented_conf_dif = float(line_com[2].strip(','))
        first6_conf_dif = float(line_com[5].strip(','))

    elif line_com[0] == 'Sample:':
        samples += 1
        entropy_before = float(line_com[4].strip(','))
        entropy_after = float(line_com[7].strip(','))
        pred_before = int(line_com[10].strip(','))
        pred_after = int(line_com[13].strip(','))
        target = int(line_com[15])

        if pred_before == target and pred_after == target:
            tt_o_ent.append(entropy_before)
            tt_a_ent.append(augmented_ent)
            tt_a_conf_dif.append(augmented_conf_dif)
            tt_f_conf_dif.append(first6_conf_dif)
        elif pred_before != target and pred_after != target:
            ff_o_ent.append(entropy_before)
            ff_a_ent.append(augmented_ent)
            ff_a_conf_dif.append(augmented_conf_dif)
            ff_f_conf_dif.append(first6_conf_dif)
        elif pred_before != target and pred_after == target:
            ft_o_ent.append(entropy_before)
            ft_a_ent.append(augmented_ent)
            ft_a_conf_dif.append(augmented_conf_dif)
            ft_f_conf_dif.append(first6_conf_dif)
        elif pred_before == target and pred_after != target:
            tf_o_ent.append(entropy_before)
            tf_a_ent.append(augmented_ent)
            tf_a_conf_dif.append(augmented_conf_dif)
            tf_f_conf_dif.append(first6_conf_dif)

pdb.set_trace()

print("Total samples: ", samples)
print("With entropy lower: consistent correct: {}, consistent wrong: {}, correct after adapt: {}, wrong after adapt: {}".format(ent_lower_consis_c, ent_lower_consis_f, ent_lower_adapt_c, ent_lower_adapt_f))
print("With entropy higher: consistent correct: {}, consistent wrong: {}, correct after adapt: {}, wrong after adapt: {}".format(ent_higher_consis_c, ent_higher_consis_f, ent_higher_adapt_c, ent_higher_adapt_f))
