import os
import pdb

# path = 'logs/0713_tpt_ViT-B-16_A_nump1_selectp1_0.005.txt'

# path = 'logs/otpt_0717_tpt_ViT-B-16_A_nump12_firstselectid5_0.005.txt'
path = 'logs/otpt_0723_tpt_ViT-B-16_A_nump12_select_selectid12_0.1%augs_threshold0.2_0.005.txt'

f = open(path, 'r')
lines = f.readlines()
f.close()

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


for line in lines:
    # pdb.set_trace()
    line_com = line.strip('\n').split(' ')
    if line_com[0] == 'Sample:':
        samples += 1
        entropy_before = float(line_com[4].strip(','))
        entropy_after = float(line_com[7].strip(','))
        pred_before = int(line_com[10].strip(','))
        pred_after = int(line_com[13].strip(','))
        target = int(line_com[15])
        if entropy_after <= entropy_before:
            if pred_before == target and pred_after == target:
                ent_lower_consis_c += 1
            elif pred_before != target and pred_after != target:
                ent_lower_consis_f += 1
            elif pred_before != target and pred_after == target:
                ent_lower_adapt_c += 1
            elif pred_before == target and pred_after != target:
                ent_lower_adapt_f += 1

        elif entropy_after > entropy_before:
            if pred_before == target and pred_after == target:
                ent_higher_consis_c += 1
            elif pred_before != target and pred_after != target:
                ent_higher_consis_f += 1
            elif pred_before != target and pred_after == target:
                ent_higher_adapt_c += 1
            elif pred_before == target and pred_after != target:
                ent_higher_adapt_f += 1

print("Total samples: ", samples)
print("With entropy lower: consistent correct: {}, consistent wrong: {}, correct after adapt: {}, wrong after adapt: {}".format(ent_lower_consis_c, ent_lower_consis_f, ent_lower_adapt_c, ent_lower_adapt_f))
print("With entropy higher: consistent correct: {}, consistent wrong: {}, correct after adapt: {}, wrong after adapt: {}".format(ent_higher_consis_c, ent_higher_consis_f, ent_higher_adapt_c, ent_higher_adapt_f))
