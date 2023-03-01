#python3 -m testing.select_fewshot_learning_electrode
import numpy as np
import glob
import json
import os

superlet_full_brain = "/storage/czw/self_supervised_seeg/full_brain_test_outs/superlet_large_pretrained/onset_finetuning/all_test_results/"
linear_full_brain =  "/storage/czw/self_supervised_seeg/outputs/2022-08-31/01-02-47/all_test_results/"

#collect linear results
def collect_results(results_path):
    results_files = glob.glob(os.path.join(results_path, "*", "*"))
    all_results = {}
    for path in results_files:
        subj = path.split("/")[-2]
        with open(path, "r") as f:
            subj_res = json.load(f)
            subj_res = subj_res[subj]
        all_results[subj] = subj_res
    return all_results

linear_results = collect_results(linear_full_brain)
superlet_results = collect_results(superlet_full_brain)

def rank_results(results):
    all_results = []
    all_results_d = {}
    for s in results:
        elecs = results[s]
        for e in elecs:
            roc = elecs[e]["roc_auc"]
            all_results.append(((s,e), roc))
    all_results = sorted(all_results, key=lambda x: -x[1])
    rank_results = [(x[0], i) for i,x in enumerate(all_results)]
    return sorted(rank_results, key=lambda x: x[0])

ranked_linear = rank_results(linear_results)
import pdb; pdb.set_trace()
ranked_superlet = rank_results(superlet_results)
assert len(ranked_linear)==len(ranked_superlet)
all_ranks = []
for i in range(len(ranked_linear)):
    se1, rank1 = ranked_superlet[i]
    se2, rank2 = ranked_linear[i]
    assert se1==se2
    all_ranks.append((se1, rank1+rank2, rank1, rank2))
print(sorted(all_ranks, key=lambda x:x[1])[:3])

#all_results = []
#for s in linear_results:
#    elecs = linear_results[s]
#    for e in elecs:
#        l_roc = elecs[e]["roc_auc"]
#        s_roc = superlet_results[s][e]["roc_auc"]
#        all_results.append(((s,e), l_roc+s_roc, l_roc, s_roc))
#        #all_results.append(((s,e), np.sqrt(l_roc*s_roc)))
#
#print(sorted(all_results, key=lambda x: x[1])[-1])
