import argparse

import gensim
import numpy as np
np.random.seed(710)
from scipy.stats import spearmanr, pearsonr

# settings
ap = argparse.ArgumentParser()

ap.add_argument("pair_score_file", type=argparse.FileType('r'), help="TSV of synset pair - score [controlled.standard.synset_pair_avg_score.tsv]")
#ap.add_argument("-ts", "--test-split", type=float, default=0.0, help="split a portion of training data for testing")

args = ap.parse_args()

# load human scores
pair_score = {}
for line in args.pair_score_file:
    s1, s2, score_str = line.strip().split('\t')
    score = float(score_str)
    pair_score[(s1, s2)] = score

# collect pairs with scores of both directions
forward_pair_set = set()
forward_score_list = []
backward_score_list = []
for (s1, s2) in pair_score:
    if (s2, s1) in pair_score and (s2, s1) not in forward_pair_set:
        forward_pair_set.add( (s1, s2) )
        forward_score_list.append(pair_score[(s1, s2)])
        backward_score_list.append(pair_score[(s2, s1)])

# compute correlation
rho, rho_p = spearmanr(forward_score_list, backward_score_list)
r, r_p = pearsonr(forward_score_list, backward_score_list)
print("Spearman's rho: %.4f (p-value: %e)" % (rho, rho_p))
print("Pearson's r: %.4f (p-value: %e)" % (r, r_p))

print("evaluated on %d synset pairs with human scores of both directions" % (len(forward_pair_set)))
