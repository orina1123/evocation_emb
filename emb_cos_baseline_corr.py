import argparse

import gensim
import numpy as np
from scipy.stats import spearmanr, pearsonr

# settings
ap = argparse.ArgumentParser()

ap.add_argument("pair_score_file", type=argparse.FileType('r'), help="TSV of synset pair - score [controlled.standard.synset_pair_avg_score.tsv]")
ap.add_argument("synset_vec", type=str, help="synset vectors in word2vec format (need -bin for binary)")
ap.add_argument("-bin", "--binary", action="store_true", help="embedding is in word2vec bin format")

args = ap.parse_args()

# load human scores
pair_list = []
human_score_list = []
for line in args.pair_score_file:
    s1, s2, score_str = line.strip().split('\t')
    pair_list.append((s1, s2))
    human_score_list.append(float(score_str))

# load synset embedding & compute cosine similarity
print("loading synset vectors from %s" % (args.synset_vec))
emb = gensim.models.KeyedVectors.load_word2vec_format(args.synset_vec, binary=args.binary)
emb_cos_list = []
OOV_cnt = 0
for (s1, s2) in pair_list:
    if s1 in emb and s2 in emb:
        cos_sim = emb.similarity(s1, s2)
    else:
        cos_sim = 0.0
        OOV_cnt += 1
    emb_cos_list.append(cos_sim)

# compute correlation
rho, rho_p = spearmanr(emb_cos_list, human_score_list)
r, r_p = pearsonr(emb_cos_list, human_score_list)
print("Spearman's rho: %.4f (p-value: %e)" % (rho, rho_p))
print("Pearson's r: %.4f (p-value: %e)" % (r, r_p))

print("# pairs with OOV:", OOV_cnt)
