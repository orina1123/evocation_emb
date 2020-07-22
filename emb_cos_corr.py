import argparse

import gensim
import numpy as np
np.random.seed(710)
from scipy.stats import spearmanr, pearsonr

# settings
ap = argparse.ArgumentParser()

ap.add_argument("pair_score_file", type=argparse.FileType('r'), help="TSV of synset pair - score [controlled.standard.synset_pair_avg_score.tsv]")
ap.add_argument("synset_vec", type=str, help="synset vectors in word2vec format (need -bin for binary)")
ap.add_argument("-bin", "--binary", action="store_true", help="embedding is in word2vec bin format")
ap.add_argument("-ts", "--test-split", type=float, default=0.0, help="split a portion of training data for testing")

args = ap.parse_args()

# load human scores
pair_list = []
human_score_list = []
for line in args.pair_score_file:
    s1, s2, score_str = line.strip().split('\t')
    pair_list.append((s1, s2))
    human_score_list.append(float(score_str))

if args.test_split > 0.0:
    indices = np.arange(len(pair_list))
    np.random.shuffle(indices)
    p = int(indices.shape[0]*args.test_split)
    pair_list = list(np.array(pair_list)[indices[:p]])
    human_score_list = list(np.array(human_score_list)[indices[:p]])

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

print("evaluated on %d synset pairs" % (len(pair_list)))
print("# pairs with OOV:", OOV_cnt)
