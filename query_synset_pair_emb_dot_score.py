import argparse
import gensim
import numpy as np

# settings
ap = argparse.ArgumentParser()

ap.add_argument("pair_score_file", type=argparse.FileType('r'), help="TSV of synset pair - score [controlled.standard.synset_pair_avg_score.tsv]")
ap.add_argument("emb_txt_prefix", type=str, help="path prefix to trained out/in synset embeddings, *.[out|in].vec.txt")

args = ap.parse_args()

# load human scores
pair_score = {}
for line in args.pair_score_file:
    s1, s2, score_str = line.strip().split('\t')
    score = float(score_str)
    pair_score[(s1, s2)] = score

# load embeddings
out_emb = gensim.models.KeyedVectors.load_word2vec_format(args.emb_txt_prefix + ".out.vec.txt", binary=False)
in_emb = gensim.models.KeyedVectors.load_word2vec_format(args.emb_txt_prefix + ".in.vec.txt", binary=False)

# accept queries
while True:
    input_str = input("Please enter two synsets, separated by a blank> ")
    s1, s2 = input_str.strip().split(' ')
    
    if (s1, s2) in pair_score:
        evo_score = pair_score[(s1, s2)]
        print("[human]\t%s -- evocation --> %s\t%f" % (s1, s2, evo_score))
    else:
        print("[human]\t%s -- evocation --> %s\t[not found]" % (s1, s2))
    pred_evo_score = np.dot(out_emb[s1], in_emb[s2])
    print("[pred]\t%s -- evocation --> %s\t%f" % (s1, s2, pred_evo_score))
    
    if (s2, s1) in pair_score:
        rev_evo_score = pair_score[(s2, s1)]
        print("[human]\t%s <-- evocation -- %s\t%f" % (s1, s2, rev_evo_score))
    else:
        print("[human]\t%s <-- evocation -- %s\t[not found]" % (s1, s2))
    pred_rev_evo_score = np.dot(out_emb[s2], in_emb[s1])
    print("[pred]\t%s <-- evocation -- %s\t%f" % (s1, s2, pred_rev_evo_score))
    print("")
