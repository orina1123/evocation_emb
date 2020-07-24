import argparse
import gensim
import numpy as np

# settings
ap = argparse.ArgumentParser()

ap.add_argument("pair_score_file", type=argparse.FileType('r'), help="TSV of synset pair - score [controlled.standard.synset_pair_avg_score.tsv]")
ap.add_argument("emb_txt_prefix", type=str, help="path prefix to trained out/in synset embeddings/models, *.[out|in].vec.txt / *.")

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
# load keras model
#TODO

# accept queries
while True:
    input_str = input("Please specify one synset> ")
    s = input_str.strip()

    if s not in out_emb:
        print("! synset %s not found in vocab" % (s))
        continue
    
    print("--[ forward evocation ]-->")
    synset_scores_list = [] # (synset t, human_score, dot(out_emb[s], in_emb[t]), model_pred_score)
    for t in in_emb.vocab:
        if t == s:
            continue
        human_score = None
        if (s, t) in pair_score:
            human_score = pair_score[(s, t)]
        dot_score = np.dot(out_emb[s], in_emb[t])
        synset_scores_list.append( (t, human_score, dot_score) )
    for (t, human_score, dot_score) in sorted(synset_scores_list, key=lambda p: p[-1], reverse=True)[:20]:
        print("", t, "\t[human]:", human_score, "\t[out*in]:", dot_score)

    print("")
