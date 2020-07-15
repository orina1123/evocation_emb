import argparse

import gensim

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer 

# settings
ap = argparse.ArgumentParser()

ap.add_argument("synset_list", type=argparse.FileType('r'), help="list of synsets to calculate embeddings")
ap.add_argument("word_vec", type=str, help="pre-trained word vectors, word2vec txt format")
ap.add_argument("method", type=str, choices=["lemma_head_avg", "lemma_all_avg"], help="method of composing synset vecotrs from word vectors")
ap.add_argument("synset_vec", type=argparse.FileType('w'), help="save synset vectors, word2vec txt format")

args = ap.parse_args()

synset_set = set()
for line in args.pair_file:
    synset_set.add(line.strip())

synset_vec = {}
synset_word_cnt = {}
word_emb = gensim.models.KeyedVectors.load_word2vec_format(args.word_vec, binary=False)
emb_size = word_emb.vector_size
lemmatizer = WordNetLemmatizer()
for w in word_emb:
    for s in wn.synsets(w):
        if s not in synset_set:
            continue

        if args.method == "lemma_head_avg":
            w_lemma = lemmatizer.lemmatize(w)
            if s.name.split('.')[0] != w_lemma:
                continue

        if s not in synset_vec:
            synset_vec[s] = np.zeros(emb_size)
            synset_word_cnt[s] = 0
        synset_vec[s] += word_emb
        synset_word_cnt[s] += 1

args.synset_vec.write("%d %d\n" % (len(synset_vec), emb_size))
for s in synset_set:
    if s not in synset_vec:
        print("!! no vector for synset %s" % (s))
        continue
    synset_vec[s] /= synset_word_cnt[s]
    args.synset_vec.write(s + ' ')
    args.synset_vec.write(' '.join( map(str, list(synset_vec[s])) ) + '\n')

