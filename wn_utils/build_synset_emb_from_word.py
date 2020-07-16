import argparse

import gensim
import numpy as np

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer 

# settings
ap = argparse.ArgumentParser()

ap.add_argument("synset_list", type=argparse.FileType('r'), help="list of synsets to calculate embeddings")
ap.add_argument("word_vec", type=str, help="pre-trained word vectors, word2vec txt format")
ap.add_argument("method", type=str, choices=["lemma_all_avg"], help="method of composing synset vecotrs from word vectors")
ap.add_argument("synset_vec", type=argparse.FileType('w'), help="save synset vectors, word2vec txt format")

args = ap.parse_args()

synset_set = set()
for line in args.synset_list:
    synset_set.add(line.strip())
#print(synset_set)
synset_vec = {}
synset_word_cnt = {}
print("loading word vectors from %s" % (args.word_vec))
word_emb = gensim.models.KeyedVectors.load_word2vec_format(args.word_vec, binary=False)
print("computing synset vectors")
emb_size = word_emb.vector_size
lemmatizer = WordNetLemmatizer()
for w in word_emb.vocab:
    for syn in wn.synsets(w):
        s = syn.name()
        if s not in synset_set:
            #print(s.name())
            continue

        #print(s)
        """
        if args.method == "lemma_head_avg":
            w_lemma = lemmatizer.lemmatize(w)
            synset_headword = s.split('.')[0]
            print(w, w_lemma, synset_headword)
            if synset_headword != w_lemma:
                continue
        """

        if s not in synset_vec:
            synset_vec[s] = np.zeros(emb_size)
            synset_word_cnt[s] = 0
        synset_vec[s] += word_emb[w]
        synset_word_cnt[s] += 1

args.synset_vec.write("%d %d\n" % (len(synset_vec), emb_size))
no_vec_cnt = 0
for s in synset_set:
    if s not in synset_vec:
        #print("!! no vector for synset %s" % (s))
        no_vec_cnt += 1
        continue
    synset_vec[s] /= synset_word_cnt[s]
    print("%s: from %d words" % (s, synset_word_cnt[s]))
    args.synset_vec.write(s + ' ')
    args.synset_vec.write(' '.join( map(str, list(synset_vec[s])) ) + '\n')
print("%d synsets are not assigned a vector" % (no_vec_cnt))
