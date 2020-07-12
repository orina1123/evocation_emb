import argparse

import numpy as np
np.random.seed(710)

from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot

# settings
ap = argparse.ArgumentParser()

ap.add_argument("pair_score_file", type=argparse.FileType('r'), help="TSV of synset pair - score [controlled.standard.synset_pair_avg_score.tsv]")
ap.add_argument("emb_size", type=int, help="embedding size / # dimensions")
ap.add_argument("-i", "--epochs", type=int, default=50, help="# training epochs")
ap.add_argument("-o", "--save-emb", type=str, help="save trained synset input/output embedding (word2vec txt format), *.[out|in].vec.txt")

args = ap.parse_args()

# build vocab & convert synsets to ids
id2syn = []
syn2id = {}
X_out = []
X_in = []
Y = []
for line in args.pair_score_file:
    s1, s2, score_str = line.strip().split('\t')

    for s in s1, s2:
        if s not in syn2id:
            syn2id[s] = len(id2syn)
            id2syn.append(s)
    X_out.append(syn2id[s1])
    X_in.append(syn2id[s2])
    
    y = float(score_str)
    Y.append(y)
print(len(X_out), len(X_in), len(Y))
X_out = np.array(X_out)
X_in = np.array(X_in)
Y = np.array(Y)

# build model
V = len(id2syn)
print("vocab. size:", V)
#  emb_out
out_inputs = Input(name="out_syn", shape=(1, ), dtype='int32')
out_emb = Embedding(V, args.emb_size, name="out_emb")(out_inputs)

#  emb_in
in_inputs = Input(name="in_syn", shape=(1, ), dtype='int32')
in_emb  = Embedding(V, args.emb_size, name="in_emb")(in_inputs)
o = Dot(axes=2)([out_emb, in_emb])
o = Reshape((1,), input_shape=(1, 1))(o)
#o = Activation('sigmoid')(o)

#model = Model(inputs=[out_inputs, in_inputs], outputs=o)
model = Model(inputs={"out_syn": out_inputs, "in_syn": in_inputs}, outputs=o)
model.summary()
model.compile(loss='mse', optimizer='adam')

# train model
model.fit(x=[X_out, X_in], y=Y, epochs=args.epochs, batch_size=32)

# save embeddings
out_emb_mat, in_emb_mat = model.get_weights()
print("saving trained embeddings to %s.{out,in}.vec.txt" % (args.save_emb))
out_emb_path = args.save_emb + ".out.vec.txt"
in_emb_path = args.save_emb + ".in.vec.txt"
with open(out_emb_path, "w") as f_emb_out, open(in_emb_path, "w") as f_emb_in:
    for f, emb_mat in zip([f_emb_out, f_emb_in], [out_emb_mat, in_emb_mat]):
        f.write("%d %d\n" % (V, args.emb_size))
        for i in range(V):
            f.write(id2syn[i])
            f.write(' ')
            f.write(' '.join( map(str, list(emb_mat[i, :])) ))
            f.write('\n')
