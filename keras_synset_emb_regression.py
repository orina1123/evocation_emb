import argparse

import numpy as np
np.random.seed(710)

from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot

import gensim

def build_emb_mat_from_gensim(id2syn, emb_size, gensim_emb):
    emb_mat = np.zeros((V, args.emb_size))
    OOV_cnt = 0
    for i, syn in enumerate(id2syn):
        if syn in gensim_emb:
            emb_mat[i] = gensim_emb[syn]
        else:
            print("no pre-trained vector for %s, using zero vector for init." % (syn))
            OOV_cnt += 1
    print("# synsets w/o pretrained vectors: %d" % (OOV_cnt))
    return emb_mat


# settings
ap = argparse.ArgumentParser()

ap.add_argument("pair_score_file", type=argparse.FileType('r'), help="TSV of synset pair - score [controlled.standard.synset_pair_avg_score.tsv]")
ap.add_argument("emb_size", type=int, help="embedding size / # dimensions")
ap.add_argument("--init-out", type=str, default=None, help="initialize emb_out with pre-trained synset embeddings")
ap.add_argument("--freeze-out", action="store_true", help="don't update emb_out during training")
ap.add_argument("--init-in", type=str, default=None, help="initialize emb_in with pre-trained synset embeddings")
ap.add_argument("--freeze-in", action="store_true", help="don't update emb_in during training")
ap.add_argument("-i", "--epochs", type=int, default=50, help="# training epochs")
ap.add_argument("-vs", "--val-split", type=float, default=0.0, help="split a portion of training data for validation")
ap.add_argument("-o", "--save-emb", type=str, default=None, help="save trained synset input/output embedding (word2vec txt format), *.[out|in].vec.txt")

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
#  shuffle (for rand. val. set)
indices = np.arange(X_out.shape[0])
np.random.shuffle(indices)
X_out = X_out[indices]
X_in = X_in[indices]
Y = Y[indices]

# build model
V = len(id2syn)
print("vocab. size:", V)
#  emb_out
out_inputs = Input(name="out_syn", shape=(1, ), dtype='int32')
#  pre-trained synset embeddings for emb_out
out_emb_W = None
if args.init_out is not None:
    print("loading pre-trained synset vectors for emb_out from %s" % (args.init_out))
    pre_emb = gensim.models.KeyedVectors.load_word2vec_format(args.init_out, binary=False)
    """
    emb_mat = np.zeros((V, args.emb_size))
    for i, syn in enumerate(id2syn):
        if syn in pre_emb:
            emb_mat[i] = pre_emb[syn]
        else:
            print("no pre-trained vector for %s, using zero vector for init." % (syn))
    out_emb_W = [emb_mat]
    """
    out_emb_W = [ build_emb_mat_from_gensim(id2syn, args.emb_size, pre_emb) ]
out_emb = Embedding(V, args.emb_size, name="out_emb", weights=out_emb_W, trainable=not args.freeze_out)(out_inputs)

#  emb_in
in_inputs = Input(name="in_syn", shape=(1, ), dtype='int32')
#  pre-trained synset embeddings for emb_in
in_emb_W = None
if args.init_in is not None:
    print("loading pre-trained synset vectors for emb_in from %s" % (args.init_in))
    if args.init_in != args.init_out:
        pre_emb = gensim.models.KeyedVectors.load_word2vec_format(args.init_in, binary=False)
    in_emb_W = [ build_emb_mat_from_gensim(id2syn, args.emb_size, pre_emb) ]
in_emb  = Embedding(V, args.emb_size, name="in_emb", weights=in_emb_W, trainable=not args.freeze_in)(in_inputs)

#  dot(out_emb, in_emb)
o = Dot(axes=2)([out_emb, in_emb])
o = Reshape((1,), input_shape=(1, 1))(o)
#o = Activation('sigmoid')(o)

model = Model(inputs=[out_inputs, in_inputs], outputs=o)
model.summary()
model.compile(loss='mse', optimizer='adam')

# train model
model.fit(x=[X_out, X_in], y=Y, epochs=args.epochs, batch_size=32, validation_split=args.val_split)

# save embeddings
if args.save_emb is not None:
    #out_emb_mat, in_emb_mat = model.get_weights()
    for layer in model.layers:
        #print(layer, layer.name)
        if layer.name == "out_emb":
            out_emb_mat = layer.get_weights()[0]
        elif layer.name == "in_emb":
            in_emb_mat = layer.get_weights()[0]
    print(out_emb_mat.shape, in_emb_mat.shape)
    print("saving trained embeddings to %s.{out,in}.vec.txt" % (args.save_emb))
    out_emb_path = args.save_emb + ".out.vec.txt"
    in_emb_path = args.save_emb + ".in.vec.txt"
    with open(out_emb_path, "w") as f_emb_out, open(in_emb_path, "w") as f_emb_in:
        for f, emb_mat in zip([f_emb_out, f_emb_in], [out_emb_mat, in_emb_mat]):
            f.write("%d %d\n" % (V, args.emb_size))
            for i in range(V):
                f.write(id2syn[i] + ' ')
                f.write(' '.join( map(str, list(emb_mat[i, :])) ) + '\n')
