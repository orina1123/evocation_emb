import argparse

import numpy as np
np.random.seed(710)

from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Reshape, Activation, Input
from keras.layers import Subtract, Lambda
from keras.layers.merge import Dot
import tensorflow as tf
from tensorflow.keras.constraints import unit_norm, max_norm, non_neg
from tensorflow.keras import regularizers
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint

import gensim

class RetroRegularizer(regularizers.Regularizer):

    def __init__(self, alpha, ref_mat):
        self.alpha = alpha
        self.ref_mat = ref_mat

    def __call__(self, x):
        return self.alpha * tf.reduce_sum(tf.square(x-ref_mat))

    def get_config(self):
        return {'alpha': self.alpha}

def build_emb_mat_from_gensim(id2syn, emb_size, gensim_emb, init_emb_mat=None):
    if init_emb_mat is not None:
        emb_mat = init_emb_mat
    else:
        emb_mat = np.zeros((V, args.emb_size))
    OOV_cnt = 0
    for i, syn in enumerate(id2syn):
        if syn in gensim_emb:
            emb_mat[i] = gensim_emb[syn]
        else:
            print("no pre-trained vector for %s, using given initialization" % (syn))
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
ap.add_argument("-sigm", "--sigmoid", action="store_true", help="apply a sigmoid (logistic) function after dot product, scores need to be scaled to [0, 1]")
ap.add_argument("-mul", "--multiply-layer", action="store_true", help="apply a non-negative multiplication layer after dot product")
ap.add_argument("-l2", "--l2-reg", type=float, default=0, help="L2 regularization factor")
ap.add_argument("-rr", "--retro-reg", type=float, default=0, help="retrofitting regularization term factor")
ap.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="")
ap.add_argument("-i", "--epochs", type=int, default=50, help="# training epochs")
ap.add_argument("-vs", "--val-split", type=float, default=0.0, help="split a portion of training data for validation")
ap.add_argument("-ts", "--test-split", type=float, default=0.0, help="split a portion of training data for testing")
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
if args.test_split > 0.0:
    p = int(indices.shape[0]*args.test_split)
    X_out_test, X_out_train = X_out[indices[:p]], X_out[indices[p:]]
    X_in_test, X_in_train = X_in[indices[:p]], X_in[indices[p:]]
    Y_test, Y_train = Y[indices[:p]], Y[indices[p:]]
else:
    X_out_train = X_out[indices]
    X_in_train = X_in[indices]
    Y_train = Y[indices]

# build model
V = len(id2syn)
print("vocab. size:", V)
#  emb_out
out_inputs = Input(name="out_syn", shape=(1, ), dtype='int32')
out_emb = Embedding(V, args.emb_size, name="out_emb", trainable=not args.freeze_out, embeddings_regularizer=regularizers.l2(args.l2_reg))
out_embbed = out_emb(out_inputs)
#  pre-trained synset embeddings for emb_out
if args.init_out is not None:
    print("loading pre-trained synset vectors for emb_out from %s" % (args.init_out))
    pre_emb = gensim.models.KeyedVectors.load_word2vec_format(args.init_out, binary=False)
    out_emb_W = [ build_emb_mat_from_gensim(id2syn, args.emb_size, pre_emb, init_emb_mat=out_emb.get_weights()[0]) ]
    out_emb.set_weights(out_emb_W)
    if args.retro_reg > 0.0:
        out_emb_reg = Embedding(V, args.emb_size, name="out_emb_reg", trainable=False, weights=out_emb_W)
        #out_emb_reg.set_weights(out_emb_W)
        out_embbed_reg = out_emb_reg(out_inputs)


#  emb_in
in_inputs = Input(name="in_syn", shape=(1, ), dtype='int32')
in_emb = Embedding(V, args.emb_size, name="in_emb", trainable=not args.freeze_in, embeddings_regularizer=regularizers.l2(args.l2_reg))
in_embbed = in_emb(in_inputs)
#  pre-trained synset embeddings for emb_in
if args.init_in is not None:
    print("loading pre-trained synset vectors for emb_in from %s" % (args.init_in))
    if args.init_in != args.init_out:
        pre_emb = gensim.models.KeyedVectors.load_word2vec_format(args.init_in, binary=False)
    in_emb_W = [ build_emb_mat_from_gensim(id2syn, args.emb_size, pre_emb, init_emb_mat=in_emb.get_weights()[0]) ]
    in_emb.set_weights(in_emb_W) 
    if args.retro_reg > 0.0:
        in_emb_reg = Embedding(V, args.emb_size, name="in_emb_reg", trainable=False, weights=in_emb_W)
        #in_emb_reg.set_weights(in_emb_W)
        in_embbed_reg = out_emb_reg(in_inputs)


#  dot(out_emb, in_emb)
o = Dot(axes=2)([out_embbed, in_embbed])
o = Reshape((1,), input_shape=(1, 1))(o)
if args.multiply_layer:
    o = Dense(1, kernel_constraint=non_neg())(o)
if args.sigmoid:
    o = Activation('sigmoid')(o)

if args.retro_reg > 0.0:
    out_emb_err = Subtract()([out_embbed, out_embbed_reg])
    #out_emb_err = Lambda(lambda x: x**2)(out_emb_err)
    in_emb_err = Subtract()([in_embbed, in_embbed_reg])
    model = Model(inputs=[out_inputs, in_inputs], outputs=[o, out_emb_err, in_emb_err])
else:
    model = Model(inputs=[out_inputs, in_inputs], outputs=o)
model.summary()
optm = Adam(lr=args.learning_rate)
if args.retro_reg > 0.0:
    model.compile(loss=["mse", "mse", "mse"], loss_weights=[1.0, args.retro_reg, args.retro_reg], optimizer=optm, metrics=["mse"])
else:
    model.compile(loss='mse', optimizer=optm, metrics=["mse", "mae"])

# train model
cb = []
if args.save_emb is not None:
    best_ckpt_path = args.save_emb + ".model.best.hdf5"
else:
    best_ckpt_path = "/tmp/evocation_emb.model.best.hdf5"
if args.val_split > 0.0:
    saveBestModel = ModelCheckpoint(best_ckpt_path, monitor='val_mse', verbose=0, save_best_only=True, mode='min')
    cb.append(saveBestModel)
if args.retro_reg > 0.0:
    _Y_reg = np.zeros( (Y_train.shape[0], args.emb_size) )
    model.fit(x=[X_out_train, X_in_train], y=[Y_train, _Y_reg, _Y_reg], epochs=args.epochs, batch_size=32, validation_split=args.val_split, callbacks=cb)
else:
    model.fit(x=[X_out_train, X_in_train], y=Y_train, epochs=args.epochs, batch_size=32, validation_split=args.val_split, callbacks=cb)

# go to best checkpoint
if args.val_split > 0.0:
    model.load_weights(best_ckpt_path)
else:
    model.save_weights(best_ckpt_path)

# evaluate model
if args.test_split > 0.0:
    if args.retro_reg > 0.0:
        _Y_reg = np.zeros( (Y_test.shape[0], args.emb_size) )
        loss, mse, mae = model.evaluate([X_out_test, X_in_test], [Y_test, _Y_reg, _Y_reg])
    else:
        loss, mse, mae = model.evaluate([X_out_test, X_in_test], Y_test)
    print("Test loss: %.4f, MSE: %.4f, MAE: %.4f" % (loss, mse, mae))

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
