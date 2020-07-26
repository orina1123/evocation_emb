# Prepare Training Data
```
python synset_pair_avg_score.py /path/to/evocation/release-0.4/controlled.word-pos-sense /path/to/evocation/release-0.4/controlled.standard > controlled.standard.synset_pair_avg_score.tsv
```

# Pre-trained Synset Vectors
## Off-the-shelf
- AutoExtend

https://drive.google.com/drive/folders/0B6KTy_3y_sxXNXNIekVRWGtvVlE?usp=sharing

## Compose Synset Vectors with Word Vectors
- _lemma_all_avg_

vec(synset) = avg (vec(w) for all lemma(w) in synset)
```
python wn_utils/build_synset_emb_from_word.py controlled.standard.synset.list /path/to/glove.6B.50d.w2v_format.txt lemma_all_avg synset_vectors/from_word/glove.6B.50d--lemma_all_avg.synset.vec.txt
```


## Dataset Property
### Correlation of Forward/Backward Evocation 
```
$ python forward_backward_corr.py controlled.standard.synset_pair_avg_score.tsv
Spearman's rho: 0.3849 (p-value: 1.057140e-251)
Pearson's r: 0.5590 (p-value: 0.000000e+00)
evaluated on 7164 synset pairs with human scores of both directions
```

# Train Asymmetric Embeddings
```
python keras_synset_emb_regression.py controlled.standard.synset_pair_avg_score.tsv 50 --init-out synset_vectors/from_word/glove.6B.50d--lemma_all_avg.synset.vec.txt --init-in synset_vectors/from_word/glove.6B.50d--lemma_all_avg.synset.vec.txt -mul -vs 0.2 -ts 0.2 -o synset_vectors/ctrl.std.avg--vs0.2ts0.2--glove.6B--lemma_all_avg.d50.mul | tee synset_vectors/ctrl.std.avg--vs0.2ts0.2--glove.6B--lemma_all_avg.d50.mul.log
```

# Evaluation
## Symmetric Correlation (Correlation with Dot Scores Given by One Embedding)
```
python emb_cos_corr.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--vs0.2ts0.2--glove.6B--lemma_all_avg.d50.mul.out.vec.txt -ts 0.2
```
## Asymmetric Correlation (Two Embeddings)
```
python asym_emb_dot_corr.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--vs0.2ts0.2--glove.6B--lemma_all_avg.d50.mul -ts 0.2
```

# Query
## Score of a Synset Pair
```
python query_synset_pair_emb_dot_score.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--vs0.2ts0.2--glove.6B--lemma_all_avg.d50.mul
```
## Top Synsets with Highest Forward/Backward Evocation Score
```
python query_synset_top_evocation_emb.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--vs0.2ts0.2--glove.6B--lemma_all_avg.d50.mul
```
