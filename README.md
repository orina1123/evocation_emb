# Asymmetric Synset Embeddings for Modeling Human Judgement of Evocation
- For each synset `s`, learn two vectors `emb_out(s)` and `emb_in(s)`.
- For every synset pair `(s1, s2)`, fit `w * dot(emb_out(s1), emb_in(s2)) + b` to human evocation score `y`, where `w` is non-negative (to avoid negative correlation).
- `dot(emb_out(s1), emb_in(s2))` can be different from `dot(emb_out(s2), emb_in(s1))`, so this model can learn asymmetric similarity judgement scores.  
- Load pre-trained synset embeddings to initialize `emb_out` and/or `emb_in` for better generalization to unseen pairs.

- Pass `-h` to each script to get detailed usage.

## Prepare Training Data
- Download the evocation dataset from:
https://wordnet.cs.princeton.edu/downloads.html
- Generate a TSV with synset pair scores, where each synset is identified by its `name`.
```
python synset_pair_avg_score.py /path/to/evocation/release-0.4/controlled.word-pos-sense /path/to/evocation/release-0.4/controlled.standard > controlled.standard.synset_pair_avg_score.tsv
```
### Dataset Property
- Correlation of Forward/Backward Evocation 
```
$ python forward_backward_corr.py controlled.standard.synset_pair_avg_score.tsv
Spearman's rho: 0.3849 (p-value: 1.057140e-251)
Pearson's r: 0.5590 (p-value: 0.000000e+00)
evaluated on 7164 synset pairs with human scores of both directions
```


## Pre-trained Synset Vectors
### Off-the-shelf
- AutoExtend

https://drive.google.com/drive/folders/0B6KTy_3y_sxXNXNIekVRWGtvVlE?usp=sharing

### Compose Synset Vectors with Word Vectors
- _lemma_all_avg_
vec(synset `s`) = avg (vec(word `w`) for all `w` if lemma(`w`) in `s`)
  * based on `glove.6B` https://nlp.stanford.edu/projects/glove/
  * word embedding file converted to word2vec txt format
```
python wn_utils/build_synset_emb_from_word.py controlled.standard.synset.list /path/to/glove.6B.50d.w2v_format.txt lemma_all_avg synset_vectors/from_word/glove.6B.50d--lemma_all_avg.synset.vec.txt
```

## Train Asymmetric Embeddings
- Note: the train/val/test split is fixed across different runs. However, model performance might differ from run to run due to multithreading.
### 20% testing, 20% validation
- Prefix: `ctrl.std.avg--vs0.2ts0.2--glove.6B--lemma_all_avg.d50.mul`
```
python keras_synset_emb_regression.py controlled.standard.synset_pair_avg_score.tsv 50 --init-out synset_vectors/from_word/glove.6B.50d--lemma_all_avg.synset.vec.txt --init-in synset_vectors/from_word/glove.6B.50d--lemma_all_avg.synset.vec.txt -mul -vs 0.2 -ts 0.2 -o synset_vectors/ctrl.std.avg--vs0.2ts0.2--glove.6B--lemma_all_avg.d50.mul | tee synset_vectors/ctrl.std.avg--vs0.2ts0.2--glove.6B--lemma_all_avg.d50.mul.log
```
### 20% testing, with specified # epochs
- Prefix: `ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul`
```
python keras_synset_emb_regression.py ../evocation/controlled.standard.synset_pair_avg_score.tsv 50 --init-out synset_vectors/from_word/glove.6B.50d--lemma_all_avg.synset.vec.txt --init-in synset_vectors/from_word/glove.6B.50d--lemma_all_avg.synset.vec.txt -mul -i 3 -ts 0.2 -o synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul | tee synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul.log
```

## Evaluation
### Symmetric Correlation (Correlation with Dot Scores Given by One Embedding)
```
python emb_cos_corr.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul.out.vec.txt -ts 0.2
python emb_cos_corr.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul.in.vec.txt -ts 0.2
```
### Asymmetric Correlation (Two Embeddings)
```
python asym_emb_dot_corr.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul -ts 0.2
```

## Query
### Score of a Synset Pair
```
python query_synset_pair_emb_dot_score.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul
```
### Top Synsets with Highest Forward/Backward Evocation Score
```
python query_synset_top_evocation_emb.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul
```

## All Experiments 
https://docs.google.com/spreadsheets/d/13si_sL6YTc7zA8ERsFbrbPulNhrEcYXMYaFDCx_798o/edit?usp=sharing
