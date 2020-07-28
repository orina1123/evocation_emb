# Asymmetric Synset Embeddings for Modeling Human Judgement of Evocation
- For each synset `s`, learn two vectors `emb_out(s)` and `emb_in(s)`.
- If we learn the synset embeddings only from the collected evocation scores, they can hardly generalize to unseen pairs due to a lack of semantic knowledge as appropriate "inductive bias". There will be too many "solutions" that "fit" the given scores.
- Load pre-trained synset embeddings to initialize `emb_out` and/or `emb_in` for better generalization to unseen pairs.
- For every synset pair `(s1, s2)`, fit `w * dot(emb_out(s1), emb_in(s2)) + b` to human evocation score `y`, where `w` is non-negative (to avoid negative correlation). The purpose of `w` and `b` is to transform dot product values of pre-trained vectors to better fit the range of the evocation scores. 
- `dot(emb_out(s1), emb_in(s2))` can be different from `dot(emb_out(s2), emb_in(s1))`, so this model can learn asymmetric similarity judgement scores.  

- Pass `-h` to each script to get detailed usage.

- As most models have been added to this repository, after generating the training pair file `controlled.standard.synset_pair_avg_score.tsv`, you can jump to [the Evaluation section](#evaluation).

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
- [AutoExtend](https://www.aclweb.org/anthology/P15-1173/)

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
- Note: a predicted score is the dot product of `emb_out` and `emb_in`, without including trained `w` and `b`. Therefore, only the relative magnitude matters.
### Score of a Synset Pair
```
python query_synset_pair_emb_dot_score.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul
```
- Example
```
$ python query_synset_pair_emb_dot_score.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul
Please enter two synsets, separated by a blank> social.a.02 urban.a.01
[human] social.a.02  -- evocation -->  urban.a.01       3.584300
[pred]  social.a.02  -- evocation -->  urban.a.01       20.416164
[human] social.a.02  <-- evocation --  urban.a.01       2.852500
[pred]  social.a.02  <-- evocation --  urban.a.01       17.990353

Please enter two synsets, separated by a blank> effective.a.01 complete.a.01
[human] effective.a.01  -- evocation -->  complete.a.01 1.448433
[pred]  effective.a.01  -- evocation -->  complete.a.01 6.641746
[human] effective.a.01  <-- evocation --  complete.a.01 [not found]
[pred]  effective.a.01  <-- evocation --  complete.a.01 3.795130

Please enter two synsets, separated by a blank> complex.a.01 equivocal.a.01
[human] complex.a.01  -- evocation -->  equivocal.a.01  2.273067
[pred]  complex.a.01  -- evocation -->  equivocal.a.01  7.453255
[human] complex.a.01  <-- evocation --  equivocal.a.01  [not found]
[pred]  complex.a.01  <-- evocation --  equivocal.a.01  3.556570

Please enter two synsets, separated by a blank> rich.a.01 retired.s.01
[human] rich.a.01  -- evocation -->  retired.s.01       0.725483
[pred]  rich.a.01  -- evocation -->  retired.s.01       5.312593
[human] rich.a.01  <-- evocation --  retired.s.01       [not found]
[pred]  rich.a.01  <-- evocation --  retired.s.01       4.071751
```

### Top Synsets with Highest Forward/Backward Evocation Score
```
python query_synset_top_evocation_emb.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul
```
- Example
```
$ python query_synset_top_evocation_emb.py controlled.standard.synset_pair_avg_score.tsv synset_vectors/ctrl.std.avg--i3-ts0.2--glove.6B--lemma_all_avg.d50.mul
Please specify one synset> trust.v.01
trust.v.01 ['trust', 'swear', 'rely', 'bank']
==================================================
 --[ forward evocation ]-->
==================================================
[out*in]: 11.405758  [human]: None        trust.v.01 --> personal.a.01 ['personal']
[out*in]: 10.954998  [human]:  2.793850   trust.v.01 --> religious.a.02 ['religious']
[out*in]: 10.886404  [human]: None        trust.v.01 --> political.a.01 ['political']
[out*in]: 10.630712  [human]: None        trust.v.01 --> value.n.06 ['value']
[out*in]: 10.264768  [human]: None        trust.v.01 --> value.n.03 ['value', 'economic_value']
[out*in]: 10.164742  [human]: None        trust.v.01 --> want.v.02 ['want', 'need', 'require']
[out*in]: 10.111633  [human]: None        trust.v.01 --> authority.n.02 ['authority']
[out*in]: 10.074888  [human]:  7.070467   trust.v.01 --> responsible.a.01 ['responsible']
[out*in]:  9.824589  [human]: None        trust.v.01 --> social.a.02 ['social']
[out*in]:  9.795878  [human]: None        trust.v.01 --> job.n.03 ['job']
[out*in]:  9.735038  [human]: None        trust.v.01 --> benefit.n.02 ['benefit', 'welfare']
[out*in]:  9.711832  [human]: None        trust.v.01 --> demand.n.01 ['demand']
[out*in]:  9.666265  [human]: None        trust.v.01 --> politics.n.01 ['politics', 'political_relation']
[out*in]:  9.618358  [human]: None        trust.v.01 --> commitment.n.02 ['commitment', 'allegiance', 'loyalty', 'dedication']
[out*in]:  9.335283  [human]: None        trust.v.01 --> ethical.a.02 ['ethical']
[out*in]:  9.040393  [human]: None        trust.v.01 --> doubt.n.02 ['doubt', 'dubiousness', 'doubtfulness', 'question']
[out*in]:  8.944983  [human]: None        trust.v.01 --> human.a.01 ['human']
[out*in]:  8.705051  [human]: None        trust.v.01 --> freedom.n.01 ['freedom']
[out*in]:  8.702453  [human]: None        trust.v.01 --> monetary_value.n.01 ['monetary_value', 'price', 'cost']
[out*in]:  8.672050  [human]: None        trust.v.01 --> ask.v.04 ['ask', 'require', 'expect']
[out*in]:  8.591640  [human]: None        trust.v.01 --> concern.n.01 ['concern']
[out*in]:  8.526812  [human]: None        trust.v.01 --> official.a.01 ['official']
[out*in]:  8.501220  [human]: None        trust.v.01 --> illegal.a.01 ['illegal']
[out*in]:  8.411656  [human]: None        trust.v.01 --> valuable.a.01 ['valuable']
[out*in]:  8.361676  [human]: None        trust.v.01 --> promise.n.01 ['promise']
[out*in]:  8.323556  [human]:  2.931867   trust.v.01 --> firm.s.03 ['firm', 'strong']
[out*in]:  8.295831  [human]: None        trust.v.01 --> expertness.n.01 ['expertness', 'expertise']
[out*in]:  8.257878  [human]:  1.611767   trust.v.01 --> pledge.v.01 ['pledge', 'plight']
[out*in]:  8.240909  [human]: None        trust.v.01 --> bargain.n.01 ['bargain', 'deal']
[out*in]:  8.156078  [human]: None        trust.v.01 --> settlement.n.05 ['settlement', 'resolution', 'closure']
==================================================
 <--[ backward evocation ]--
==================================================
[out*in]: 11.210469  [human]:  3.522900   trust.v.01 <-- transaction.n.01 ['transaction', 'dealing', 'dealings']
[out*in]: 10.275481  [human]: None        trust.v.01 <-- depository_financial_institution.n.01 ['depository_financial_institution', 'bank', 'banking_concern', 'banking_company']
[out*in]: 10.207652  [human]: None        trust.v.01 <-- investment.n.02 ['investment', 'investment_funds']
[out*in]: 10.120443  [human]: None        trust.v.01 <-- religious.a.02 ['religious']
[out*in]: 10.010049  [human]: None        trust.v.01 <-- authority.n.02 ['authority']
[out*in]:  9.954803  [human]: None        trust.v.01 <-- political.a.01 ['political']
[out*in]:  9.926022  [human]: None        trust.v.01 <-- commitment.n.02 ['commitment', 'allegiance', 'loyalty', 'dedication']
[out*in]:  9.805646  [human]: None        trust.v.01 <-- duty.n.01 ['duty', 'responsibility', 'obligation']
[out*in]:  9.080510  [human]:  4.120000   trust.v.01 <-- institution.n.01 ['institution', 'establishment']
[out*in]:  8.612347  [human]: None        trust.v.01 <-- court.n.01 ['court', 'tribunal', 'judicature']
[out*in]:  8.562712  [human]: None        trust.v.01 <-- personal.a.01 ['personal']
[out*in]:  8.559251  [human]: None        trust.v.01 <-- value.n.03 ['value', 'economic_value']
[out*in]:  8.436293  [human]: None        trust.v.01 <-- benefit.n.02 ['benefit', 'welfare']
[out*in]:  8.432035  [human]: None        trust.v.01 <-- charity.n.05 ['charity']
[out*in]:  8.381653  [human]: None        trust.v.01 <-- promise.n.01 ['promise']
[out*in]:  8.366024  [human]: None        trust.v.01 <-- justice.n.01 ['justice', 'justness']
[out*in]:  8.337902  [human]: None        trust.v.01 <-- pension.n.01 ['pension']
[out*in]:  8.212354  [human]: None        trust.v.01 <-- school.n.06 ['school']
[out*in]:  8.175872  [human]: None        trust.v.01 <-- currency.n.01 ['currency']
[out*in]:  8.072791  [human]: -0.129367   trust.v.01 <-- slave.n.01 ['slave']
[out*in]:  8.068376  [human]: None        trust.v.01 <-- freedom.n.01 ['freedom']
[out*in]:  7.948745  [human]:  0.110000   trust.v.01 <-- job.n.03 ['job']
[out*in]:  7.895317  [human]: None        trust.v.01 <-- confession.n.01 ['confession']
[out*in]:  7.771608  [human]: None        trust.v.01 <-- social.a.02 ['social']
[out*in]:  7.758136  [human]: None        trust.v.01 <-- suffering.n.04 ['suffering', 'hurt']
[out*in]:  7.706671  [human]: None        trust.v.01 <-- member.n.01 ['member', 'fellow_member']
[out*in]:  7.673578  [human]: -0.086767   trust.v.01 <-- campaign.n.03 ['campaign', 'military_campaign']
[out*in]:  7.659255  [human]: None        trust.v.01 <-- official.a.01 ['official']
[out*in]:  7.605260  [human]: None        trust.v.01 <-- wrong.a.05 ['wrong']
[out*in]:  7.570884  [human]: None        trust.v.01 <-- demand.n.01 ['demand']
```

## All Experiments 
https://docs.google.com/spreadsheets/d/13si_sL6YTc7zA8ERsFbrbPulNhrEcYXMYaFDCx_798o/edit?usp=sharing
