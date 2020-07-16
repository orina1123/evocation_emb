import argparse
from sklearn import preprocessing
import numpy as np

# settings
ap = argparse.ArgumentParser()

ap.add_argument("pair_score_file", type=argparse.FileType('r'), help="TSV of synset pair - score [controlled.standard.synset_pair_avg_score.tsv]")
ap.add_argument("scale_min", type=float, help="min of scaling range")
ap.add_argument("scale_max", type=float, help="max of scaling range")
ap.add_argument("scaled_file", type=argparse.FileType('w'), help="TSV with scaled scores")

args = ap.parse_args()

pair_list = []
Y = []
for line in args.pair_score_file:
    s1, s2, score_str = line.strip().split('\t')
    pair_list.append((s1, s2))
    y = float(score_str)
    Y.append([y])

scaler = preprocessing.MinMaxScaler(feature_range=(args.scale_min, args.scale_max))
scaled_Y = scaler.fit_transform(np.array(Y))

for i, (s1, s2) in enumerate(pair_list):
    args.scaled_file.write("%s\t%s\t%f\n" % (s1, s2, scaled_Y[i][0]))
