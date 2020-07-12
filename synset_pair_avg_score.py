import argparse

# settings
ap = argparse.ArgumentParser()

ap.add_argument("pair_file", type=argparse.FileType('r'), help="list of synset pairs [controlled.word-pos-sense]")
ap.add_argument("score_file", type=argparse.FileType('r'), help="list of human-rated evocation scores, multiple scores per line [controlled.standard]")

args = ap.parse_args()

pair_list = []
for line in args.pair_file:
    s1, s2 = line.strip().split(',')
    pair_list.append( (s1, s2) )

cnt = 0
for line in args.score_file:
    scores = list(map( lambda x: float(x), line.strip().split(' ') ))
    avg_score = sum(scores)/len(scores)
    
    s1, s2 = pair_list[cnt]
    print("%s\t%s\t%f" % (s1, s2, avg_score))

    cnt += 1
