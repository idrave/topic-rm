from gensim.corpora.dictionary import Dictionary
from gensim import corpora
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('inputdir')
parser.add_argument('n', type=int)
parser.add_argument('output')

args = parser.parse_args()

dic_agg = None
cfs = None

for file in Path(args.inputdir).iterdir():
    if not file.is_dir():
        d = Dictionary.load(str(file))
        if dic_agg == None:
            dic_agg = d
            cfs = dict(d.cfs)
        else:
            d_to_agg = dic_agg.merge_with(d)
            for i, count in d_to_agg[d.cfs.items()]:
                if i not in cfs:
                    cfs[i] = 0
                cfs[i] += count
    
from heapq import nlargest

n = args.n
topn = nlargest(n, cfs, key=lambda x: cfs[x])
print('top 20', ', '.join('{},{}'.format(dic_agg[w], cfs[w]) for w in topn[:20]))

dic_agg.filter_tokens(good_ids=topn)
dic_agg.compactify()
dic_agg.save(args.output)