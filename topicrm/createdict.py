import glob
import os
import math
import tqdm
import string
import time
from lm_dataformat import Reader
from pathlib import Path
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_short, strip_tags, strip_multiple_whitespaces,strip_numeric,stem_text
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaModel
from gensim import corpora

class CorpusLoader:
    def __init__(self, path, transforms=None, set_names=None):
        self.dataset_file = path
        self.reader = Reader(self.dataset_file)
        self.stream = None
        self.transforms = transforms
        self.set_names = set_names
        self.__len = None
    
    def __iter__(self):
        for text, meta in self.reader.stream_data(get_meta=True):
            set_name = meta['pile_set_name']
            if self.set_names == None or set_name in self.set_names:
                if self.transforms != None:
                    for transform in self.transforms:
                        text = transform(text)
                yield text
                
    def __len__(self):
        if self.__len != None:
            return self.__len
        self.__len = 0
        for _, meta in self.reader.stream_data(get_meta=True):
            set_name = meta['pile_set_name']
            if self.set_names == None or set_name in self.set_names:
                self.__len += 1
        return self.__len
            
class UntilIter:
    def __init__(self, source, n):
        self.source = source
        self.n = n
        self.__len = None
    
    def __iter__(self):
        iterator = iter(self.source)
        for _ in range(self.n):
            try:
                yield next(iterator)
            except StopIteration:
                return
            
    def __len__(self):
        if self.__len == None:
            self.__len = len(self.source)
        return min(self.__len, self.n)
            
class GensimPre:
    def __init__(self, steps):
        self.steps = steps
    def __call__(self, word):
        return preprocess_string(word, self.steps)
    
    
translation = str.maketrans('', '', string.punctuation+'´')

l = [strip_multiple_whitespaces,
        strip_tags,
        strip_numeric,
        lambda x: x.lower().translate(translation),
        remove_stopwords,
        stem_text,
         strip_short
    ]

    
set_names = {
    'Wikipedia (en)'}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('id', type=int)
parser.add_argument('output')
args = parser.parse_args()

directory = args.dir
file_id = args.id
outdir = args.output

filename = "%s/%.2d.jsonl" % (directory, file_id)
corpus = CorpusLoader(filename, transforms=[GensimPre(l)], set_names=set_names)
#print('docs', len(corpus))
start = time.time()
common_dictionary = Dictionary(corpus)
print(set_names)
print('dictionary', time.time() - start)
print('words', len(common_dictionary))
print('most common', common_dictionary.most_common(20))
#common_dictionary.filter_extremes(keep_n=200000)
#print('filtered dictionary')
#print('words', len(common_dictionary))
#print('most common', common_dictionary.most_common(20))
Path(outdir).mkdir(parents=True, exist_ok=True)
common_dictionary.save('%s/%.2d.pkl'%(outdir, file_id))
#corpus.transforms = [GensimPre(l), common_dictionary.doc2bow]
"""
n_topics = 5
start = time.time()
lda = LdaMulticore(corpus, id2word=common_dictionary, num_topics=n_topics, workers=4)
print('lda', time.time() - start)
lda.print_topics(num_topics=n_topics, num_words=10)
lda.save('/cluster/scratch/irodrigu/project/lda_val.pkl')"""