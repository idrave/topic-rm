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
from gensim.corpora.mmcorpus import MmCorpus

class CorpusLoader:
    def __init__(self, path):
        self.path = Path(path)
        self.__len = None
    
    def __iter__(self):
        for file in self.path.iterdir():
            if file.suffix == '.mm':
                print('hey')
                corpus = MmCorpus(str(file))
                for text in corpus:
                    yield text
                
    def __len__(self):
        if self.__len != None:
            return self.__len
        self.__len = 0
        for file in self.path.iterdir():
            if file.suffix == '.mm':
                print(file)
                corpus = MmCorpus(str(file))
                print(len(corpus))
                self.__len += len(corpus)
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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dictionary')
parser.add_argument('data')
parser.add_argument('output')
parser.add_argument('-w', type=int, default=4)
parser.add_argument('-t', type=int, default=10)
args = parser.parse_args()

dictpath = args.dictionary
datapath = args.data
outdir = args.output
workers = args.w
n_topics = args.t

dictionary = Dictionary.load(dictpath)
corpus = CorpusLoader(datapath)
#corpus = UntilIter(corpus, 100)

start = time.time()
print('starting')
print(len(corpus))

lda = LdaMulticore(corpus, id2word=dictionary, num_topics=n_topics, workers=workers)
#lda = LdaModel(corpus, id2word=dictionary, num_topics=n_topics, distributed=True)
print('lda', time.time() - start)
print(lda.print_topics(num_topics=n_topics, num_words=10))
Path(outdir).parent.mkdir(parents=True, exist_ok=True)
lda.save(outdir)
open(Path(outdir).parent/'cmd.txt', 'w').write(str(args))