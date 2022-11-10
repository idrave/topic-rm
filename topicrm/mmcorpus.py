import glob
import os
import math
import tqdm
import string
import time
from lm_dataformat import Reader, Archive
from pathlib import Path
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_short, strip_tags, strip_multiple_whitespaces,strip_numeric,stem_text
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaModel
from gensim import corpora
from gensim.corpora.mmcorpus import MmCorpus

class CorpusLoader:
    def __init__(self, path, transforms=None, set_name=None):
        self.dataset_file = path
        self.reader = Reader(self.dataset_file)
        self.stream = None
        self.transforms = transforms
        self.set_name = set_name
        self.__len = None
    
    def __iter__(self):
        for text, meta in self.reader.stream_data(get_meta=True):
            set_name = meta['pile_set_name']
            if self.set_name == None or set_name == self.set_name:
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
            if self.set_name == None or set_name == self.set_name:
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

    
set_name = 'Wikipedia (en)'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dictionary')
parser.add_argument('data')
parser.add_argument('output')
parser.add_argument('--archive', action='store_true')
args = parser.parse_args()

dictpath = args.dictionary
datapath = args.data
outdir = args.output
is_archive = args.archive

dictionary = Dictionary.load(dictpath)

start = time.time()

Path(outdir).parent.mkdir(parents=True, exist_ok=True)

if not is_archive:
    corpus = CorpusLoader(datapath, transforms=[GensimPre(l), dictionary.doc2bow], set_name=set_name)
    MmCorpus.serialize(outdir, corpus, dictionary)
    open(Path(outdir).parent/'cmd.txt', 'w').write(str(args))
else:
    corpus = CorpusLoader(datapath, set_name=set_name)
    archive = Archive(outdir)
    for text in corpus:
        archive.add_data(text, meta=None)
    archive.commit()
    open(Path(outdir)/'cmd.txt', 'w').write(str(args))
    
print('serialize', time.time() - start)
