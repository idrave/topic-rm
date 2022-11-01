from lm_dataformat import Reader
from pathlib import Path
from gensim.corpora.mmcorpus import MmCorpus
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_short, strip_tags, strip_multiple_whitespaces,strip_numeric,stem_text
import random
import string

class CorpusLoader:
    def __init__(self, path, transforms=None, set_names=None):
        self.dataset_file = path
        self.reader = Reader(self.dataset_file)
        self.transforms = transforms
        self.set_names = set_names
        self.__len = None
    
    def __iter__(self):
        for text, meta in self.reader.stream_data(get_meta=True):
            if self.set_names == None or (meta['pile_set_name'] in self.set_names):
                if self.transforms != None:
                    for transform in self.transforms:
                        text = transform(text)
                yield text
                
    def __len__(self):
        if self.__len != None:
            return self.__len
        self.__len = 0
        for _, meta in self.reader.stream_data(get_meta=True):
            if self.set_names == None or (meta['pile_set_name'] in self.set_names):
                self.__len += 1
        return self.__len

    def bool_index(self, idx):
        assert idx.shape == (len(self),)
        for doc, b in zip(self, idx):
            if b:
                yield doc

    def random_loader(self, p):
        return CorpusRandLoader(self, p=p)

    def iter_offset(self, offset):
        it = iter(self)
        for _ in range(offset):
            next(it)
        return it


class GensimCorpusLoader(CorpusLoader):
    def __init__(self, path, dictionary, set_names=None):
        translation = str.maketrans('', '', string.punctuation+'´')
        parse = [strip_multiple_whitespaces,
            strip_tags,
            strip_numeric,
            lambda x: x.lower().translate(translation),
            remove_stopwords,
            stem_text,
            strip_short
        ]
        transforms = [lambda x: preprocess_string(x, parse), dictionary.doc2bow]
        super().__init__(path, set_names=set_names, transforms=transforms)

class MmCorpusLoader:
    def __init__(self, path):
        self.path = Path(path)
        self.__len = None
    
    def __iter__(self):
        if self.path.is_dir():
            for file in self.path.iterdir():
                if file.suffix == '.mm':
                    corpus = MmCorpus(str(file))
                    for text in corpus:
                        yield text
        else:
            assert self.path.suffix == '.mm', 'Wrong file type %s instead of .mm' % (self.path.suffix)
            corpus = MmCorpus(str(self.path))
            for text in corpus:
                yield text
                
    def __len__(self):
        if self.__len != None:
            return self.__len
        if self.path.is_dir():
            self.__len = 0
            for file in self.path.iterdir():
                if file.suffix == '.mm':
                    corpus = MmCorpus(str(file))
                    self.__len += len(corpus)
        else:
            assert self.path.suffix == '.mm', 'Wrong file type %s instead of .mm' % (self.path.suffix)
            corpus = MmCorpus(str(self.path))
            self.__len = len(corpus)
        return self.__len

class CorpusRandLoader:
    def __init__(self, corpus, p=0.1):
        self.corpus = corpus
        self.p = p
    
    def __iter__(self):
        for s in self.corpus:
            sample = None
            while sample is None:
                if random.random() < self.p:
                    sample = s
            yield sample

class TopicDataset(GensimCorpusLoader):
    def __init__(self, path, dictionary, topic_model, topic_id):
        super().__init__(path, dictionary)
        self.dataset_file = path
        self.reader = Reader(self.dataset_file)
        self.topic_model = topic_model
        self.topic_id = topic_id
    
    def __iter__(self):
        for text in super():
            pass