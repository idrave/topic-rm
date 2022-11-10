from typing import Iterable
from lm_dataformat import Reader, Archive
from pathlib import Path
from gensim.corpora.mmcorpus import MmCorpus
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_short, strip_tags, strip_multiple_whitespaces,strip_numeric,stem_text
from torch.utils.data import IterableDataset
from scipy import sparse
import numpy as np
import random
import string
import argparse
import yaml

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
                doc = text
                if self.transforms != None:
                    try:
                        for transform in self.transforms:
                            doc = transform(doc)
                    except RecursionError as e:
                        print('Error: Failed to apply transforms with input:\n"%s"'%(text)) # TODO: do proper logging
                        continue
                yield doc
                
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
        transforms = [GensimCorpusLoader.preprocessing(), dictionary.doc2bow]
        super().__init__(path, set_names=set_names, transforms=transforms)

    @staticmethod
    def preprocessing():
        replace = str.maketrans('', '', string.punctuation+'´')
        parse = [strip_multiple_whitespaces,
            strip_tags,
            strip_numeric,
            lambda x: x.lower().translate(replace),
            remove_stopwords,
            stem_text,
            strip_short
        ]
        return lambda x: preprocess_string(x, parse)

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

class TopicDataset:
    def __init__(self, path, topic_prob_path, topic_ids, threshold=0.75, keep=True):
        self.datapath = Path(path)
        self.probs = Path(topic_prob_path)
        self.topic_ids = topic_ids
        self.threshold = threshold
        self.keep = keep

    def __iter__(self):
        if self.datapath.is_dir():
            files = self.datapath.iterdir()
            npz_files = (self.probs/('%s.npz' % path.stem) for path in files)
        else:
            files = [self.datapath]
            npz_files = [self.probs]

        for data, topic_prob_file in zip(files, npz_files):
            try:
                probs = sparse.load_npz(topic_prob_file).toarray()
            except:
                print('Error: failed to load %s for %s' % (topic_prob_file, data))
                continue
            if self.keep:
                to_yield = np.any(probs[:,self.topic_ids] >= self.threshold, axis=1)
            else:
                to_yield = np.any(probs[:,self.topic_ids] < self.threshold, axis=1)
            for doc, b in zip(Reader(str(data)).stream_data(), to_yield):                
                if b:
                    yield doc
    
    def save(self, output):
        archive = Archive(output)
        for doc in self:
            archive.add_data(doc)
        archive.commit()

class FinetuneDataset(IterableDataset):
    def __init__(self, data, topic_prob_path, topic_data, topic_ids, threshold=0.75):
        # TODO how to match threshold with topic_prob_path's
        self.topic_data = CorpusLoader(topic_data)
        self.non_topic_data = TopicDataset(data, topic_prob_path, topic_ids,
                                            threshold=threshold, keep=False)
        # if size of non_topic_data larger than size of topic_data,
        # iterate multiple times over the former 
        self.non_topic_iter = iter(self.non_topic_data)

    def __iter__(self):
        for doc1 in self.topic_data:
            doc2 = next(self.non_topic_iter, None)
            if doc2 is None:
                self.non_topic_iter = iter(self.non_topic_data)
                doc2 = next(self.non_topic_iter)
            yield doc1, doc2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--id', default=None)
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if args.id is not None:
        data = config["data"].format(id=args.id)
        topics = config["topics"].format(id=args.id)
        output = config["output"].format(id=args.id)
    else:
        data = config["data"]
        topics = config["topics"]
        output = config["output"]
    topic_ids = config["topic_ids"]
    threshold = config["threshold"]
    
    dataset = TopicDataset(data, topics, topic_ids, threshold=threshold)
    Path(output).mkdir(parents=True, exist_ok=True)
    dataset.save(output)
