from typing import Iterable
from lm_dataformat import Reader, Archive
from pathlib import Path
from gensim.models import LdaModel
from gensim.corpora.mmcorpus import MmCorpus
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_short, strip_tags, strip_multiple_whitespaces,strip_numeric,stem_text
from scipy import sparse
import numpy as np
import random
import string
import argparse
import yaml
import logging
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

class CorpusLoader:
    def __init__(self, path, transforms=None, include=None, exclude=None):
        self.dataset_file = path
        self.reader = Reader(self.dataset_file)
        self.transforms = transforms
        self.include = include
        self.exclude = exclude
        info_path = Path(path)/'info.yaml'
        if Path(path).is_dir() and info_path.exists():
            logger.debug('Found info.yaml for %s'%path)
            info = yaml.load(open(str(info_path), 'r'), Loader=yaml.Loader)
            self.__len = info.get('length', None)
        else:
            self.__len = None

    def is_pile_set_included(self, pile_set):
        return ((self.include is None or pile_set in self.include) \
                and (self.exclude is None or pile_set not in self.exclude))
    
    def __iter__(self):
        for text, meta in self.reader.stream_data(get_meta=True):
            if self.is_pile_set_included(meta["pile_set_name"]):
                doc = text
                if self.transforms != None:
                    try:
                        for transform in self.transforms:
                            doc = transform(doc)
                    except RecursionError as e:
                        logger.warn('Failed to apply transforms with input:\n"%s"'%(text)) # TODO: do proper logging
                        yield None
                        continue
                yield doc
                
    def __len__(self):
        if self.__len != None:
            return self.__len
        self.__len = 0
        for _, meta in self.reader.stream_data(get_meta=True):
            if self.include == None or (meta['pile_set_name'] in self.include):
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
    def __init__(self, path, dictionary, include=None, exclude=None):
        transforms = [GensimCorpusLoader.preprocessing(), dictionary.doc2bow]
        super().__init__(path, include=include, exclude=exclude, transforms=transforms)

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

class TopicLoaderLda(CorpusLoader):
    def __init__(self, path, dictionary: Dictionary, ldamodel: LdaModel, topic_ids, threshold=0.75, keep=True, include=None, exclude = None):
        super().__init__(path, include=include, exclude=exclude)
        self.dictionary = dictionary
        self.ldamodel = ldamodel
        self.topic_ids = topic_ids
        self.keep = keep
        self.threshold = threshold

    def __iter__(self):
        preprocessing = GensimCorpusLoader.preprocessing()
        for doc in super().__iter__():
            bow = self.dictionary.doc2bow(preprocessing(doc))
            topics = self.ldamodel.get_document_topics(bow)
            has_topics = False
            for topic, prob in topics:
                if topic in self.topic_ids and prob >= self.threshold:
                    has_topics = True
                    break
            if has_topics == self.keep:
                yield doc

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
    def __init__(self, path, topic_prob_path, topic_ids, threshold=0.75, keep=True, get_meta=False):
        self.datapath = Path(path)
        self.probs = Path(topic_prob_path)
        self.topic_ids = topic_ids
        self.threshold = threshold
        self.keep = keep
        self.get_meta = get_meta

    def iter_files(self):
        if self.datapath.is_dir():
            files = list(self.datapath.iterdir())
            npz_files = list((self.probs/('%s.npz' % path.stem) for path in files))
        else:
            files = [self.datapath]
            npz_files = [self.probs]
        for data, topic_prob_file in zip(files, npz_files):
            try:
                probs = sparse.load_npz(topic_prob_file).toarray()
            except:
                logger.error('Failed to load %s for %s' % (topic_prob_file, data))
                continue
            yield data, probs

    def __iter__(self):
        topic_ids = np.array(self.topic_ids)
        for data, probs in self.iter_files():
            doc_topics = probs[:,topic_ids] >= self.threshold
            for doc, topics in zip(Reader(str(data)).stream_data(get_meta=self.get_meta), doc_topics):                
                if any(topics) == self.keep:
                    if self.get_meta:
                        doc[1]["topics"] = topic_ids[topics].tolist()
                    yield doc
    
    def __len__(self):
        for _, probs in self.iter_files():
            if self.keep:
                return np.sum(np.any(probs[:,self.topic_ids] >= self.threshold, axis=1))
            else:
                return np.sum(np.any(probs[:,self.topic_ids] < self.threshold, axis=1))
    
    def save(self, output):
        archive = Archive(output)
        for doc in self:
            if not self.get_meta:
                archive.add_data(doc)
            else:
                archive.add_data(doc[0], meta=doc[1])
        archive.commit()
        self.write_info(output)

    def write_info(self, output):
        info = {
            "length": len(self),
            "datapath": str(self.datapath),
            "probs": str(self.probs),
            "topic_ids": self.topic_ids,
            "threshold": self.threshold,
            "keep": self.keep
        }
        yaml.dump(info, open(Path(output)/"info.yaml",'w'), Dumper=yaml.Dumper)

class ConcatDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.__len = None

    def __iter__(self):
        for dataset in self.datasets:
            for data in dataset:
                yield data

    def __len__(self):
        if self.__len is None:
            self.__len = sum(len(dataset) for dataset in self.datasets)
        return self.__len        
    
    @staticmethod
    def corpus_from_dir(path, include=None, exclude=None):
        return ConcatDataset([CorpusLoader(p, include=include, exclude=exclude) for p in Path(path).iterdir()])

class FinetuneDataset(IterableDataset):
    def __init__(self, data, topic_prob_path, topic_data, topic_ids, threshold=0.75):
        # TODO how to match threshold with topic_prob_path's
        self.topic_data = topic_data
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
    if args.id is not None:
        print(f'Id: {args.id}')
    
    dataset = TopicDataset(data, topics, topic_ids, threshold=threshold, get_meta=True)
    Path(output).mkdir(parents=True, exist_ok=True)
    dataset.save(output)
    print(f'number of documents %s'%len(dataset), output)
    #dataset.write_info(output)
