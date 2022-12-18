from dataloader import MmCorpusLoader, GensimCorpusLoader
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from scipy.sparse import csr_matrix, save_npz
import numpy as np
import argparse
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--id', default=None)

args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
id = args.id

if id is not None:
    path = config["data"].format(id)
    output = Path(config["output"].format(id))
else:
    path = config["data"]
    output = Path(config["output"])
dict_file = config["dictionary"]
model_path = config["model"]

MATRIX = "MmCorpusLoader"
GENSIM = "GensimCorpusLoader"

loader = config["dataloader"]

model = LdaMulticore.load(model_path)
dictionary = Dictionary.load(dict_file)
if loader == MATRIX:
    corpus = MmCorpusLoader(path)
elif loader == GENSIM:
    corpus = GensimCorpusLoader(path, dictionary)
else:
    raise ValueError("Invalid data loader %s" % (loader))

data = []
indices = []
indptr = np.zeros(len(corpus)+1, dtype=np.int32)
indptr[0] = 0

for i, doc in enumerate(corpus):
    if doc is not None:
        topics, probs =  zip(*model.get_document_topics(doc))
        indptr[i+1] = indptr[i] + len(topics)
        data += list(probs)
        indices += list(topics)
    else:
        indptr[i+1] = indptr[i]

indptr[-1] = len(data)
# assert i == len(corpus)-1
print(id, i, len(corpus))
mat = csr_matrix((data, indices, indptr), shape=(len(corpus), model.num_topics)).asformat('csc')
output.parent.mkdir(parents=True, exist_ok=True)
save_npz(output, mat)
