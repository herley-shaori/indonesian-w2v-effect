import multiprocessing
import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim

namaFileModel = "hoax_case"
model = gensim.models.Word2Vec.load(namaFileModel)

#panjang_vocab=len(model.wv.vocab)
#print(panjang_vocab)

hasil = model.wv.most_similar("hoax")
print(hasil)
