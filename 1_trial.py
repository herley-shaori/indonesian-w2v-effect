import multiprocessing
import logging
import os.path
import sys
import multiprocessing
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

namaFileModel = "w2vec_wiki_id_case"
model = gensim.models.Word2Vec.load(namaFileModel)
hasil = model.wv.most_similar("Bandung")
print("Bandung:{}".format(hasil))
index2word_set = set(model.wv.index2word)
print('Panjang Model: ',len(model['Bandung']))

