import pandas

data=pandas.read_csv('news_data_6.csv')
kalimat_panjang=data['teks'].str.cat(sep='')
with open("news_data_8.txt", "w") as text_file:
    text_file.write(kalimat_panjang)

import multiprocessing
import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

namaFileInput = "news_data_8.txt"
namaFileOutput = "hoax_en_case"

model = Word2Vec(LineSentence(namaFileInput), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())

# trim unneeded model memory = use (much) less RAM
model.init_sims(replace=True)
model.save(namaFileOutput)

