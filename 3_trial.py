import pandas
import string
import multiprocessing
import logging
import os.path
import sys
import multiprocessing
import gensim
import numpy as np
from gensim.models import Word2Vec

def toLowecase(text):
	text=text.replace('-',' ')
	text=text.replace(';',' ')
	# Teks masukan masih tercampur dengan label kelas.
	text=text.replace('Valid','')
	text=text.replace('Hoax','')
	text = text.translate(text.maketrans('','',string.punctuation))
	return text.lower()

def vectorRepresent(model,data):
	# Pre-initialising empty numpy array for speed.
	# Menggunakan panjang vektor sama dengan 400.
	index2word_set = set(model.wv.index2word)
	
	vectorData=[]
	for index,row in data.iterrows():
		kalimat=row['text']
		featureVec = np.zeros(400,dtype="float32")
		nwords = 0

		for word in kalimat.split():
#			print(word)
        		if word in index2word_set:
            			nwords = nwords + 1
            			featureVec = np.add(featureVec,model[word])
		# Dividing the result by number of words to get average
		featureVec = np.divide(featureVec, nwords)
		vectorData.append({'vector':featureVec,'class':row['class']})
	return pandas.DataFrame(vectorData)

#for index,row in data.iterrows():
	#	print(row['text'],'\n')

data=pandas.read_csv('news_data.csv')
data['text']=data['text'].apply(toLowecase)

# Gensim Preparation.
namaFileModel = "w2vec_wiki_id_case"
model = gensim.models.Word2Vec.load(namaFileModel)

vectorDataframe=vectorRepresent(model,data)
data.to_csv('news_data_2.csv',index=False)
vectorDataframe.to_csv('vector_data.csv',index=False)

