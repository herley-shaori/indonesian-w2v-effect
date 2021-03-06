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

	# remove duplicate data
	print(data.shape)
	data=data.drop_duplicates(subset=['teks'], keep=False)
	print(data)
	print(data.shape)

	index2word_set = set(model.wv.index2word)	
	vectorData=[]
	for index,row in data.iterrows():
		kalimat=row['teks']
		featureVec = np.zeros(400,dtype="float32")
		nwords = 0

		for word in kalimat.split():
        		if word in index2word_set:
            			nwords = nwords + 1
            			featureVec = np.add(featureVec,model[word])
		# Dividing the result by number of words to get average
		featureVec = np.divide(featureVec, nwords)
		localIndex=0
		localDict={}
		for x in np.nditer(featureVec):
			localDict[localIndex]=x
			localIndex+=1
		localDict['kelas']=row['kelas']
		vectorData.append(localDict)

	gammaDataframe=pandas.DataFrame(vectorData)
	gammaDataframe=gammaDataframe.astype('float64')
	gammaDataframe.kelas = gammaDataframe.kelas.astype('int64')
	return gammaDataframe

data=pandas.read_csv('news_data_6.csv')
#data['text']=data['text'].apply(toLowecase)

# Gensim Preparation.
namaFileModel = "hoax_en_case"
model = gensim.models.Word2Vec.load(namaFileModel)

vectorDataframe=vectorRepresent(model,data)
#data.to_csv('news_data_2.csv',index=False)
vectorDataframe.to_csv('en_vector_data.csv',index=False)

 
