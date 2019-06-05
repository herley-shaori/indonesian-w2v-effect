from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd

class DenseTfidfVectorizer(TfidfVectorizer):

    def transform(self, raw_documents, copy=True):
        X = super().transform(raw_documents, copy=copy)
        df = pd.DataFrame(X.toarray(), columns=self.get_feature_names())
        return df

    def fit_transform(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents, y=y)
        df = pd.DataFrame(X.toarray(), columns=self.get_feature_names())
        return df

data=pd.read_csv('news_data_2.csv')
abc=DenseTfidfVectorizer()
tfidf=abc.fit_transform(data['text'],data['class'])
tfidf['kelas']=data['class']

a_data=tfidf['kelas']
b_data=tfidf.drop('kelas',1)
X_train, X_test, y_train, y_test = train_test_split(b_data, a_data, test_size=0.33, random_state=100
,shuffle=True)

# SVM.
clf=svm.SVC(gamma='scale')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('SVM')
print(classification_report(y_test, y_pred))
print('-----------------------------------------')


# Gaussian-NB.
clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('Gaussian-NB')
print(classification_report(y_test, y_pred))
print('-----------------------------------------')

# Random Forest.
clf=RandomForestClassifier(random_state=10,n_estimators=101)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('Random Forest')
print(classification_report(y_test, y_pred))
print('-----------------------------------------')

# PCA.
pca = PCA(n_components=2)# adjust yourself
pca.fit(X_train)
X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)

clf=svm.SVC(gamma='scale')
clf.fit(X_t_train,y_train)
y_pred=clf.predict(X_t_test)
print('SVM-PCA')
print(classification_report(y_test, y_pred))
print('-----------------------------------------')

