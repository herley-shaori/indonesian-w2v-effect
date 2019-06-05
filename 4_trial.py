import pandas
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

data=pandas.read_csv('vector_data.csv')
a_data=data['kelas']
b_data=data.drop('kelas',1)
X_train, X_test, y_train, y_test = train_test_split(b_data, a_data, test_size=0.33, random_state=100
,shuffle=True)

# Random Forest.
clf=RandomForestClassifier(random_state=10,n_estimators=101)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('Random Forest')
print(classification_report(y_test, y_pred))
print('-----------------------------------------')

# SVM.
clf=svm.SVC(gamma='auto',kernel='rbf')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('SVM-rbf')
print(classification_report(y_test, y_pred))
print('-----------------------------------------')

# SVM.
clf=svm.SVC(gamma='auto',kernel='linear')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('SVM-linear')
print(classification_report(y_test, y_pred))
print('-----------------------------------------')

# SGD.
clf=SGDClassifier(loss="hinge", penalty="l2", max_iter=500,random_state=10)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('SGD')
print(classification_report(y_test, y_pred))
print('-----------------------------------------')

# Gaussian-NB.
clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('Gaussian-NB')
print(classification_report(y_test, y_pred))
print('-----------------------------------------')

