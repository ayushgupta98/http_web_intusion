import pandas as pd
import numpy as np
from utils import *

df_data_1 = pd.read_csv("input.csv")
#replacing all the NULL entries with a '' (string with zero charaters)
df_data_1 = df_data_1.replace(np.nan, '', regex=True)
print(df_data_1.head())

"""
Feature Extraction & Encoding
"""
df_data_1['length_payload'] = df_data_1["payload"].str.len()
df_data_1['nd_payload'], df_data_1['nc_payload'], df_data_1['nsp_payload'] =  zip(*df_data_1['payload'].map(count_alpha_numeric_spchars))
df_data_1['trucated_url'] = df_data_1['url'].apply(length_path)
df_data_1['nd_url'], df_data_1['nc_url'], df_data_1['nsp_url'] =  zip(*df_data_1['trucated_url'].map(count_alpha_numeric_spchars))
df_data_1['length_url'] = df_data_1['trucated_url'].str.len()
df_data_1['nd_cookie'], df_data_1['nc_cookie'], df_data_1['nsp_cookie']  = zip(*df_data_1['cookie'].map(count_alpha_numeric_spchars))
df_data_1['contentType'] = df_data_1['contentType'].apply(content_type)
df_data_1['contentLength'] = df_data_1['contentLength'].apply(content_length)
df_data_1['nkeys'] = df_data_1['payload'].apply(number_of_keywords)
print(df_data_1.head())

#performing group-by operation
grouped_single = df_data_1.groupby('index').agg({'length_payload': ['sum'],
                                                 'nd_payload': ['sum'],
                                                 'nc_payload': ['sum'],
                                                 'nsp_payload': ['sum'],
                                                 'nd_url': ['sum'],
                                                 'nc_url': ['sum'],
                                                 'nsp_url': ['sum'],
                                                 'length_url': ['max'],
                                                 'method' : ['max'],
                                                 'nc_cookie': ['max'],
                                                 'nd_cookie': ['max'],
                                                 'contentType': ['max'],
                                                 'contentLength': ['max'],
                                                 'nkeys': ['sum']
                                                })

grouped_single['count_args'] = df_data_1.groupby("index")["payload"].size()
grouped_single['label_join'] = df_data_1.groupby("index")["label"].apply(lambda label: ','.join(label))
grouped_single['label'] = grouped_single['label_join'].apply(label_fix)
grouped_single['method_get'], grouped_single['method_post'], grouped_single['method_put'] = zip(*grouped_single['method']['max'].map(encode_method))
print(grouped_single.head())

#final dataframe having the relevant features
df = pd.DataFrame(columns = ['method_get', 'method_post', 'method_put'])
df['method_get'] = grouped_single['method_get']
df['method_post'] = grouped_single['method_post']
df['method_put'] = grouped_single['method_put']
df['length_args'] = grouped_single['length_payload']['sum']
df['number_args'] = grouped_single['count_args']
df['nd_payload'] = grouped_single['nd_payload']['sum']
df['nd_url'] = grouped_single['nd_url']['sum']
df['length_path'] = grouped_single['length_url']['max']
df['nc_payload'] = grouped_single['nc_payload']['sum']
df['nc_url'] = grouped_single['nc_url']['sum']
df['nsp_url'] = grouped_single['nsp_url']['sum']
df['nsp_payload'] = grouped_single['nsp_payload']['sum']
df['nd_cookie'] = grouped_single['nd_cookie']['max']
df['nc_cookie'] = grouped_single['nc_cookie']['max']
df['contentType'] = grouped_single['contentType']['max']
df['contentLength'] = grouped_single['contentLength']['max']
df['nkeys'] = grouped_single['nkeys']['sum']
df['label'] = grouped_single['label']
print(df.head())

"""
Machine Learning Models
"""
#Dividing the Dataset and normalize the fields
X = df.iloc[:, [0,1,2,4,5,6,8,10,11,12,13,14,15,16]].values
y = df.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X1 = sc.fit_transform(X)

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

"""
Logstic Regression
"""
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'sag', warm_start=True, random_state=0, max_iter = 1500)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("LR: ", (cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))

# cross-validation
lasso = LogisticRegression(solver = 'sag', warm_start=True, random_state=0, max_iter = 1500)
cv_results = cross_validate(lasso, X1, y, cv=20)
print("LR CV: ",cv_results['test_score'].mean())

"""
Support Vector Machine
"""
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("SVM: ", (cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))

# cross-validation
lasso = SVC(kernel = 'linear', random_state = 0)
cv_results = cross_validate(lasso, X1, y, cv=20)
print("SVM CV: ", cv_results['test_score'].mean())

"""
SGD
"""
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss="modified_huber", penalty="l2", max_iter=1000)#SGDClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("SGD: ", (cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))

# cross-validation
lasso = SGDClassifier()
cv_results = cross_validate(lasso, X1, y, cv=20)
print("SGD CV: ",cv_results['test_score'].mean())

"""
Multilayer Perceptron
"""
from sklearn.neural_network import MLPClassifier
lasso = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
cv_results = cross_validate(lasso, X1, y, cv=20)
print("MLP: ", cv_results['test_score'].mean())


"""
Linear Discriminant Analysis
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
classifier = LinearDiscriminantAnalysis(solver='lsqr')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("LDA: ", (cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))

#cross-validation
lasso = LinearDiscriminantAnalysis(solver='lsqr')
cv_results = cross_validate(lasso, X1, y, cv=20)
print("LDA CV: ", cv_results['test_score'].mean())

"""
CART or Bagging Classifier
"""
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X1, y, cv=kfold)
print("Bagging Classifier: ", results.mean())