#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:35:04 2019

@author: rushikesh
"""

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import urllib.parse
from sklearn import svm

import pandas as pd

data = pd.read_csv('payload_full')
res = data['label']
queries = data['payload']

y = []
validCount=0
badCount=0
for r in res:
    if r=='norm':
        y.append(0)
        validCount+=1
    else:
        y.append(1)
        badCount+=1
        
vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(3,4)) #converting data to vectors
X = vectorizer.fit_transform(queries)
from sklearn.preprocessing import scale

scaled_X = scale(X, with_mean=False)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)


len(vectorizer.get_feature_names())
svc = svm.SVC(probability=True,cache_size=200)


svc.fit(X_train, y_train)

pred = svc.predict(X_test)

# import pickle
# pickle.dump(svc, open('svm_model', 'wb'))



svcn = pickle.load(open('svm_model', 'rb'))

fpr, tpr, _ = metrics.roc_curve(y_test, (svc.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)

print("Bad samples: %d" % badCount)
print("Good samples: %d" % validCount)
print("Baseline Constant negative: %.6f" % (validCount / (validCount + badCount)))
print("------------")
print("Accuracy: %f" % svcn.score(X_test, y_test))  #checking the accuracy
print("Precision: %f" % metrics.precision_score(y_test, pred))
print("Recall: %f" % metrics.recall_score(y_test, pred))
print("F1-Score: %f" % metrics.f1_score(y_test, pred))
print("AUC: %f" % auc)