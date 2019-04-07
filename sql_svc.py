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

#lgs = LogisticRegression(class_weight={1: 2 * validCount / badCount, 0: 1.0}) # class_weight='balanced')
#lgs.fit(X_train, y_train) #training our model

len(vectorizer.get_feature_names())
svc = svm.SVC(probability=True,cache_size=200)


svc.fit(X_train, y_train)

pred = svc.predict(X_test)

import pickle
pickle.dump(svc, open('svm_model', 'wb'))

#predicted = lgs.predict(X_test)
#predicted = svc.predict(X_test)

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




def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


import matplotlib.pyplot as plt
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, svc, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)