import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import urllib.parse
from sklearn import svm
from sklearn.metrics import confusion_matrix

import pandas as pd
import pickle
from data_description import test_data

vect, X_test, y_test = test_data()

svc = pickle.load(open('svm_model', 'rb'))
pred = svc.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, (svc.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)
cm = confusion_matrix(y_test, pred)
print("------------")
print("Accuracy: %f" % svc.score(X_test, y_test))  #checking the accuracy
print("Precision: %f" % metrics.precision_score(y_test, pred))
print("Recall: %f" % metrics.recall_score(y_test, pred))
print("F1-Score: %f" % metrics.f1_score(y_test, pred))
print("AUC: %f" % auc)
print("Printing Confusion matrix:")
print(cm)