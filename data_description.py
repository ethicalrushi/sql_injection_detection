import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import urllib.parse
from sklearn import svm

import pandas as pd

def test_data():
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
    return vectorizer, X_test, y_test

def vect():
    def loadFile(name):
        directory = str(os.getcwd())
        filepath = os.path.join(directory, name)
        with open(filepath,'r') as f:
            data = f.readlines()
        data = list(set(data))
        result = []
        for d in data:
            d = str(urllib.parse.unquote(d))   #converting url encoded data to simple string
            result.append(d)
        return result

    badQueries = loadFile('badqueries.txt')
    validQueries = loadFile('goodqueries.txt')

    badQueries = list(set(badQueries))
    validQueries = list(set(validQueries))

    allQueries = badQueries + validQueries
    yBad = [1 for i in range(0, len(badQueries))]  #labels, 1 for malicious and 0 for clean
    yGood = [0 for i in range(0, len(validQueries))]
    y = yBad + yGood
    queries = allQueries

    vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(3,4)) #converting data to vectors
    X = vectorizer.fit_transform(queries)
    return vectorizer