# -*- coding: utf-8 -*-
"""

@author: Ajit
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stop = stopwords.words('english')


def preprocessor(text):
    """ function takes a string and does the following:
        1. removes all html tags
        2. removes all numbers
        3. a variable emoticons to store emoticons that would be helpful in sentiment analysis
        4. changing text to lowercase for consistency
        5. stemming of words using PorterStemmer
        6.joining the emoticons at the end
    """
    text = re.sub('<[^>]*>', '', text)
    #text = re.sub('[0-9]','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    porter = PorterStemmer()
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')
    text = [porter.stem(word) for word in text.split() if word not in set(stop)]
    return text
    
def stream_data(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # this skips the header of the csv file
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label 

def getminibatch(doc_stream, size):
    """function uses the stream_data function to get a mini-batch of
        the data for training from the movie_data.csv
        
        doc_stream : stores the document stream
        size : specifies the size of mini-batch required
    """
    datas, y = [], []
    try:
        for _ in range(size):
            (text, label) = next(doc_stream)
            datas.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return datas, y

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error = 'ignore', n_features = 2 ** 21, tokenizer = preprocessor)
classifier = SGDClassifier(loss = 'log', random_state=1, n_jobs=-1)
doc_stream = stream_data(path = './movie_data.csv')


#traing the model in mini_batches
classes = np.array([0, 1])
for _ in range(45):
    X_train , y_train = getminibatch(doc_stream, size = 1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    classifier.partial_fit(X_train,y_train, classes= classes)
    
#testing the model
X_test, y_test = getminibatch(doc_stream, size = 5000)
X_test = vect.transform(X_test)
print('ACCURACY = %.3f' % classifier.score(X_test,y_test))

#after testing through test data use it to train the classifier
classifier = classifier.partial_fit(X_test, y_test)