# -*- coding: utf-8 -*-
"""
@author: Tristan Sparks
@author: Mahyar Bayran
"""

import random

from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np 

