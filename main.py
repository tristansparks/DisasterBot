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


'''
this seems to run really slow
used a simple dictionary instead

class Character:
    
    def __init__(self, name, firstline):
        self.name = name 
        self.listOfLines = []
        self.addToLines(firstline)
        
    def addToLines(self, newline):
        self.listOfLines.append(newline) #a list of lists
'''

file_corpus = open('corpus.txt')
corpus = file_corpus.readlines()

# a list of character names
charNames = []
Characters = {}

for line in corpus:
    a = line.index(':')
    name = line[0:a]
    sent = line[a+2:-1]
    sent = word_tokenize(sent.lower().strip())
    sent.append('\n') # indicate the end of the sentence
    
    if name not in charNames:
        charNames.append(name)
        #newChar = Character(name, sent)
        #Characters[name] = newChar
        Characters[name] = []
    #else:
        #Characters[name].addToLines(sent)

    Characters[name].append(sent)


for sent in Characters['Denny']:
    print(sent)
