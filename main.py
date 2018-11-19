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
import os.path


class MarkovModel:

    def __init__(self, name, listOfLines):
        self.name = name
        self.states = {} #a mapping from the word to number (indice of the word in states vector)
        count = 0
        # create mapping for states
        for line in listOfLines:
            for w in line:
                if w not in list(self.states.keys()): #new word found
                    self.states[w] = count
                    count += 1
                    
        # with mappings we can easily build initial distributions and transition matrix

        #### Lots of them have zero probabilities... so why keep them? (NOT DONE!!!!!!)
        #### For transition, keep a list of next possible states and then count and normalize
        #### For Initial_dist, keep a list of possible first words and then cound and normalize
        self.transition = np.zeros( (count,count) ) # Transition probabilities
        self.initial_dist = np.zeros( count ) # same length as states
        
        for line in listOfLines:
            # for now set the initial_dist by the first words in each LINE, not sentences
            self.initial_dist[self.states[line[0]]] += 1
            for i in list(range(0,len(line)-1)):
                self.transition[ self.states[line[i]], self.states[line[i+1]] ] += 1
        self.initial_dist /= self.initial_dist.sum()
        for i in list(range(0, len(self.transition)-1)):
            self.transition[i] /= self.transition[i].sum() 
    
    def write_info(self, file):
        file.write('States: \n'+ str(list(self.states.items())) +'\n\n\n')
        file.write('Initial Distributions: \n' + str(self.initial_dist.tolist()) + '\n\n\n')
        file.write('Transition Probabilities: \n' + str(self.transition.tolist() )+ '\n\n\n')
        
        
    #def generate(self, length):
        
    
class Character:

    def __init__(self, name, firstline):
        self.name = name 
        self.listOfLines = []
        self.MM =[]
        
    def addToLines(self, newline):
        self.listOfLines.append(newline) #a list of lists

    def BuildAMarkovModel(self):
        self.MM = MarkovModel(self.name, self.listOfLines)

    def write_info(self):
        filename = "%s.txt"%self.name
        completeName = os.path.join(os.getcwd() + '\\Characters', filename)
        file = open(completeName, "w")
        file.write("Character's Name:\t%s\n\n\n"%self.name)
        self.MM.write_info(file)
        
######### PRE-PROCESSING ###################

file_corpus = open('corpus.txt')
corpus = file_corpus.readlines()

# a list of character names
charNames = []
Characters = {}

# read lines and create new characters
for line in corpus:
    a = line.index(':')
    name = line[0:a]
    sent = line[a+2:] # -1 or nothing?
    sent = word_tokenize(sent.lower().strip())
    sent.append('\n') # indicate the end of the sentence
    
    if name not in charNames:
        charNames.append(name)
        newChar = Character(name, sent)
        Characters[name] = newChar

    Characters[name].addToLines(sent)

#for sent in Characters['Denny']:
#    print(sent)

############## Markov Model ###########

for name in charNames:
    Characters[name].BuildAMarkovModel()
    Characters[name].write_info()







