# -*- coding: utf-8 -*-
"""
@author: Tristan Sparks
@author: Mahyar Bayran
"""

from nltk.tokenize import sent_tokenize

import numpy as np 
import os.path
import string
from CharacterModule import Character
from MarkovModelModule import MarkovModel, WeightedMarkovModel, ComboMarkovModel

from nltk.corpus import gutenberg

######### PRE-PROCESSING ###################

def strip_line_with_sentences(line):
    a = line.index(':')
    name = line[0:a]
    l = line[a+2:] 
    # should remove punctuation at some point
    sentences = []

    punctuation = "\"#$%&'()*+,-/:;<=>@[\]^_`{|}~"
    end_tokens = [".", "?", "!"]
    
    # I think dividing into sentences was a mistake
    raw_sentences = sent_tokenize(l)
    for raw_sent in raw_sentences:
        sent = []
        
        # tokenize sentences by hand because NLTK doesn't like words like "gonna"
        # which will obviously be used FREQUENTLY for our purposes
        words = raw_sent.split(" ")
        table = str.maketrans('', '', punctuation)
        stripped = [w.translate(table) for w in words]
        
        for word in stripped:
            if (word not in string.punctuation):
                if (True in [word.strip().endswith(c) for c in end_tokens]):
                    end_char = word.strip()[-1]
                    sent.append(word.strip()[:-1].lower())
                    sent.append(end_char)
                else:
                    sent.append(word.strip().lower())
                
        if (sent != []):
            sentences.append(sent)
            
    return name, sentences

def strip_line_no_sentences(line):
    a = line.index(':')
    name = line[0:a]
    l = line[a+2:]

    punctuation = "\"#$%&'()*+,-/:;<=>@[\]^_`{|}~"
    end_tokens = [".", "?", "!"]
    # tokenize sentences by hand because NLTK doesn't like words like "gonna"
    # which will obviously be used FREQUENTLY for our purposes
    words = l.split(" ")
    table = str.maketrans('', '', punctuation)
    stripped = [w.translate(table) for w in words]
            
    sent = []
    for word in stripped:
        if (word not in string.punctuation):
            if (True in [word.strip().endswith(c) for c in end_tokens]):
                end_char = word.strip()[-1]
                sent.append(word.strip()[:-1].lower())
                sent.append(end_char)
            else:
                sent.append(word.strip().lower())
                
    return name, sent
    
file_corpus = open('corpus.txt')
corpus = file_corpus.readlines()

# a list of character names
charNames = []
Characters = {}

FullCorpus = Character("full")
external_corpus_raw = gutenberg.sents('bryant-stories.txt')[10:-1]

punctuation = "\"#$%&'()*+,-/:;<=>@[\]^_`{|}~"
end_tokens = [".", "?", "!"]

# read lines and create new characters
for line in corpus:
    #name, sentences = strip_line_with_sentences(line)
    name, sentences = strip_line_no_sentences(line)
    
    FullCorpus.addToLines(sentences)
    
    if name not in charNames:
        charNames.append(name)
        newChar = Character(name, sentences)
        Characters[name] = newChar

    else:
        Characters[name].addToLines(sentences)
        
#for name in charNames:
#    Characters[name].BuildAMarkovModel()
#    Characters[name].write_info()

external_corpus = []
for line in external_corpus_raw:
    new_line = []
    for word in line:
        if (word not in punctuation):
            new_line.append(word.lower())
     
    if (new_line != []):
        external_corpus.append(new_line)
    
fc = MarkovModel("fc", FullCorpus.listOfLines, smooth_param=0.5)
jane = MarkovModel("jane", external_corpus, smooth_param=1)
print(fc.generate(100))