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
from Classifier import find_similar_scripts

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
    
fc = MarkovModel("fc", FullCorpus.listOfLines, smooth_param=0.0001)
#jane = MarkovModel("jane", external_corpus, smooth_param=1)
#print(fc.generate(20))

scripts = ['Goodfellas.1990.txt', 'Love.And.Other.Drugs.txt', 'Notting.Hill.1999.txt', 'The.Notebook.2004.txt', 'Eternal.Sunshine.Of.The.Spotless.Mind.txt', 'Blue.Valentine.2010.txt', 'Sweet.November.2001.txt', '500.Days.of.Summer.2009.txt', 'One.Flew.Over.the.Cuckoos.Nest.txt', 'Sliding.Doors.txt']

external_raw = []
for script in scripts:
    file = open('cleanedSRT/'+script, 'r')
    external_raw.append(file.readlines())
    file.close()
    
external = []

punctuation = "\"#$%&'()*+,-/:;<=>@[\]^_`{|}~"
end_tokens = [".", "?", "!"]

for s in external_raw[0]:
    sentences = []
    
    # I think dividing into sentences was a mistake
    raw_sentences = sent_tokenize(s)
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
    external.append(sentences)

all_external = []

for e in external:
    for l in e:
        all_external.append(l)
        
ex = MarkovModel("ex", all_external, 0.0001)

combo = ComboMarkovModel(fc, ex, 0.9)

print(combo.generate(100))