# -*- coding: utf-8 -*-
"""
@author: Tristan Sparks
@author: Mahyar Bayran
"""

from nltk.tokenize import sent_tokenize

import matplotlib.pyplot as plt

import numpy as np 
import os.path
import string
from CharacterModule import Character
from MarkovModelModule import MarkovModel, WeightedComboMarkovModel, NormalizedComboMarkovModel
#from LSTM import LSTM_byChar

from nltk.corpus import gutenberg
from nltk.util import ngrams


######### PRE-PROCESSING ###################
def ModelPerplexity(distribution):
    exp = 0
    for e in distribution:
        if (e != 0):
            exp += e*np.log2(e)
    
    return 2**(-1*exp)

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
    
############################# MAIN ############################################

print("reading files")
file_corpus = open('corpus.txt')
corpus = file_corpus.readlines()
file_corpus.close()

# a list of character names
charNames = []
Characters = {}

FullCorpus = Character("full")

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

best_scripts = [('Goodfellas.1990.txt', 0.29524089306698004), ('Prime_eng.txt', 0.26586368977673325), ('Love.And.Other.Drugs.txt', 0.2517626321974148), ('Notting.Hill.1999.txt', 0.24294947121034077), ('Selena.1997.txt', 0.23678025851938894), ('The.Notebook.2004.txt', 0.23325499412455933), ('Eternal.Sunshine.Of.The.Spotless.Mind.txt', 0.23237367802585193), ('Blue.Valentine.2010.txt', 0.23061104582843714), ('Sweet.November.2001.txt', 0.22943595769682726), ('Wicker.Park.2004.txt', 0.22032902467685075)]
worst_scripts = [('August.Rush.2007.txt', 0.11339600470035252), ('Tristan.and.Isolde.2006.txt', 0.13072855464159813), ('Jane.Eyre.2011.txt', 0.13719153936545242), ('Firelight.1997.txt', 0.15481786133960046), ('Original.Sin.2001.txt', 0.15804935370152762), ('Drive.1997.txt', 0.15834312573443007), ('Tuck.Everlasting.txt', 0.1601057579318449), ('dakota-skye.txt', 0.1627497062279671), ('Seven.Samurai.1954.txt', 0.1636310223266745), ('The.Lake.House.2006.txt', 0.1653936545240893)]

external_raw = []
for script, score in best_scripts:
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
   
print("generating models")     
ex = MarkovModel("ex", all_external, n=2, smooth_param=0)
        
#fc = MarkovModel("fc", FullCorpus.listOfLines, n=2, smooth_param=0)
mm = MarkovModel('Johnny', Characters['Johnny'].listOfLines, n=2, smooth_param=0)

combo1 = WeightedComboMarkovModel(mm, ex, weight=0.9)
combo2 = NormalizedComboMarkovModel(mm, ex, weight=0.9)


print("calculating markov perplexity")
print("Markov Perplexity: ", ModelPerplexity(mm.initial_dist))

print("calculating weight perplexity")
print("Weight Perplexity: ", ModelPerplexity(combo1.initial_dist))

print("calculating normal perplexity")
print("Normal Perplexity: ", ModelPerplexity(combo2.initial_dist))

print(mm.generate())
#print(ex.generate())
print(combo1.generate())
print(combo2.generate())

#Johnny_text = ""
#for line in corpus:
#    a = line.index(':')
#    name = line[0:a]
#    l = line[a+2:]
#    if name == 'Johnny':
#        Johnny_text +=  l
#    
#A = LSTM_byChar(Johnny_text)






    
