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


from nltk.translate.bleu_score import sentence_bleu


######### PRE-PROCESSING ###################
def ModelPerplexity(distribution):
    exp = 0
    for e in distribution:
        if (e != 0):
            exp += np.log(1/e)
    
    #return 2**(-exp/len(distribution))
    return np.exp(exp/len(distribution))

def strip_line(line):
    punctuation = "\"#$%&'()*+,-/:;<=>@[\]^_`{|}~"
    end_tokens = [".", "?", "!"]
    # tokenize sentences by hand because NLTK doesn't like words like "gonna"
    # which will obviously be used FREQUENTLY for our purposes
    words = line.split(" ")
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
                
    return sent
    
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

#read lines and create new characters
for line in corpus:
    a = line.index(':')
    name = line[0:a]
    l = line[a+2:]
    tokens = strip_line(l)
    
    FullCorpus.addToLines(tokens)
    
    if name not in charNames:
        charNames.append(name)
        newChar = Character(name, tokens)
        Characters[name] = newChar

    else:
        Characters[name].addToLines(tokens)

#taken from output of Classifier.py
best_scripts = ['Goodfellas.1990.txt', 'Prime_eng.txt', 'Love.And.Other.Drugs.txt', 'Notting.Hill.1999.txt', 'Selena.1997.txt', 'The.Notebook.2004.txt', 'Eternal.Sunshine.Of.The.Spotless.Mind.txt', 'Blue.Valentine.2010.txt', 'Sweet.November.2001.txt', 'Wicker.Park.2004.txt']
worst_scripts = ['August.Rush.2007.txt', 'Tristan.and.Isolde.2006.txt', 'Jane.Eyre.2011.txt', 'Firelight.1997.txt', 'Original.Sin.2001.txt', 'Drive.1997.txt', 'Tuck.Everlasting.txt', 'dakota-skye.txt', 'Seven.Samurai.1954.txt', 'The.Lake.House.2006.txt']
all_scripts = os.listdir('cleanedSRT')

external_raw = []
for script in best_scripts:
    file = open('cleanedSRT/'+script, 'r')
    external_raw.append(file.readlines())
    file.close()
    
external = []

punctuation = "\"#$%&'()*+,-/:;<=>@[\]^_`{|}~"
end_tokens = [".", "?", "!"]

for script in external_raw:
    external.append(strip_line(script[0]))

print("generating models")  
print("generating external standard markov model")   
ex = MarkovModel("ex", external, n=2, smooth_param=0)
print("generating primary standard markov model")   
mm = MarkovModel("fc", FullCorpus.listOfLines, n=2, smooth_param=0)

print("generating M1")   
combo1 = WeightedComboMarkovModel(mm, ex, weight=0.9)
print("generating M2")
combo2 = NormalizedComboMarkovModel(mm, ex, weight=0.9)

print("MARKOV SAMPLES\n\n")
for _ in range(2):
    print(" ".join(mm.generate()), "\n\n")
print("M1 SAMPLES\n\n")
for _ in range(2):
    print(" ".join(combo1.generate()), "\n\n")
print("M2 SAMPLES\n\n")
for _ in range(2):
    print(" ".join(combo2.generate()), "\n\n")

print("calculating markov perplexity")
print("Markov Perplexity: ", ModelPerplexity(mm.initial_dist))

print("calculating weight perplexity")
print("Weight Perplexity: ", ModelPerplexity(combo1.initial_dist))

print("calculating normal perplexity")
print("Normal Perplexity: ", ModelPerplexity(combo2.initial_dist))

mm_samples = []
combo2_samples = []
combo1_samples = []

for _ in range(200):
    mm_samples.append(mm.generate())
    combo1_samples.append(combo1.generate())
    combo2_samples.append(combo2.generate())
    
mm_bleu_scores = []
combo1_bleu_scores = []
combo2_bleu_scores = []

reference = FullCorpus.listOfLines

for i in range(200):
    mm_bleu_scores.append(sentence_bleu(reference, mm_samples[i], weights=(0, 1, 0, 0)))
    combo1_bleu_scores.append(sentence_bleu(reference, combo1_samples[i], weights=(0, 1, 0, 0)))
    combo2_bleu_scores.append(sentence_bleu(reference, combo2_samples[i], weights=(0, 1, 0, 0)))

print("calculating markov bleu")
print("Markov Bleu: ", np.mean(mm_bleu_scores), "+/-", np.std(mm_bleu_scores), "\n\n")

print("calculating weight bleu")
print("Weight Bleu: ", np.mean(combo1_bleu_scores), "+/-", np.std(combo1_bleu_scores), "\n\n")

print("calculating normal bleu")
print("Normal Bleu: ", np.mean(combo2_bleu_scores), "+/-", np.std(combo2_bleu_scores), "\n\n")

