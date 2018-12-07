# -*- coding: utf-8 -*-
"""
@author: Tristan Sparks
@author: Mahyar Bayran
"""

from nltk.tokenize import sent_tokenize

import numpy as np 
import os.path
import string


class MarkovModel:

    def __init__(self, name, listOfLines):
        self.name = name
        self.states = {} #a mapping from the word to number (index of the word in states vector)
        self.listOfLines = listOfLines
        self.wordCount = 0
        
        # create mapping for states
        for line in listOfLines:
            for w in line:
                if w not in list(self.states.keys()): #new word found
                    self.states[w] = self.wordCount
                    self.wordCount += 1
                    
        self.calc_initial(1)
        self.calc_transition(1)
        self.generate()
        self.generate_deterministic()
        
    # with mappings we can easily build initial distributions and transition matrix
    def calc_initial(self, smooth_param=0):
        #### For Initial_dist, keep a list of possible first words and then cound and normalize
        self.initial_dist = np.zeros( self.wordCount ) # same length as states

        for line in self.listOfLines:
            self.initial_dist[self.states[line[0]]] += 1 
            
        self.initial_dist = np.array( [ elem + smooth_param for elem in self.initial_dist ] )
        self.initial_dist /= ( len(self.initial_dist) * smooth_param + len(self.listOfLines) )
    
    def calc_transition(self, smooth_param=0):
        self.transition = np.zeros( (self.wordCount, self.wordCount) ) # Transition probabilities
        
        for line in self.listOfLines:
            for i in range(len(line) - 1):
                self.transition[ self.states[line[i]], self.states[line[i+1]] ] += 1
                
        for i in range(len(self.transition)):
            corpus_size = self.transition[i].sum()
            self.transition[i] = [ elem + smooth_param for elem in self.transition[i] ]
            self.transition[i] /= ( corpus_size + smooth_param * self.wordCount)
        
    def write_info(self, file):
        file.write('States: \n'+ str(list(self.states.items())) +'\n\n\n')
        file.write('Initial Distributions: \n' + str(self.initial_dist.tolist()) + '\n\n\n')
        file.write('Transition Probabilities: \n' + str(self.transition.tolist() )+ '\n\n\n')
        
        
    def generate(self, word_limit=20):
        end_tokens = [".", "?", "!"]
        
        words = [0 for _ in range(len(self.states))]
        
        for elem in self.states.items():
            words[elem[1]] = elem[0]
    
        first_word = np.random.choice(words, p=self.initial_dist)
        
        generated_sent = [first_word]
        
        prev = first_word
        for _ in range(word_limit - 1):
            probs = self.transition[self.states[prev]]
            new = np.random.choice(words, p=probs)
            generated_sent.append(new)
            
            prev = new
            
        while(new not in end_tokens):
            new = words[self.transition[self.states[prev]].tolist().index(max(self.transition[self.states[prev]]))]
            generated_sent.append(new)
            
            prev = new
            
        return generated_sent
        
    def generate_deterministic(self, word_limit=20):
        end_tokens = [".", "?", "!"]
        
        words = [0 for _ in range(len(self.states))]
        
        for elem in self.states.items():
            words[elem[1]] = elem[0]
    
        first_word = np.random.choice(words, p=self.initial_dist)
        
        generated_sent = [first_word]
        
        prev = first_word
        for _ in range(word_limit - 1):
            new = words[self.transition[self.states[prev]].tolist().index(max(self.transition[self.states[prev]]))]
            generated_sent.append(new)
            
            prev = new
            
        while(new not in end_tokens):
            new = words[self.transition[self.states[prev]].tolist().index(max(self.transition[self.states[prev]]))]
            generated_sent.append(new)
            
            prev = new
            
        return generated_sent      
    
class Character:
    def __init__(self, name, firstlines=[]):
        self.name = name 
        self.listOfLines = []
        self.MM = []
        
        if firstlines != []:
            self.addToLines(firstlines)
                    
    def addToLines(self, newlines):
        if isinstance(newlines[0], list):
            for line in newlines:
                self.listOfLines.append(line)
        else:
            self.listOfLines.append(newlines) #a list of lists
            
    def BuildAMarkovModel(self):
        self.MM = MarkovModel(self.name, self.listOfLines)

    def write_info(self):
        filename = "%s.txt"%self.name
        completeName = os.path.join(os.getcwd() + '\\Characters', filename)
        file = open(completeName, "w")
        file.write("Character's Name:\t%s\n\n\n"%self.name)
        self.MM.write_info(file)
        
######### PRE-PROCESSING ###################

def strip_line_with_sentences(line):
    a = line.index(':')
    name = line[0:a]
    l = line[a+2:] # -1 or nothing?
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
    l = line[a+2:] # -1 or nothing?

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

#for sent in Characters['Denny'].listOfLines:
#    print(sent)

############## Markov Model ###########

#for name in charNames:
#    Characters[name].BuildAMarkovModel()
#    Characters[name].write_info()
        
Characters['Johnny'].BuildAMarkovModel()


FullCorpus.BuildAMarkovModel()

for _ in range(10):
    print(FullCorpus.MM.generate(8))
