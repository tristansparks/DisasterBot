# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:15:46 2018

@author: Tristan Sparks
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
                    
        self.calc_initial()
        self.calc_transition()
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

class WeightedMarkovModel:
    def __init__(self, name, listOfLines, externalCorpus):
        # we are basically going to chuck out external data which is unrelated to our primary corpus
        # and then weight the resulting distributions
        self.name = name
        self.states = {} #a mapping from the word to index of the word in probability arrays
        self.inverseStates = {}
        self.listOfLines = listOfLines
        self.wordCount = 0
        self.externalCorpus = externalCorpus
        self.externalCounts = {}
        self.externalBigrams = {}
        
        # create mapping for states
        for line in listOfLines:
            for w in line:
                if w not in list(self.states.keys()): #new word found
                    self.states[w] = self.wordCount
                    self.wordCount += 1
                    
        for line in self.externalCorpus:
            # set up bigrams
            for i in range(len(line) - 1):
                if (line[i], line[i + 1]) not in self.externalBigrams.keys():
                    self.externalBigrams[(line[i], line[i + 1])] = 0
                self.externalBigrams[(line[i], line[i + 1])] += 1
            
            # set up common word counts
            for k, v in self.states.items():
                if (k not in self.externalCounts.keys()):
                    self.externalCounts[k] = 0
                
                self.externalCounts[k] += line.count(k)
                
        print("word count", self.wordCount)
        self.inverseStates = {v: k for k, v in self.states.items()}
        self.calc_initial(0.1)
        self.calc_transition(0.1)
        
                    
    def calc_initial(self, smooth_param=0, external_weight=0.5):
        # Use weighted average for now
        initial_dist = np.zeros( self.wordCount ) # same length as states
        ex_initial_dist = np.zeros( self.wordCount )
        self.initial_dist = np.zeros( self.wordCount )

        for line in self.listOfLines:
            initial_dist[self.states[line[0]]] += 1 
            
        used_line_count = 0
        for k in self.states.keys():
            c = [ line[0] for line in self.externalCorpus ].count(k)
            ex_initial_dist[self.states[k]] = c
            used_line_count += c
            
        initial_dist = np.array( [ elem + smooth_param for elem in initial_dist ] )
        initial_dist /= ( len(initial_dist) * smooth_param + len(self.listOfLines) )
        
        ex_initial_dist = np.array( [ elem + smooth_param for elem in ex_initial_dist ] )
        ex_initial_dist /= ( len(initial_dist) * smooth_param + used_line_count )
            
        for i in range(len(initial_dist)):
            self.initial_dist[i] = initial_dist[i] * (1 - external_weight) + ex_initial_dist[i] * external_weight
            
        
    def calc_transition(self, smooth_param=0, external_weight=0.5):
        self.transition = [ [] for _ in range(self.wordCount) ]
        transition = np.zeros( (self.wordCount, self.wordCount) ) # Transition probabilities
        ex_transition = np.zeros( ( self.wordCount, self.wordCount ) )
        
        for line in self.listOfLines:
            for i in range(len(line) - 1):
                transition[ self.states[line[i]], self.states[line[i+1]] ] += 1

        for i in range(len(transition)):            
            for j in range(len(transition)):
                # this is so gross come back and make it better
                if (self.inverseStates[i], self.inverseStates[j]) in self.externalBigrams.keys():
                    ex_transition[i][j] = self.externalBigrams[(self.inverseStates[i], self.inverseStates[j])]
                else:
                    ex_transition[i][j] = 0
                    
            corpus_size = transition[i].sum()
            ex_corpus_size = ex_transition[i].sum()
            transition[i] = [ elem + smooth_param for elem in transition[i] ]
            transition[i] /= ( corpus_size + smooth_param * self.wordCount )
            
            ex_transition[i] = [ elem + smooth_param for elem in ex_transition[i] ]
            ex_transition[i] /= ( ex_corpus_size + smooth_param * self.wordCount )
            
            self.transition[i] = transition[i] * (1 - external_weight) + ex_transition[i] * external_weight
                        
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
            print(new)
            new = words[self.transition[self.states[prev]].tolist().index(max(self.transition[self.states[prev]]))]
            generated_sent.append(new)
            
            prev = new
            
        return generated_sent