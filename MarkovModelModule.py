# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:15:46 2018

@author: Tristan Sparks
@author: Mahyar Bayran
"""

from nltk.tokenize import sent_tokenize
from nltk.util import ngrams

import numpy as np 
import os.path
import string
from random import randint
import random

def random_choice(options, probabilities):
    #sort probabilities in descending order and pick with probability 1/randomness
    d = {}
    for i in range(len(options)):
        d[options[i]] = probabilities[i]

    sorted_by_value = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    
    cumm_sorted = [0]
    for i in range(len(sorted_by_value)):
        cumm_sorted.append(cumm_sorted[-1] + sorted_by_value[i][1])
    
    random_num = random.uniform(0, 1)
    
    for i in range(len(sorted_by_value)):
        if ( (random_num >= cumm_sorted[i]) & (random_num <= cumm_sorted[i+1]) ):
            index = i
            print(index)
            break

    return sorted_by_value[index][0]
    
class MarkovModel:    
    def __init__(self, name, listOfLines, n, smooth_param=1):
        self.name = name
        self.listOfLines = listOfLines
        self.n = n
        self.smooth_param = smooth_param
        self.states = {}
        self.wordCount = 0
        
        # initialize states, map from word to index of word in distributions for convienence
        for line in listOfLines:
            for word in line:
                if (word not in self.states.keys()):
                    self.states[word] = self.wordCount
                    self.wordCount += 1

        print("wordcount ", self.wordCount)
        self.shape = [ self.wordCount for _ in range(self.n) ]
        self.initial_dist = np.zeros(self.wordCount)
        self.transition = np.zeros(self.shape)
        self.calc_initial()
        self.calc_transition()
        
    def calc_initial(self):
        for line in self.listOfLines:
            self.initial_dist[self.states[line[0]]] += 1 
            
        self.initial_dist = np.array([ elem + self.smooth_param for elem in self.initial_dist ])
        self.initial_dist /= ( len(self.initial_dist) * self.smooth_param + len(self.listOfLines) )
        
    def calc_transition(self):
        # First Order Markov
        if (self.n == 2):
            for line in self.listOfLines:
                for i in range(len(line) - (self.n - 1)):
                    self.transition[self.states[line[i]]][self.states[line[i+1]]] += 1
            for i in range(len(self.transition)):
                corpus_size = self.transition[i].sum() # corpus size is the # pairs of states where i is the first element in the pair
                self.transition[i] = [ elem + self.smooth_param for elem in self.transition[i] ]
                self.transition[i] /= ( corpus_size + self.smooth_param * self.wordCount)
        
        # Second Order Markov
        elif (self.n == 3):
            for line in self.listOfLines:
                for i in range(len(line) - (self.n - 1)):
                    self.transition[self.states[line[i]]][self.states[line[i+1]]][self.states[line[i+2]]] += 1
            for i in range(self.wordCount):
                for j in range(self.wordCount):
                    corpus_size = self.transition[i][j].sum() # corpus size is the # triplets of states where i,j are the first elements in the triplet
                    self.transition[i][j] = [ elem + self.smooth_param for elem in self.transition[i][j] ]
                    self.transition[i][j] /= ( corpus_size + self.smooth_param * self.wordCount)
        else:
            print("n must be 2 or 3")
            return
        
    def write_info(self, file):
        file.write('States: \n'+ str(list(self.states.items())) +'\n\n\n')
        file.write('Initial Distributions: \n' + str(self.initial_dist.tolist()) + '\n\n\n')
        file.write('Transition Probabilities: \n' + str(self.transition.tolist() )+ '\n\n\n')
        
        
    def generate(self, word_limit=20, overflow_allowed=20):
        end_tokens = [".", "?", "!"]
        
        words = [0 for _ in range(len(self.states))]
        for elem in self.states.items():
            words[elem[1]] = elem[0]
        
        # how to pick seed words for n=3?
        first_word = random_choice(words, self.initial_dist)
        generated_sent = [first_word]
        if (self.n == 3):
            # for now pick second word from random line that begins with the first word
            lines = []
            for line in self.listOfLines:
                if (line[0] == first_word):
                    lines.append(line)
                    
            line = random.choice(lines)
            generated_sent.append(line[1])
        
        for i in range(self.n - 1, word_limit):
            if (self.n == 2):
                probs = self.transition[self.states[generated_sent[i-1]]]
            elif (self.n == 3):
                probs = self.transition[self.states[generated_sent[i-2]]][self.states[generated_sent[i-1]]]
            
            generated_sent.append(random_choice(words, probs))
            
        i = word_limit
        while(generated_sent[-1] not in end_tokens and len(generated_sent) <= word_limit + overflow_allowed):
            if (self.n == 2):
                probs = self.transition[self.states[generated_sent[i-1]]]
            elif (self.n == 3):
                probs = self.transition[self.states[generated_sent[i-2]]][self.states[generated_sent[i-1]]]
            
            generated_sent.append(random_choice(words, probs))
            i += 1
            
        return generated_sent


#######Both of these might be stupid########################
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
        absolute_max = word_limit + 20
        
        words = [0 for _ in range(len(self.states))]
        
        for elem in self.states.items():
            words[elem[1]] = elem[0]
    
        first_word = np.random.choice(words, p=self.initial_dist)
        
        generated_sent = [first_word]
        
        prev = first_word
        for _ in range(word_limit - 1):
            probs = self.transition[self.states[prev]]
            new = random_choice(words, probs)
            generated_sent.append(new)
            
            prev = new
            
        while(new not in end_tokens and absolute_max > 0):
            absolute_max -= 1
            probs = self.transition[self.states[prev]]
            new = random_choice(words, probs)
            generated_sent.append(new)
            
            prev = new
            
        return generated_sent
        

class ComboMarkovModel:
    # takes two markov models and the weight of the first one
    def __init__(self, mm1, mm2, weight):
        self.mm1 = mm1
        self.mm2 = mm2
        self.weight = weight
        self.states = {}
        
        index = 0
        for word in mm1.states.keys():
            self.states[word] = index
            index += 1
        for word in mm2.states.keys():
            if word not in self.states.keys():
                self.states[word] = index
                index += 1

        self.calc_initial()
        self.calc_transition()
        
    def calc_initial(self):
        self.initial_dist = np.zeros(len(self.states))
        for gram, index in self.states.items():
            if gram in self.mm1.states.keys():
                i1 = self.mm1.initial_dist[self.mm1.states[gram]]
            else:
                i1 = 0
            if gram in self.mm2.states.keys():
                i2 = self.mm2.initial_dist[self.mm2.states[gram]]
            else:
                i2 = 0  
            
            self.initial_dist[index] = i1 * self.weight + i2 * (1 - self.weight)
            
        
    def calc_transition(self):
        self.transition = np.zeros( (len(self.states), len(self.states)) )
        
        for gram1, index1 in self.states.items():
           # print("w1: ", word1)
            for gram2, index2 in self.states.items():
                if (gram1 in self.mm1.states.keys() and gram2 in self.mm1.states.keys()):
                    t1 = self.mm1.transition[self.mm1.states[gram1]][self.mm1.states[gram2]]
                else:
                    t1 = 0
                if (gram1 in self.mm2.states.keys() and gram2 in self.mm2.states.keys()):
                    t2 = self.mm2.transition[self.mm2.states[gram1]][self.mm2.states[gram2]]
                else:
                    t2 = 0
                
                
                if (gram1 not in self.mm1.states.keys()):
                    self.transition[index1][index2] = t2
                elif (gram1 not in self.mm2.states.keys()):
                    self.transition[index1][index2] = t1
                else:
                    self.transition[index1][index2] = t1 * self.weight + t2 * (1 - self.weight)
                    
    def generate(self, word_limit=20):
        
        end_tokens = [".", "?", "!"]
        absolute_max = word_limit + 20
        
        #grams = [0 for _ in range(len(self.states))]
        #for elem in self.states.items():
        #    gram[elem[1]] = elem[0]

        grams = list(self.states.keys())
    
        first_gram = random_choice(grams, self.initial_dist)
        
        generated_sent = list(first_gram)
        
        prev = first_gram
        for _ in range(word_limit):
            probs = self.transition[self.states[prev]]
            new = random_choice(grams, probs)
            generated_sent.append(new[-1])          
            prev = new
            
        while(new[-1] not in end_tokens and absolute_max > 0):
            absolute_max -= 1
            probs = self.transition[self.states[prev]]
            new = random_choice(grams, probs)
            generated_sent.append(new[-1])     
            prev = new
            
        return generated_sent
        
