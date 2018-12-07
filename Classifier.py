'''
Find movies that are similar in scripts to the original movie
'''

import os
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.util import ngrams


def ROUGE(A, B):
    # A: list of reference n-grams
    # B: list of system n-grams
    count = 0
    for bi in A:
        if bi in B:
            count += 1
    return count/len(A)

def remove_punc(s):

    table = str.maketrans('', '', string.punctuation)
    new_s = s.translate(table)
    return new_s
    
def get_bigrams(myString):

    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(myString)
    stemmer = PorterStemmer()
    bigrams =  list(ngrams(tokens, 2)) 
    #bigram_finder = BigramCollocationFinder.from_words(tokens)
    #bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)
    
    for bigram_tuple in bigrams:
        x = "%s %s" % bigram_tuple
        tokens.append(x)
    
    result = [' '.join([stemmer.stem(w).lower() for w in x.split()]) for x in tokens if x.lower() not in stopwords.words('english') and len(x) > 8]
    return result

def find_similar_scripts(method):

    ### clean the original and extract bigrams
    file = open('corpus.txt', 'r')
    corpus = file.readlines()
    file.close()

    bigrams_corpus = []
    for line in corpus:
        tmp = line.find(':')
        text = line[tmp+2:].strip().lower()
        text = remove_punc(text)
        bigrams_line = get_bigrams(text)
        bigrams_corpus += bigrams_line
    ### load other scripts and do the same
    chosen_ones = []
    threshold = 0.1
    
    names = os.listdir('cleanedSRT')

    for name in names:
        print('checking: '+ name + '    ...')
        file = open('cleanedSRT/'+name, 'r')
        # there is one line only
        text = file.readline()
        text = remove_punc(text) #already did in cleansrt.py
        bigrams_ex = get_bigrams(text)
        score = ROUGE(bigrams_corpus, bigrams_ex)
        print(score)
        if score >= threshold:
            chosen_ones.append(name)
    
    return chosen_ones

method = 'ROUGE'
print(find_similar_scripts(method))
