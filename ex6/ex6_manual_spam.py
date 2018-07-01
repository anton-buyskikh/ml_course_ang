#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:40:43 2018

@author: Anton Buyskikh
@brief: Spam classification.
...
"""


#%% libraries

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import re 
import nltk, nltk.stem.porter


#%% functions

def preProcess( email ):
    email = email.lower()
    # Strip html tags. replace with a space
    email = re.sub('<[^<>]+>', ' ', email);
    #Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)
    #Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    #Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email);
    #The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email);
    return email



def getVocabDict(reverse=False):
    """
    Function to read in the supplied vocab list text file into a dictionary
    Dictionary key is the stemmed word, value is the index in the text file
    If "reverse", the keys and values are switched.
    """
    vocab_dict = {}
    with open('data/vocab.txt') as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key
                
    return vocab_dict



def email2TokenList( raw_email ):
    """
    Function that takes in preprocessed (simplified) email, tokenizes it,
    stems each word, and returns an (ordered) list of tokens in the e-mail
    """
    
    stemmer = nltk.stem.porter.PorterStemmer()
    email = preProcess( raw_email )

    #Split the e-mail into individual words (tokens) (split by the delimiter ' ')
    #Splitting by many delimiters is easiest with re.split()
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)
    
    #Loop over each token and use a stemmer to shorten it, check if the word is in the vocab_list... if it is, store index
    tokenlist = []
    for token in tokens:
      
        token = re.sub('[^a-zA-Z0-9]', '', token);
        stemmed = stemmer.stem( token )
        #Throw out empty tokens
        if not len(token): continue
        #Store a list of all unique stemmed words
        tokenlist.append(stemmed)
            
    return tokenlist


	
def email2VocabIndices( raw_email, vocab_dict ):
    #returns a list of indices corresponding to the location in vocab_dict for each stemmed word 
    tokenlist = email2TokenList( raw_email )
    index_list = [ vocab_dict[token] for token in tokenlist if token in vocab_dict ]
    return index_list



def email2FeatureVector( raw_email, vocab_dict ):
    # returns a vector of shape(n,1) where n is the size of the vocab_dict.
    #he first element in this vector is 1 if the vocab word with index == 1 is in raw_email, else 0
    n = len(vocab_dict)
    result = np.zeros((n,1))
    vocab_indices = email2VocabIndices( email_contents, vocab_dict )
    for idx in vocab_indices:
        result[idx] = 1
    return result

#%% PART I
# load data

vocab_dict = getVocabDict()
email_contents = open('data/emailSample1.txt', 'r' ).read()
test_fv = email2FeatureVector( email_contents, vocab_dict )

print("Length of feature vector is %d" % len(test_fv))
print("Number of non-zero entries is: %d" % sum(test_fv==1))


