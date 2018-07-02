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
import scipy.io

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import re 
import nltk, nltk.stem.porter


#%% functions

def preProcess(email):
    # Lower case the email
    email=email.lower()
    # Strip html tags. replace with a space
    email=re.sub('<[^<>]+>',' ',email);
    # Any numbers get replaced with the string 'number'
    email=re.sub('[0-9]+','number',email)
    # Anything starting with http or https:// replaced with 'httpaddr'
    email=re.sub('(http|https)://[^\s]*','httpaddr',email)
    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email=re.sub('[^\s]+@[^\s]+','emailaddr',email);
    # The '$' sign gets replaced with 'dollar'
    email=re.sub('[$]+','dollar',email);
    return email



def getVocabDict(reverse=False):
    vocab_dict={}
    with open('data/vocab.txt') as f:
        for line in f:
            (val,key)=line.split()
            if not reverse:
                vocab_dict[key]=int(val)
            else:
                vocab_dict[int(val)]=key
                
    return vocab_dict



def email2TokenList(raw_email):
    # Preprocess the email
    email=preProcess(raw_email)

    # Split the email into individual words (tokens)
    tokens=re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]',email)
    
    # Loop over each token and use a stemmer to shorten it, check if the word
    # is in the vocab_list... if it is, store index
    tokenlist=[]
    stemmer=nltk.stem.porter.PorterStemmer()
    for token in tokens:
        # remove all special characters
        token=re.sub('[^a-zA-Z0-9]','',token)
        # remove morphological affixes from words
        stemmed=stemmer.stem(token)
        # Throw out empty tokens
        if not len(stemmed): continue
        # Store a list of all unique stemmed words
        tokenlist.append(stemmed)
            
    return tokenlist


	
def email2VocabIndices(raw_email,vocab_dict):
    # Split the email in the separate words (tokens) --- stemming
    tokenlist=email2TokenList(raw_email)
    # Find each token in the vocabluary and save its index
    index_list=[vocab_dict[token] for token in tokenlist if token in vocab_dict]
    return index_list



def email2FeatureVector(raw_email,vocab_dict):
    # Returns a vector of the vocab_dict size. The element of the vector is
    # 0 if the corresponding word has not appeared in the email, 
    # and 1 if it is there.
    vocab_indices=email2VocabIndices(email_contents,vocab_dict)
    result=np.zeros(len(vocab_dict),dtype=bool)
    result[vocab_indices]=True
    return result

#%% Sec. 2.1 processing emails

vocab_dict=getVocabDict()
email_contents=open('data/emailSample1.txt','r').read()
test_fv=email2FeatureVector(email_contents,vocab_dict)

print("Length of feature vector is %d" % len(test_fv))
print("Number of non-zero entries is: %d" % sum(test_fv==1))


