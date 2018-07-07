#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:40:43 2018

@author: Anton Buyskikh
@brief: Spam classification. Regular Expression Engine.
Natural Language Toolkit.
...
"""


#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import sklearn.svm
#from sklearn.metrics import accuracy_score

import re 
import nltk,nltk.stem.porter

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



def getVocabDict(filename):
    vocab_dict={}
    with open(filename) as f:
        for line in f:
            (val,key)=line.split()
            vocab_dict[key]=int(val)
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

vocab_dict=getVocabDict('data/vocab.txt')
email_contents=open('data/emailSample1.txt','r').read()
test_fv=email2FeatureVector(email_contents,vocab_dict)

print("Length of feature vector is %d" % len(test_fv))
print("Number of non-zero entries is: %d" % sum(test_fv==1))
print()

#%% Sec. 2.3 svm for spam classification
# load data

data_train=scipy.io.loadmat('data/spamTrain.mat')
X_train=data_train.get('X')
y_train=data_train.get('y').flatten()

data_test=scipy.io.loadmat('data/spamTest.mat')
X_test=data_test.get('Xtest')
y_test=data_test.get('ytest').flatten()

print('Total number of training emails = ',len(y_train))
print('Number of training    spam emails = ',np.count_nonzero(  y_train))
print('Number of training nonspam emails = ',np.count_nonzero(1-y_train))
print()

#%% training SVM

# In order to meet accuracy from the PDF:
# - training accuracy of about 99.8% 
# - test accuracy of about 98.5%"
# we choose C=0.1 and 'linear' kernel
svm=sklearn.svm.SVC(C=0.1,kernel='linear')
svm.fit(X_train,y_train)

y_train_pred=svm.predict(X_train)
train_acc=100.0*(y_train_pred==y_train).mean()
print('Training accuracy = %0.2f%%' % train_acc)

y_test_pred=svm.predict(X_test)
test_acc=100.0*(y_test_pred==y_test).mean()
print('Test set accuracy = %0.2f%%' % test_acc)
print()

#%% Find words which are more/less likely to be an identificator of spam

# create an inversed dictionary
vocab_dict_inv={v:k for k,v in vocab_dict.items()}

# sort indicies from the most important to least important
sorted_indices=np.argsort(svm.coef_,axis=None )[::-1]

fig,ax=plt.subplots()
ax.plot(svm.coef_[:,sorted_indices].reshape(-1),'.')
ax.set_xlabel('sorted indices')
ax.set_ylabel('SVM coefficients')
plt.show()

print("The 15 most  important words to classify a spam e-mail are:")
print([vocab_dict_inv[x] for x in sorted_indices[:15 ]])
print()
print("The 15 least important words to classify a spam e-mail are:")
print([vocab_dict_inv[x] for x in sorted_indices[-15:]])
print()

#%% Try test emails with already trained 

# just change the file name and test the algorithm
email_contents=open('data/spamSample2.txt','r').read()
X_sample=email2FeatureVector(email_contents,vocab_dict).reshape(1,-1)

print('###### Email: ###################')
print(email_contents)
print('#################################')
print('Length of feature vector is %d' % X_sample.size)
print('Number of non-zero entries is: %d' % sum(sum(X_sample==1)))
print(svm.predict(X_sample))
print()








