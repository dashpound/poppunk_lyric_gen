# =============================================================================
# Poppunker
# =============================================================================


# =============================================================================
# Establish working environment
# =============================================================================

import multiprocessing
import re,string
import os
from pprint import pprint
import json

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer,\
    CountVectorizer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_decomposition import CCA  # canonical correlation
from sklearn.model_selection import train_test_split

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


import nltk
from nltk.stem import PorterStemmer

# =============================================================================
# Set global variables
# =============================================================================

stoplist = nltk.corpus.stopwords.words('english')
DROP_STOPWORDS = False
SET_RANDOM = 9999
STEMMING = False  # judgment call, parsed documents more readable if False
MAX_NGRAM_LENGTH = 2  # try 1 and 2 and see which yields better modeling results
VECTOR_LENGTH_LIST = [8, 16, 32, 64, 128, 256, 512]  # set vector length for TF-IDF and Doc2Vec
WRITE_VECTORS_TO_FILE = True

# JSON lines file for storing canonical correlatin results across many runs
cancor_results_file = open('cancor-results-file.jsonlines', 'a+') # open new file or append to existing

#%%
# =============================================================================
# Utility Functions 
# =============================================================================

# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ['\r', '\t']    


# text parsing function for entire document string
def parse_doc(text):
 #  text = text.lower()
 #   text = re.sub(r'&(.)+', "", text)  # no & references  
#    text = re.sub(r'pct', 'percent', text)  # replace pct abreviation  
#    text = re.sub(r"[^\w\d'\s]+", '', text)  # no punct except single quote 
#    text = re.sub(r'[^\x00-\x7f]',r'', text)  # no non-ASCII strings    
#    if text.isdigit(): text = ""  # omit words that are all digits    
    for code in codelist:
        text = re.sub(code, ' ', text)  # get rid of escape codes  
    # replace multiple spacess with one space
    text = re.sub('\s+', ' ', text)        
    return text

# text parsing for words within entire document string
# splits the document string into words/tokens
# parses the words and then recreates a document string
# returns list of parsed words/tokens and parsed document string
# =============================================================================
def parse_words(text): 
     # split document into individual words
     tokens=text.split('\n')
#     re_punc = re.compile('[%s]' % re.escape(string.punctuation))
#     # remove punctuation from each word
#     tokens = [re_punc.sub('', w) for w in tokens]
#     # remove remaining tokens that are not alphabetic
#     tokens = [word for word in tokens if word.isalpha()]
#     # filter out tokens that are one or two characters long
#     tokens = [word for word in tokens if len(word) > 2]
#     # filter out tokens that are more than twenty characters long
#     tokens = [word for word in tokens if len(word) < 21]
#     # filter out stop words if requested
#     if DROP_STOPWORDS:
#         tokens = [w for w in tokens if not w in stoplist]         
#     # perform word stemming if requested
#     if STEMMING:
#         ps = PorterStemmer()
#         tokens = [ps.stem(word) for word in tokens]
#     # recreate the document string from parsed words
#     text = ''
#     for token in tokens:
#         text = text + ' ' + token
     return tokens, text 
# =============================================================================

#%%     
# =============================================================================
# Import data from JSON lines file
# =============================================================================

# identify directory JSON lines files 
docdir = r'C:\Users\johnk\Desktop\Grad School\6. Spring 2019\1. MSDS_453_NLP\6. Homework\week8\scraper_bot\lyrics'

print('\nList of file names in the data directory:\n')
print(os.listdir(docdir))

all_data = []

for file in os.listdir(docdir): 
    if file.endswith('.jsonlines'):
        file_name = file.split('.')[0]  # keep name without extension
        with open(os.path.join(docdir,file), 'rb') as f:
            for line in f:
                all_data.append(json.loads(line))


#%%
# =============================================================================
# Unpack the list of dictionaries to create data frame
# =============================================================================

a = []
url = []
title = []
text = []
final_processed_tokens = []  # list of token lists for Doc2Vec

for doc in all_data:
    url.append(doc['url'])
    title.append(doc['title'])
    text_string = doc['text']
    tokens, text_string = parse_words(text_string)
    for line in tokens:
        a.append(parse_doc(line))
        a = list(filter(lambda f: f != " ", a))
        a = list(filter(lambda f: f != "", a))
        a = [x for x in a if "album:" not in x]
        a = [x for x in a if "EP:" not in x]
    final_processed_tokens.append(a)
    


#%%
len(final_processed_tokens)
final_processed_tokens[:1]
#%%
df = pd.DataFrame({"url": url,
                   "title": title,
                   "text": final_processed_tokens
                   },)

#the following is an example of what the processed text looks like.  
print('\nBeginning and end of the data frame:\n')

#%%
df['text'][0]

#%%

s = df.apply(lambda x:pd.Series(x['text']), axis=1).stack().reset_index(level=1, drop=True)
s.name= 'texts'

df = df.drop('text', axis=1).join(s)
#df = df.reset_index()

#%%

df.to_csv(r'C:\Users\johnk\Desktop\Grad School\6. Spring 2019\1. MSDS_453_NLP\6. Homework\week8\preprocessing\df.csv')


