# =============================================================================
# Poppunker
# =============================================================================


# =============================================================================
# Establish working environment
# =============================================================================

import re,string
import os
from pprint import pprint
import json

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
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
cancor_results_file = open(r'C:\Users\johnk\Desktop\Grad School\6. Spring 2019\1. MSDS_453_NLP\6. Homework\week8\preprocessing\results\cancor-results-file.jsonlines', 'a+') # open new file or append to existing

#%%
# =============================================================================
# Utility Functions 
# =============================================================================

# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ['\r', '\n', '\t']    

# text parsing function for entire document string
def parse_doc(text):
    text = text.lower()
    text = re.sub(r'&(.)+', "", text)  # no & references  
    text = re.sub(r'pct', 'percent', text)  # replace pct abreviation  
    text = re.sub(r"[^\w\d'\s]+", '', text)  # no punct except single quote 
    text = re.sub(r'[^\x00-\x7f]',r'', text)  # no non-ASCII strings    
    if text.isdigit(): text = ""  # omit words that are all digits    
    for code in codelist:
        text = re.sub(code, ' ', text)  # get rid of escape codes  
    # replace multiple spacess with one space
    text = re.sub('\s+', ' ', text)        
    return text

# text parsing for words within entire document string
# splits the document string into words/tokens
# parses the words and then recreates a document string
# returns list of parsed words/tokens and parsed document string
def parse_words(text): 
    # split document into individual words
    tokens=text.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out tokens that are one or two characters long
    tokens = [word for word in tokens if len(word) > 2]
    # filter out tokens that are more than twenty characters long
    tokens = [word for word in tokens if len(word) < 21]
    # filter out stop words if requested
    if DROP_STOPWORDS:
        tokens = [w for w in tokens if not w in stoplist]         
    # perform word stemming if requested
    if STEMMING:
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
    # recreate the document string from parsed words
    text = ''
    for token in tokens:
        text = text + ' ' + token
    return tokens, text 

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

url = []
title = []
tags = []
text = []
labels = []
final_processed_tokens = []  # list of token lists for Doc2Vec
final_processed_text = [] # list of document strings for TF-IDF
labels = []  # use filenames as labels
for doc in all_data:
    url.append(doc['url'])
    title.append(doc['title'])
    text_string = doc['text']
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    text.append(text_string)
    final_processed_tokens.append(tokens)
    final_processed_text.append(text_string)

df = pd.DataFrame({"url": url,
                   "title": title,
                   "text": text
                   },)

#the following is an example of what the processed text looks like.  
print('\nBeginning and end of the data frame:\n')
print(df.head(2))
print(df.tail(2))

#%%
# =============================================================================
# Split the corpus into training & testing sets
# =============================================================================

train_data, test_data = train_test_split(all_data, random_state=1)

#%%
# =============================================================================
# Preprocess the training set; set asside labels
# =============================================================================
train_titles = []
train_tokens = []  # list of token lists for gensim Doc2Vec
train_text = [] # list of document strings for sklearn TF-IDF
train_target = []  # use filenames as labels
for doc in train_data:
    train_titles.append(doc['title'])
    text_string = doc['text']
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    train_tokens.append(tokens)
    train_text.append(text_string)
    
    
print('\nNumber of training documents:',
	len(train_text))	
#print('\nFirst item after text preprocessing, train_text[0]\n', 
#	train_text[0])
print('\nNumber of training token lists:',
	len(train_tokens))	
#print('\nFirst list of tokens after text preprocessing, train_tokens[0]\n', 
#	train_tokens[0])
#%%
# =============================================================================
# Spot check; confirm labels & titles match up
# =============================================================================

pprint(train_titles[:10])
pprint(train_target[:10])

#%%
# =============================================================================
# Preprocess the testing set; set asside labels
# =============================================================================
test_tokens = []  # list of token lists for gensim Doc2Vec
test_text = [] # list of document strings for sklearn TF-IDF
test_target= []  # use filenames as labels
test_titles = []

for doc in test_data:
    test_titles.append(doc['title'])
    text_string = doc['text']
    # parse the entire document string
    text_string = parse_doc(text_string)
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    test_tokens.append(tokens)
    test_text.append(text_string)


print('\nNumber of testing documents:',
	len(test_text))	
#print('\nFirst item after text preprocessing, test_text[0]\n', 
#	test_text[0])
print('\nNumber of testing token lists:',
	len(test_tokens))	
#print('\nFirst list of tokens after text preprocessing, test_tokens[0]\n', 
#	test_tokens[0])
#%%
# =============================================================================
# Spot check; confirm labels & titles match up
# =============================================================================

pprint(test_titles[:10])
pprint(test_target[:10])

#%%

# =============================================================================
# Perform TFIDF & Word2Vec canonical correlation analysis
# =============================================================================
 
# create list for saving canonical correlation results
cancor_results = [] 

for VECTOR_LENGTH in VECTOR_LENGTH_LIST: 
    print('\n---------- VECTOR LENGTH ', str(VECTOR_LENGTH), ' ----------\n')
    # =============================================================================
    # TF-IDF
    # =============================================================================
    # note the ngram_range will allow you to include multiple-word tokens 
    # within the TFIDF matrix
    # Call Tfidf Vectorizer
    print('\nWorking on TF-IDF vectorization')
    Tfidf = TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), 
    	max_features = VECTOR_LENGTH)

    # fit the vectorizer using final processed documents.  
    TFIDF_matrix = Tfidf.fit_transform(final_processed_text)  

    tfidf_solution = pd.DataFrame(TFIDF_matrix.toarray())  # for modeling work  

    #creating datafram from TFIDF Matrix
    matrix = pd.DataFrame(TFIDF_matrix.toarray(), 
    	columns = Tfidf.get_feature_names(), 
    	index = title)

    if WRITE_VECTORS_TO_FILE:
        tfidf_file_name = 'tfidf-matrix-'+ str(VECTOR_LENGTH) + '.csv'
        matrix.to_csv(r'C:/Users/johnk/Desktop/Grad School/6. Spring 2019/1. MSDS_453_NLP/6. Homework/week8/preprocessing/results/' + tfidf_file_name)
        print('\nTF-IDF vectorization complete, matrix saved to ', tfidf_file_name, '\n')

    # =============================================================================
    # gensim Doc2Vec
    # =============================================================================
        
    print("\nWorking on Doc2Vec vectorization")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(final_processed_tokens)]
    model = Doc2Vec(documents, vector_size = VECTOR_LENGTH, window = 2, 
    	min_count = 1, workers = 4)

    doc2vec_df = pd.DataFrame()
    for i in range(0,len(final_processed_tokens)):
        vector = pd.DataFrame(model.infer_vector(final_processed_tokens[i])).transpose()
        doc2vec_df = pd.concat([doc2vec_df,vector], axis=0)

    doc2vec_solution = doc2vec_df  # for modeling work

    doc2vec_df = doc2vec_df.reset_index()

    doc_titles = {'title': labels}
    t = pd.DataFrame(doc_titles)

    doc2vec_df = pd.concat([doc2vec_df,t], axis=1)

    doc2vec_df = doc2vec_df.drop('index', axis=1)
    doc2vec_df = doc2vec_df.set_index('title')

    if WRITE_VECTORS_TO_FILE:
        doc2vec_file_name = 'doc2vec-matrix-'+ str(VECTOR_LENGTH) + '.csv'
        doc2vec_df.to_csv(r'C:/Users/johnk/Desktop/Grad School/6. Spring 2019/1. MSDS_453_NLP/6. Homework/week8/preprocessing/results/'+ doc2vec_file_name)
        print('\nDoc2Vec vectorization complete, matrix saved to ', doc2vec_file_name, '\n')

    # =============================================================================
    # Canonical Correlation... show relationship between TF-IDF and Doc2Vec
    # =============================================================================

    n_components = 3
    cca = CCA(n_components)
    cca.fit(X = tfidf_solution, Y = doc2vec_solution)

    U, V = cca.transform(X = tfidf_solution, Y = doc2vec_solution)

    for i in range(n_components):
        corr = np.corrcoef(U[:,i], V[:,i])[0,1]

    print('\nCanonical correlation betwen TF-IDF and Doc2Vec for vectors of length ', 
        str(VECTOR_LENGTH), ':', np.round(corr, 3), '\n')

    cancor_results.append(np.round(corr, 3))

    data = json.dumps({"STEMMING":STEMMING,
        "MAX_NGRAM_LENGTH":MAX_NGRAM_LENGTH,
        "VECTOR_LENGTH":VECTOR_LENGTH,
        "CANCOR":np.round(corr, 3)}) 
    cancor_results_file.write(data)
    cancor_results_file.write('\n')

print('\nSummary of Canonoical Correlation between TF-IDF and Doc2Vec Vectorizations\n')
print('\nVector Length Correlation')
print('\n-------------------------')
for item in range(len(VECTOR_LENGTH_LIST)):
    print('     ', VECTOR_LENGTH_LIST[item], '      ', cancor_results[item])

cancor_results_file.close()

#%%

