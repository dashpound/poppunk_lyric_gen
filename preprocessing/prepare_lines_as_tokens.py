# =============================================================================
# Poppunker
# =============================================================================


# =============================================================================
# Establish working environment
# =============================================================================

import re
import os
import json

import pandas as pd

# =============================================================================
# Set global variables
# =============================================================================

SET_RANDOM = 9999

#%%
# =============================================================================
# Utility Functions 
# =============================================================================

# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ['\r', '\t']    

# text parsing function for entire document string
def parse_doc(text):
    for code in codelist:
        text = re.sub(code, ' ', text)  # get rid of escape codes  
    # replace multiple spacess with one space
    text = re.sub('\s+', ' ', text)        
    return text

#%%
# =============================================================================
# 
# =============================================================================

def parse_words(text): 
     # split document into individual words
     tokens=text.split('\n')
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
# =============================================================================
# 
# =============================================================================

len(final_processed_tokens)
final_processed_tokens[:1]
#%%
# =============================================================================
# 
# =============================================================================

df = pd.DataFrame({"url": url,
                   "title": title,
                   "text": final_processed_tokens
                   },)

#the following is an example of what the processed text looks like.  
print('\nBeginning and end of the data frame:\n')

#%%
df['text'][0]

#%%
# =============================================================================
# 
# =============================================================================
s = df.apply(lambda x:pd.Series(x['text']), axis=1).stack().reset_index(level=1, drop=True)
s.name= 'texts'

df = df.drop('text', axis=1).join(s)
#df = df.reset_index()

#%%
# =============================================================================
# 
# =============================================================================
df.to_csv(r'C:\Users\johnk\Desktop\Grad School\6. Spring 2019\1. MSDS_453_NLP\6. Homework\week8\preprocessing\df.csv')
print('\n-------------------------')
print('\nDataframe successfully written')
print('\n-------------------------')
