# =============================================================================
# 
# =============================================================================

# Import the dependencies
from textgenrnn import textgenrnn
import numpy as np
import pandas as pd
import sys 
from keras.models import Sequential
from keras.layers import LSTM, Activation, Flatten, Dropout, Dense, Embedding, TimeDistributed, CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#textgen =textgenrnn()
#textgen.generate()

#Load the dataset
dataset = pd.read_csv(r'C:\Users\johnk\Desktop\Grad School\6. Spring 2019\1. MSDS_453_NLP\6. Homework\week8\preprocessing\df.csv', encoding = "latin1")

# =============================================================================
# 
# =============================================================================

def processFirstLine(lyrics, songID, songName, row):
    lyrics.append(row['texts'] + '\n')
    songID.append( row['url']*100)
    songName.append(row['title'])
    return lyrics,songID,songName

# =============================================================================
# 
# =============================================================================

# define empty lists for the lyrics , songID , songName 
lyrics = []
songID = []
songName = []

# songNumber indicates the song number in the dataset
songNumber = 1

# i indicates the song number
i = 0
isFirstLine = True

# Iterate through every lyrics line and join them together for each song independently 
for index,row in dataset.iterrows():
    if(songNumber == row[0]):
        if (isFirstLine):
            lyrics,songID,songName = processFirstLine(lyrics,songID,songName,row)
            isFirstLine = False
        else :
            #if we still in the same song , keep joining the lyrics lines    
            lyrics[i] +=  row['texts'] + '\n'
    #When it's done joining a song's lyrics lines , go to the next song :    
    else :
        lyrics,songID,songName = processFirstLine(lyrics,songID,songName,row)
        songNumber = row[0]
        i+=1

# =============================================================================
#         
# =============================================================================

# Define a new pandas DataFrame to save songID , songName , Lyrics in it to use them later
lyrics_data = pd.DataFrame({'songID':songID, 'songName':songName, 'lyrics':lyrics })

# =============================================================================
# 
# =============================================================================

# Save Lyrics in .txt file
with open('lyricsText.txt', 'w',encoding="utf-8") as filehandle:  
    for listitem in lyrics:
        filehandle.write('%s\n' % listitem)
        
# =============================================================================
#         
# =============================================================================

# Load the dataset and convert it to lowercase :
textFileName = 'lyricsText.txt'
raw_text = open(textFileName, encoding = 'UTF-8').read()
raw_text = raw_text.lower()

# =============================================================================
# 
# =============================================================================

# Mapping chars to ints :
chars = sorted(list(set(raw_text)))
int_chars = dict((i, c) for i, c in enumerate(chars))
chars_int = dict((i, c) for c, i in enumerate(chars))

# =============================================================================
# 
# =============================================================================

# Get number of chars and vocab in our text :
n_chars = len(raw_text)
n_vocab = len(chars)

# =============================================================================
# 
# =============================================================================

print('Total Characters : ' , n_chars) # number of all the characters in lyricsText.txt
print('Total Vocab : ', n_vocab) # number of unique characters

# =============================================================================
# 
# =============================================================================

# process the dataset:
seq_len = 100
data_X = []
data_y = []

for i in range(0, n_chars - seq_len, 1):
    # Input Sequeance(will be used as samples)
    seq_in  = raw_text[i:i+seq_len]
    # Output sequence (will be used as target)
    seq_out = raw_text[i + seq_len]
    # Store samples in data_X
    data_X.append([chars_int[char] for char in seq_in])
    # Store targets in data_y
    data_y.append(chars_int[seq_out])
n_patterns = len(data_X)
print( 'Total Patterns : ', n_patterns)

# =============================================================================
# 
# =============================================================================

# Reshape X to be suitable to go into LSTM RNN :
X = np.reshape(data_X , (n_patterns, seq_len, 1))
# Normalizing input data :
X = X/ float(n_vocab)
# One hot encode the output targets :
y = np_utils.to_categorical(data_y)

# =============================================================================
# 
# =============================================================================
LSTM_layer_num = 4 # number of LSTM layers
layer_size = [256,256,256,256] # number of nodes in each layer

model = Sequential()

model.add(CuDNNLSTM(layer_size[0], input_shape =(X.shape[1], X.shape[2]), return_sequences = True))

for i in range(1,LSTM_layer_num) :
    model.add(CuDNNLSTM(layer_size[i], return_sequences=True))
    
model.add(Flatten())

# =============================================================================
# 
# =============================================================================

model.add(Dense(y.shape[1]))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

model.summary()

# =============================================================================
# 
# =============================================================================


# Configure the checkpoint :
checkpoint_name = 'Weights-LSTM-improvement-{epoch:03d}-{loss:.5f}-bigger.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose = 1, save_best_only = True, mode ='min')
callbacks_list = [checkpoint]

# =============================================================================
# 
# =============================================================================


# Fit the model :
model_params = {'epochs':30,
                'batch_size':128,
                'callbacks':callbacks_list,
                'verbose':1,
                'validation_split':0.2,
                'validation_data':None,
                'shuffle': True,
                'initial_epoch':0,
                'steps_per_epoch':None,
                'validation_steps':None}

model.fit(X,
          y,
          epochs = model_params['epochs'],
           batch_size = model_params['batch_size'],
           callbacks= model_params['callbacks'],
           verbose = model_params['verbose'],
           validation_split = model_params['validation_split'],
           validation_data = model_params['validation_data'],
           shuffle = model_params['shuffle'],
           initial_epoch = model_params['initial_epoch'],
           steps_per_epoch = model_params['steps_per_epoch'],
           validation_steps = model_params['validation_steps'])

# =============================================================================
# 
# =============================================================================

# Load wights file :
wights_file = './models/Weights-LSTM-improvement-004-2.49538-bigger.hdf5' # weights file path
model.load_weights(wights_file)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# =============================================================================
# 
# =============================================================================

# set a random seed :
start = np.random.randint(0, len(data_X)-1)
pattern = data_X[start]
print('Seed : ')
print("\"",''.join([int_chars[value] for value in pattern]), "\"\n")

# How many characters you want to generate
generated_characters = 300

# Generate Charachters :
for i in range(generated_characters):
    x = np.reshape(pattern, ( 1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x,verbose = 0)
    index = np.argmax(prediction)
    result = int_chars[index]
    #seq_in = [int_chars[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print('\nDone')