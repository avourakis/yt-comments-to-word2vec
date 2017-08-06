import numpy as np
import math
import os
import random
import zipfile
import tensorflow as tf
import json
import collections

"""
Reference: https://www.tensorflow.org/tutorials/word2vec
"""

## READ DATA ##

#Note: Comments can also be seen as documents and vice versa

def read_comments_from_json(filename):
    """Read youtube comments from json file and return a list of each comment as a list of words."""  
    comments_vocabulary = []
    filename += '.json' # Add extension to filename
    file_path = os.path.join('.', filename)
    with open(file_path) as file:
        data = json.load(file) #deserialize to python object
        for comment in data:
            # Each comment will represent a document which can be used to implement Doc2Vec
            comments_vocabulary.append(comment["commentText"].split())
    return comments_vocabulary


filename = 'comments'
comments_vocabulary = read_comments_from_json(filename)
data_size = len(comments_vocabulary)
print('Data size', data_size)
print('Comments preview:\n', comments_vocabulary[:5])

## BUILD DICTIONARY ##

vocabulary_size = 10 # This number will depend on the max size of a youtube comment

def build_dataset(documents_vocabulary, n_words):
    """Process raw inputs into datasets"""
    documents_dataset = {'data': list(), 'count': list(), 'dictionary': list(), 'reversed_dictionary': list()}
    for words in documents_vocabulary:
        count = [['UNK', -1]] #Keeps track of common terms and unknow terms along with their count
        count.extend(collections.Counter(words).most_common(n_words - 1)) # extends "count" by adding the n_words (n most common words) found in each document
        dictionary = dict() #Keeps track of words found in count along with their id. 
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list() #keeps track of the id of the words that appear in the dictinary in the order they appear in the vocabulary
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count # updata 'UNK' to reflect the number of unknown terms found so far
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        documents_dataset['data'].append(data)
        documents_dataset['count'].append(count)
        documents_dataset['dictionary'].append(dictionary)
        documents_dataset['reversed_dictionary'].append(reversed_dictionary)
        
    return documents_dataset
    
documents_dataset = build_dataset(comments_vocabulary, vocabulary_size)

del comments_vocabulary #Not needed anymore so delete to reduce memory   
print('Data in first 5 documents', documents_dataset['data'][:5])
print('Most common words (+UNK) in first 5 documents', documents_dataset['count'][:5])
#print('Keys', documents_dataset.keys())
#print('Values', documents_dataset.values())
print(len(documents_dataset['data']))

## GENERATE TRAINING BATCH (SKIP-GRAM MODEL) ##

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    documents_batches = {'batch': list(), 'labels': list()}
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    for document_n in range(data_size):

        data = documents_dataset['data'][document_n] #Data from each document (total of data_size documents)
        if(len(data) > 10): #TODO: Take care of documents that don't contain enough data. Some documents could be single word
            
            batch = np.ndarray(shape=(batch_size), dtype=np.int32)
            labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32) # shape=(rows,cols)
            span = 2* skip_window + 1 # [skip_window target skip_window]
            buffer = collections.deque(maxlen=span) #Keeps track all words being analized during each iteration

            print("Size of data", len(data))
            print("Span", span)
            for _ in range(span): # Buffers only the words to be looked at. Based on span (depends on size of skip window). 
                print("Before", data_index)
                buffer.append(data[data_index])
                print("data: ", data[data_index])
                data_index = (data_index + 1) % len(data) #increments by 1 until it reaches the end of data in document
                print("After", data_index)
            for i in range(batch_size // num_skips): # // Divides with integral result (I forget!)
                target = skip_window # Word in middle of window. target label at the center of the buffer
                targets_to_avoid = [skip_window] # Do not consider word in middle of window (target)
                for j in range(num_skips):
                    while target in targets_to_avoid: #Starts as true 
                        target = random.randint(0, span - 1) #NOT SURE
                    targets_to_avoid.append(target)
                    batch[i * num_skips + j] = buffer[skip_window] #Save the target word in batch
                    labels[i* num_skips + j, 0] = buffer[target] #Save target in label. Target is not necessarely the same as target word

                buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data)
                print("Last change: ", data_index)

            documents_batches['batch'].append(batch)
            documents_batches['labels'].append(labels)

        #data_index = (data_index + len(data) - span) % len(data) # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = 0 # Fixed out of index error. VERIFY!!
        
    return documents_batches
        
documents_batches = generate_batch(8,2,1)

for i in range(len(documents_batches['batch'])):
    for j in range(8):
        batch = documents_batches['batch'][i]
        labels = documents_batches['labels'][i]
        reversed_dictionary = documents_dataset['reversed_dictionary'][i]
        print(batch[j], reversed_dictionary[batch[j]], '->',
                labels[j,0], reversed_dictionary[labels[j, 0]])

## BUILD AND TRAIN SKIP-GRAM MODEL ##

batch_size = 128
embeddings_size = 128 #Dimension of embedding vector
skip_window = 1 # How many words to consider left and right
num_skips = 2 #How many times to reuse and input to generate a label
