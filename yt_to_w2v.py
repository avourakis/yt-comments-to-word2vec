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
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys())) # Keep track of dictionary. Key is the id and Value is the word.
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
    documents_not_skipped = []
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    for document_n in range(data_size):

        data = documents_dataset['data'][document_n] #Data from each document (total of data_size documents)
        if(len(data) > 10): #TODO: Take care of documents that don't contain enough data. Some documents could be a single word
            
            documents_not_skipped.append(document_n) #Keeps track of index of documents for which batches are created
            batch = np.ndarray(shape=(batch_size), dtype=np.int32)
            labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32) # shape=(rows,cols)
            span = 2* skip_window + 1 # [skip_window target skip_window]
            buffer = collections.deque(maxlen=span) #Keeps track all words being analized during each iteration

            for _ in range(span): # Buffers only the words to be looked at. Based on span (depends on size of skip window). 
                buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data) #increments by 1 until it reaches the end of data in document

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

            # Context word (labels) from it's target word (batch).
            documents_batches['batch'].append(batch)
            documents_batches['labels'].append(labels)

        #data_index = (data_index + len(data) - span) % len(data) # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = 0 # Fixed "out of index error". VERIFY!!
        
    return documents_batches, documents_not_skipped
        
# For testing:

documents_batches, documents_not_skipped = generate_batch(batch_size=8, num_skips=2, skips_window=1)

for i in range(len(documents_batches['batch'])): #iterate through each document's data
    print("Viewing document ", i)
    batch = documents_batches['batch'][i]
    labels = documents_batches['labels'][i]
    reversed_index = documents_not_skipped[i] #Index of the documents not skipped during batch creation
    reversed_dictionary = documents_dataset['reversed_dictionary'][reversed_index]
    #print(reversed_dictionary)

    for j in range(8): #iterate through each document's batches
        print(batch[j], reversed_dictionary[batch[j]], '->',
                labels[j,0], reversed_dictionary[labels[j, 0]])

## BUILD AND TRAIN SKIP-GRAM MODEL ##

batch_size = 128 #Number of samples that are going to be propagated through the network
embedding_size = 128 #Dimension of embedding vector
skip_window = 1 # How many words to consider left and right
num_skips = 2 #How many times to reuse and input to generate a label

# We pick a random validation set to sample nearest neighbors. Here
# we limit the validation samples to the words that have low numering
# ID, which by construction are also the most frequent.

valid_size = 16 #Random set of words to evaluate similarity on
valid_window = 100 #Only pick dev samples in the head of distribution
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64 #Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

    '''
    tf.placeholder: used to feed actual training examples
    tf.Variable: trainable variables

    '''
    
    #input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size]) #batch
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) #labels
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    #Ops and variables pinned to the CPU because of missing GPU implementation

    with tf.device('/cpu:0'):
        #look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) #Random values from a uniform distribution.
        embed = tf.nn.embedding_lookup(embeddings, train_inputs) #Retrieves rows (train_inputs) of the embeddings tensor.

        #Construct the variables for the NCE loss
        #NCE: https://datascience.stackexchange.com/questions/13216/intuitive-explanation-of-noise-contrastive-estimation-nce-loss

        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # NCE: Noise Constractive Estimation
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))

    #Construct the SGD optimizer using a learning rate of 1.0
    #SGD: Stochastic Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    #Compute the cosine similarity between minibatch examples and all embeddings

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings/norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)

    init = tf.global_variables_initializer()

 ## BEGIN TRAINING ##
num_steps = 100001

with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')
    
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it in the list of returned values for session.run())
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 = 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 # Number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dicitionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)

            final_embeddings = normalized_embeddings.eval()


## VISUALIZE THE EMBEDDINGS ##





    






