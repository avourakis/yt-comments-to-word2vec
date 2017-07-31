import numpy as np
import math
import os
import random
import zipfile
import tensorflow as tf
import json
import collections

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
