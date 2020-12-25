import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import nltk
import pymorphy2
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from MultyHead import *


class NewNeuralNet:

    def __init__(self):
        self.w2v_model = KeyedVectors.load("word_to_vec.wordvectors", mmap='r')
        self.vocab_size = 20000  # Only consider the top 20k words
        self.maxlen = 3
        self.embed_dim = 64  # Embedding size for each token
        self.num_heads = 4  # Number of attention heads
        self.ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        self.out_dim = 100
        inputs = layers.Input(shape=(self.maxlen,))
        embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size, self.embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(500, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(self.out_dim, activation="relu")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        print("init completed")

    def split_sentence(self, data):
        X = np.array([data[i:i+3] for i in range(2831751)])
        #X = np.array([data[i:i+3] for i in range(2831755)])
        y = np.array([data[i] for i in range(3, 2831751)])
        #y = to_categorical(y, num_classes=20000)
        x = np.zeros((2831751, 3))
        for i in range(2831751):
            x[i] = X[i]
        print("split completed")
        return [x, y]

    def create_learning_data(self):
        with open('test.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
            texts = texts.replace('\ufeff', '')
        maxWordsCount = 20000
        self.tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                                  lower=True, split=' ', char_level=False)
        self.tokenizer.fit_on_texts([texts])
        self.dataqwe = self.tokenizer.texts_to_sequences([texts])
        (X, y) = self.split_sentence(self.dataqwe[0])
        y = self.convert_y(y)
        print("create data completed")
        return [X, y]

    def convert_y(self, data):
        Y = list()
        for i in range(data.size - 1):
            Y.append(self.tokenizer.index_word[data[i]])
        yp = np.array(Y)
        arr = np.zeros(shape=(np.shape(yp)[0], 100))
        print(arr.shape)
        for i in range(np.shape(yp)[0]):
            try:
                arr[i] = np.array([self.w2v_model.wv[yp[i]]])
            except:
                arr[i] = np.array([np.zeros(shape=(100))])
        return arr

    # def main(self):
    #     ann = NewNeuralNet()
    #     (X_train, y) = ann.create_learning_data()
    #     Y = list()
    #     for i in range(y.size):
    #         Y.append(self.tokenizer.index_word[y[i]])
    #     Y = np.array(Y)
    #     y_train = ann.convert_y(Y)
    #     self.model.compile("nadam", loss="log_cosh", metrics=['accuracy'])
    #     history = self.model.fit(X_train[:2831748], y_train[:2831748], epochs=1)
