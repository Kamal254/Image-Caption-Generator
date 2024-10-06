from sourcecode import logger
from sourcecode.model.components.build_model import Models
from sourcecode.model.utils.utils import create_dir
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf

class ModelTraining():
    def __init__(self):
        self.max_length = 10
        self.vocab_size = 119
        self.batch_size = 2
        self.epochs = 1


    def data_generator(self):
        X1, X2, y = list(), list(), list()
        n = 0
        with open('../../../artifacts/saved_data/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        with open('../../../artifacts/saved_data/features.pkl', 'rb') as f:
            features = pickle.load(f)

        with open('../../../artifacts/saved_data/mapping.pkl', 'rb') as f:
            mapping = pickle.load(f)

        while 1:
            for key in mapping.keys():
                n += 1
                captions = mapping[key]
                # process each caption
                for caption in captions:
                    # encode the sequence
                    seq = tokenizer.texts_to_sequences([caption])[0]
                    # split the sequence into X, y pairs
                    for i in range(1, len(seq)):
                        # split into input and output pairs
                        in_seq, out_seq = seq[:i], seq[i]
                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq],num_classes=self.vocab_size)[0]
                        # store the sequences
                        X1.append(features[key][0])
                        X2.append(in_seq)
                        y.append(out_seq)
                if n == self.batch_size:
                    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                    yield (tf.convert_to_tensor(X1, dtype=tf.float32), tf.convert_to_tensor(X2, dtype = tf.int32)), tf.convert_to_tensor(y, dtype=tf.float32)
                    X1, X2, y = list(), list(), list()
                    n = 0

    def train_model(self) -> str:
        model = Models()
        model = model.build_training_model(self.max_length, self.vocab_size)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        with open('../../../artifacts/saved_data/mapping.pkl', 'rb') as f:
            mapping = pickle.load(f)
        steps = len(mapping.keys())//self.batch_size
        print(mapping.keys())
        print(steps)
        for i in range(self.epochs):
            generator = self.data_generator()
            print(i)
            # fit for one epoch
            model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

        model_save_path = "../../../artifacts/model"
        create_dir([model_save_path])
        model_path = model_save_path+"/model.h5"
        model.save(model_path)


train_model = ModelTraining()
train_model.train_model()