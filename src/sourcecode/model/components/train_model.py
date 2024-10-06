


class ModelTraining():
    def __init__():
        pass

    def data_generator(self) ->str:
        X1, X2, y = list(), list(), list()
        n = 0
        while 1:
            for key in data_keys:
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
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
                        # store the sequences
                        X1.append(features[key][0])
                        X2.append(in_seq)
                        y.append(out_seq)
                if n == batch_size:
                    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                    yield (tf.convert_to_tensor(X1, dtype=tf.float32), tf.convert_to_tensor(X2, dtype = tf.int32)), tf.convert_to_tensor(y, dtype=tf.float32)
                    X1, X2, y = list(), list(), list()
                    n = 0

    def train_model(self) -> str:
        ## load the model
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        epochs = 15
        batch_size = 32
        steps = len(ids)//batch_size

        for i in range(epochs):
            generator = data_generator1(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
            # fit for one epoch
            model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)