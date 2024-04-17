# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > 0.96 and logs.get('accuracy') > 0.96):
            print("\nReached 96% accuracy so stopping training!")
            self.model.stop_training = True

def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"

    train_size = int(training_portion * len(bbc))
    train_labels, val_labels = bbc['category'][:train_size], bbc['category'][train_size:]
    train_text, val_text = bbc['text'][:train_size], bbc['text'][train_size:]

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(vocab_size, oov_token=oov_tok,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=' ')
    tokenizer.fit_on_texts(train_text)

    # You can also use Tokenizer to encode your label.
    train_sequences = tokenizer.texts_to_sequences(train_text)
    train_padded = pad_sequences(train_sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length)

    val_sequences = tokenizer.texts_to_sequences(val_text)
    val_padded = pad_sequences(val_sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(bbc['category'])

    train_labels_seq = label_tokenizer.texts_to_sequences(train_labels)
    val_labels_seq = label_tokenizer.texts_to_sequences(val_labels)

    train_labels_seq = np.array(train_labels_seq)
    val_labels_seq = np.array(val_labels_seq)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=40, restore_best_weights=True, verbose=2)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',patience=10, factor=0.5, min_lr=5e-6)

    model.fit(
        train_padded,
        train_labels_seq,
        epochs=100,
        validation_data=(val_padded, val_labels_seq),
        callbacks=[reduce_lr,early_stop, myCallback()])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
