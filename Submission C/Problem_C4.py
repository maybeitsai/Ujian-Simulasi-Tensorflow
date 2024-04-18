# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('sarcasm.json', 'r') as file:
        data = json.load(file)

    for item in data:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    # Split
    train_text, test_text = sentences[:training_size], sentences[training_size:]
    train_label, test_label = np.array(labels[:training_size]), np.array(labels[training_size:])

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
    tokenizer.fit_on_texts(train_text)

    train_sequences = tokenizer.texts_to_sequences(train_text)
    train_padded = pad_sequences(train_sequences, max_length, truncating=trunc_type, padding=padding_type)

    test_sequences = tokenizer.texts_to_sequences(test_text)
    test_padded = pad_sequences(test_sequences, max_length, truncating=trunc_type, padding=padding_type)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=4,
        factor=0.3,
        min_lr=1e-6
    )

    model.fit(
        train_padded,
        train_label,
        validation_data=(test_padded, test_label),
        epochs=30,
        callbacks=[reduce_lr]
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
