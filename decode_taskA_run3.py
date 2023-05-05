


#pip install tokenizers

#pip install transformers

import numpy as np
import regex as re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import math
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
import tokenizers
from transformers import RobertaTokenizer, TFRobertaModel

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set (always set in Kaggle)
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
except ValueError:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)

MODEL_NAME = 'roberta-base'
MAX_LEN = 256
ARTIFACTS_PATH = '../artifacts/'

BATCH_SIZE = 8 * strategy.num_replicas_in_sync
EPOCHS = 3

if not os.path.exists(ARTIFACTS_PATH):
    os.makedirs(ARTIFACTS_PATH)

df = pd.read_csv('TaskA-TrainingSet.csv')
df.head()

X_train = df[['dialogue']].to_numpy().reshape(-1)
y_train = df[['section_header']].to_numpy().reshape(-1)

n_texts = len(X_train)
print('Texts in dataset: %d' % n_texts)

categories = df['section_header'].unique()
n_categories = len(categories)
print('Number of section header: %d' % n_categories)

print('Done!')


dfvalid = pd.read_csv('TaskA-ValidationSet.csv')
dfvalid.head()

X_valid = dfvalid[['dialogue']].to_numpy().reshape(-1)
y_valid = dfvalid[['section_header']].to_numpy().reshape(-1)



dftest = pd.read_csv('TaskA-TestSet.csv')
dftest.head()

X_test = dftest[['dialogue']].to_numpy().reshape(-1)
#y_test = df[['section_header']].to_numpy().reshape(-1)

n_texts_test = len(X_test)

def roberta_encode(texts, tokenizer):
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32') # Not used in text classification

    for k, text in enumerate(texts):
        # Tokenize
        tok_text = tokenizer.tokenize(text)
        
        # Truncate and convert tokens to numerical IDs
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN-2)])
        
        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN
        
        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k,:input_length] = np.asarray([0] + enc_text + [2], dtype='int32')
        
        # Set to 1s in the attention input
        attention_mask[k,:input_length] = 1

    return {
        'input_word_ids': input_ids,
        'input_mask': attention_mask,
        'input_type_ids': token_type_ids
    }

# Transform categories into numbers
category_to_id = {}
category_to_name = {}

for index, c in enumerate(y_train):
    if c in category_to_id:
        category_id = category_to_id[c]
    else:
        category_id = len(category_to_id)
        category_to_id[c] = category_id
        category_to_name[category_id] = c
    
    y_train[index] = category_id

#category_to_name

#X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=777)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

X_train = roberta_encode(X_train, tokenizer)
X_valid = roberta_encode(X_train, tokenizer)
X_test = roberta_encode(X_test, tokenizer)


y_train = np.asarray(y_train, dtype='int32')
y_valid = np.asarray(y_train, dtype='int32')

def build_model(n_categories):
    with strategy.scope():
        input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

        # Import RoBERTa model from HuggingFace
        roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)
        x = roberta_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

        # Huggingface transformers have multiple outputs, embeddings are the first one,
        # so let's slice out the first position
        x = x[0]

        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(n_categories, activation='softmax')(x)

        model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return model

with strategy.scope():
    model = build_model(n_categories)
    model.summary()

with strategy.scope():
    print('Training...')
    history = model.fit(X_train,
                        y_train,
                        #epochs=1,
                        #batch_size=16,                      
                        epochs=5,
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        validation_data=(X_train, y_train))

# This plot will look much better if we train models with more epochs, but anyway here is
plt.figure(figsize=(10, 10))
plt.title('Accuracy')

xaxis = np.arange(len(history.history['accuracy']))
plt.plot(xaxis, history.history['accuracy'], label='Train set')
plt.plot(xaxis, history.history['val_accuracy'], label='Validation set')
plt.legend()
y_pred = model.predict(X_test)
y_pred = [np.argmax(i) for i in model.predict(X_test)]
print(y_pred)

#val_predictions=category_to_name[y_pred]
# Importing library
import csv
 
file = open('run2.csv', 'w+', newline ='')
 
# writing the data into the file
with file:   
    write = csv.writer(file)
    i = 0
    for row in y_pred:
      write.writerow([i,row])
      i += 1

"""
def plot_confusion_matrix(X_test, model):
    y_pred = model.predict(X_test)
    y_pred = [np.argmax(i) for i in model.predict(X_test)]
    print(y_pred)

    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    label_names = list(range(len(con_mat_norm)))

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=label_names, 
                              columns=label_names)

    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(con_mat_df, cmap=plt.cm.Blues, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))


plot_confusion_matrix(X_test, y_test, model)
"""
