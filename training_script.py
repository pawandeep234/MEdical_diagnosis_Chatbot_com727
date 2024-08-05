import matplotlib.pyplot as plt
import nltk

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
with open('data/data.json') as f:
    data = json.load(f)

train_x = data['embeddings']
train_y = [x for x in range(len(data['embeddings']))]

train_x = np.array(train_x)
print(train_x.shape)

model = Sequential()

model.add(Dense(128, input_shape=(train_x.shape[1], ), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(45, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


# fitting and saving the model
hist = model.fit(train_x, np.array(train_y), epochs=60, verbose=1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['accuracy'])
plt.show()

model.save('models/model.h5', hist)

print("model created")