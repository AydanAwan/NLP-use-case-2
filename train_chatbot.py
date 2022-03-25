"""creating a chat bot."""
import os
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")  # setting the path so that the file can access the cuda functions needed to compute using GPU
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

# code for getting all the important information out of our intents.json and creating files to be used for training the bot
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))  # this is a 2d list that contains a tag and the wordds asociated with that tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
pickle.dump(words, open('words.pkl', 'wb'))  # holds all the words that appear in the patterns in lower case without duplicates or the ignore words and saves them in a form that can be used for training our model
pickle.dump(classes, open('classes.pkl', 'wb'))  # holds all the tags that appear in the patterns and saves them in a form that can be used for training our model


# code for turning our lists of words into usable training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0) #generates the bag of words the length of the words list with 1s or 0s to show if the word in that position is present in this document
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1  # this puts 1 in the position of the tag that this document refers to
    training.append([bag, output_row])
    random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])  # contains the word positions
train_y = list(training[:, 1])  # contains the tag positions

# create the training model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. 
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # a learning rate optimizer to control how much to change the model in response ot estimated error nesterov is just a different way of calculating the gradient
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # compiles the model using the sgd  we defined and categorical cross entropy
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)  # creates the histogram of the model
model.save('chatbot_model.h5', hist)
print("model created")
