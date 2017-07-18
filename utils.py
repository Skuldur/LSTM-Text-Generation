import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.utils import np_utils

def getText(filename):
  raw_text = open(filename, encoding="utf8").read()
  return raw_text

#create mapping of unique chars to integers
def getDataset(raw_text):
  chars = sorted(list(set(raw_text)))
  char_to_int = dict((char, number) for number, char in enumerate(chars))

  n_chars = len(raw_text)
  n_vocab = len(chars)

  print("Total Characters: ", n_chars)
  print("Total Vocab: ", n_vocab)

  # prepare the dataset of input to output pairs encoded as integers
  sequence_length = 100

  dataInput, dataOutput = getInputOutput(raw_text, n_chars, char_to_int, sequence_length)

  n_patterns = len(dataInput)
  print("Total Patterns: ", n_patterns)

  #reshape X to be [samples, time steps, features]
  input = numpy.reshape(dataInput, (n_patterns, sequence_length, 1))
  #normalize
  input = input / float(n_vocab)
  #one hot encode the output variable
  output = np_utils.to_categorical(dataOutput)

  return { 'vocab': n_vocab, 'input': input, 'output': output };

def getInputOutput(text, n_chars, char_to_int, sequence_length):
  input = []
  output = []
  for i in range(0, n_chars - sequence_length, 1):
    sequence_in = text[i:i + sequence_length]
    sequence_out = text[i + sequence_length]
    input.append([char_to_int[char] for char in sequence_in])
    output.append(char_to_int[sequence_out])

  return input, output


def defineModel(data):
  input = data['input']
  n_vocab = data['vocab']

  model = Sequential()
  model.add(LSTM(256, input_shape=(input.shape[1], input.shape[2]), return_sequences=True))
  model.add(Dropout(0.3))
  model.add(LSTM(256, return_sequences=True))
  model.add(Dropout(0.3))
  model.add(LSTM(256))
  model.add(Dense(n_vocab))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

  return model
  