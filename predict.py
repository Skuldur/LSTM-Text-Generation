# -*- coding: utf-8 -*-
import numpy, sys

from utils import getDataset, defineModel, getText, getInputOutput

def predict():
  raw_text = getText()
  data = getDataset(raw_text)
  model = defineModel(data)

  loadWeights(model)
  generateText(model, raw_text)

def loadWeights(model):
  filename = "weights-improvement-68-1.0938-bigger.hdf5"
  model.load_weights(filename)
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def generateText(model, raw_text):
  chars = sorted(list(set(raw_text)))
  int_to_char = dict((i, c) for i, c in enumerate(chars))

  n_chars = len(raw_text)
  n_vocab = len(chars)

  # prepare the dataset of input to output pairs encoded as integers
  sequence_length = 100

  input, output = getInputOutput(raw_text, n_chars, char_to_int, sequence_length)

  start = numpy.random.randint(0, len(input)-1)
  pattern = input[start]

  for i in range(1000):
  	x = numpy.reshape(pattern, (1, len(pattern), 1))
  	x = x / float(n_vocab)
  	prediction = model.predict(x, verbose=0)
  	index = numpy.argmax(prediction)
  	result = int_to_char[index]
  	seq_in = [int_to_char[value] for value in pattern]
  	sys.stdout.write(result)
  	pattern.append(index)
  	pattern = pattern[1:len(pattern)]
  print("\nDone.")

if __name__ == "__main__":
    predict()