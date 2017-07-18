from keras.callbacks import ModelCheckpoint

from utils import getDataset, defineModel, getText

def train():
  raw_text = getText()
  data = getDataset(raw_text)
  model = defineModel(data)
  fitModel(model, data)

def fitModel(model, data):
  filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
  callbacks_list = [checkpoint]

  # fit the model
  model.fit(data['input'], data['output'], nb_epoch=100, batch_size=64, callbacks=callbacks_list)
    
if __name__ == "__main__":
    train()