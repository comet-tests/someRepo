'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import sys

from comet_ml import Experiment
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

def main():

    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    train(x_train,y_train,x_test,y_test)


def build_model_graph(input_shape=(784,)):
    model = Sequential()
    model.add(Dense(256, activation='sigmoid', input_shape=(784,)))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(), metrics=['accuracy'])

    return model


def train(x_train,y_train,x_test,y_test):

	
    # Define model
    model = build_model_graph()
    experiment = Experiment(api_key="XKTAnSmH2wzK24paIskC4bMV9", project_name='wat3')
		
    class Histories(keras.callbacks.Callback):
      counter = 0
      def on_train_begin(self, logs={}):
        self.losses = []
        self.counter = 0

      def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.counter = self.counter + 1				
        plt.plot(self.losses)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        with experiment.validate():
          experiment.log_figure(figure=plt, figure_name=counter)
				
#      def on_epoch_end(self, epoch, logs={}):
#        plt.plot(history.history['acc'])
#        plt.plot(history.history['val_acc'])
#        plt.plot(history.history['loss'])
#        plt.plot(history.history['val_loss'])
#        plt.title('model accuracy')
#        plt.ylabel('accuracy')
#        plt.xlabel('epoch')
#        plt.legend(['train', 'test'], loc='upper left')
#        with experiment.validate():
#          experiment.log_figure(figure=plt)
#        counter = counter + 1
#        return
    history = Histories();
    model.fit(x_train, y_train, batch_size=120, epochs=1, validation_data=(x_test, y_test), callbacks=[history])

		
#    print(history.history.keys())
#		
#		
#		# summarize history for accuracy
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
##    experiment.log_figure("test2", plt)   
#    # summarize history for loss
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
##    experiment.log_figure(figure=plt)
#    # summarize history for loss and acc
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
##    experiment.log_figure("test4")
#		
#		
    experiment.log_other("other_param","other_value")
    experiment.log_html("<a href='www.google.com'> Link </a>")
    score = model.evaluate(x_test, y_test, verbose=0)

if __name__ == '__main__':
    main()
