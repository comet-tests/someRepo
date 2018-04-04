'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import sys

from comet_ml import Experiment


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
    experiment = Experiment(api_key="C6bHgvq0atB3HzOQVGVQv40Oh", project_name='test3')
    with open("file",'r') as f:  
		    html_str = f.read().replace('\n', '')
#    experiment.log_html(html_str)
#    experiment.log_other("other_key","other_value")
#    experiment.log_other("other_key1","other_value")
#    experiment.log_other("2other_key","oFFther_value")
#    experiment.log_other("o3ther_key","othFer_value")
#    experiment.log_other("ot4her_key","otheFr_value")
#    experiment.log_other("oth5er_key","otherFF_value")
#    experiment.log_other("othe6r_key","other_vFFFalue")
#    experiment.log_other("other7_key","other_valFue")
#    experiment.log_other("other_7key","other_valuFe")
#    experiment.log_other("other_k8ey","Fother_value")
#    experiment.log_other("other_ke9y","oFther_value")
#    experiment.log_other("other_key0","otFher_value")
#    experiment.log_other("1other_key","othFer_value")
#    experiment.log_other("o2ther_key","otheFr_value")
#    experiment.log_other("ot2her_key","otherF_value")
#    experiment.log_other("oth3er_key","other_FFFvalue")
#    experiment.log_other("othe4r_key","other_valFFue")
		
    model.fit(x_train, y_train, batch_size=120, epochs=12, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

if __name__ == '__main__':
    main()
