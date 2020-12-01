import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from random import randint
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

# Disable some troublesome logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IH = 28
IW = 28
IZ = 1
IMAGE_SIZE = 784

GENERATIONS = 12
INDIVIDUALS = 10
mutate_factor = 0.1
layers =
individuals = []

#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain, xTest = xTrain / 255, xTest / 255
    xTrain = np.ndarray.flatten(xTrain).reshape(xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2])
    xTest = np.ndarray.flatten(xTest).reshape(xTest.shape[0], xTest.shape[1] * xTest.shape[2])
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return (xTrain, yTrainP), (xTest, yTestP)


def build_conv_net(x, y, eps=10, dropout=True, dropRate=0.2):
    model = keras.Sequential()
    inShape = (IH, IW, IZ)
    lossType = keras.losses.categorical_crossentropy

    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=inShape))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())

    if dropout:
        model.add(keras.layers.Dropout(dropRate))

    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='Adam', loss=lossType)
    model.fit(x, y, epochs=eps)

    return model


def mutation(individual_new):
    genes = []
    for gene in individual_new:






#=========================<Main>=================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)

if __name__ == '__main__':
    main()