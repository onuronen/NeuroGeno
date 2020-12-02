import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from random import randint
import random

GENERATIONS = 10
INDIVIDUALS = 10
mutate_factor = 0.05
NUM_CLASSES = 10
individuals_list = []
LAYERS = [0, 3, 5]
IH = 28
IW = 28
IZ = 1
inShape = (IH, IW , IZ)

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

def preprocess_data():
    ((xTrain, yTrain), (xTest, yTest)) = getRawData()
    xTrain = xTrain / 256
    xTest = xTest / 256

    xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
    xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return (xTrainP, yTrainP), (xTestP, yTestP)


def mutation_weights(new_individual):

    for i in LAYERS:
        for bias in range(len(new_individual.layers[i].get_weights()[1])):
            n = random.random()
            if (n < mutate_factor):
                new_individual.layers[i].get_weights()[1][bias] *= random.uniform(-0.5, 0.5)


    for i in LAYERS:
        for weight in new_individual.layers[i].get_weights()[0]:
            n = random.random()
            if (n < mutate_factor):
                for j in range(len(weight)):
                    if (random.random() < mutate_factor):
                        new_individual.layers[i].get_weights()[0][j] = new_individual.layers[i].get_weights()[0][j] * \
                                                                       random.uniform(-0.5, 0.5)

    return new_individual


def build_conv_net(dropout = True):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape = inShape))

    lossType = keras.losses.categorical_crossentropy

    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(64, activation='relu'))

    if dropout:
        model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam', loss=lossType, metrics=['accuracy'])

    model.save("my_model.h5", model)

    return model


def train(models):
    #train model with 1 epoch and return trained model with its loss
    (xTrain, yTrain), (xTest, yTest) = preprocess_data()
    losses = []
    for i in range(len(models)):
        history = models[i].fit(x=xTrain,y=yTrain, epochs=1, validation_data=(xTest, yTest))
        losses.append(round(history.history['loss'][-1], 4))
    return models, losses


#mating and forming new individuals with better weights
def crossover(individuals):
    new_individuals = []

    #append the 2 elites
    new_individuals.append(individuals[0])
    new_individuals.append(individuals[1])


    for i in range(2, INDIVIDUALS):
        #randomly choose last 2 from the list
        if (i >= (INDIVIDUALS - 2)):
            new_individual = random.choice(individuals[:])

        else:
            if (i == 2):
                first_parent = random.choice(individuals[:3])
                second_parent = random.choice(individuals[:3])

            else:
                first_parent = random.choice(individuals[:])
                second_parent = random.choice(individuals[:])

            for i in LAYERS:
                temp = first_parent.layers[i].get_weights()[1]
                first_parent.layers[i].get_weights()[1] = second_parent.layers[i].get_weights()[1]
                second_parent.layers[i].get_weights()[1] = temp

                new_individual = random.choice([first_parent, second_parent])


        new_individuals.append(mutation_weights(new_individual))

    return new_individuals


def evolve(individuals, losses):
    #sort current population, find elites and parents for the new generation
    sorted_list = sorted(range(len(losses)), key=lambda x: losses[x])
    individuals = [individuals[i] for i in sorted_list]

    new_individuals = crossover(individuals)

    return new_individuals



def train_individuals(copy_individuals_list):
    for i in range(INDIVIDUALS):
        copy_individuals_list.append(build_conv_net())
    return copy_individuals_list


#Iterate through number of generations and perform evolution
def train_gens(copy_individuals_list):
    for generation in range(GENERATIONS):
        copy_individuals_list, losses = train(copy_individuals_list)
        print(losses)

        copy_individuals_list = evolve(copy_individuals_list, losses)

    return copy_individuals_list



#from matplotlib import pyplot as plt

if __name__ == '__main__':
    individuals_list = train_individuals(individuals_list)
    individuals_list = train_gens(individuals_list)
    (xTrain, yTrain), (xTest, yTest) = preprocess_data()
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    #print(yTrain[0])
    final_model = load_model("my_model.h5")
    final_result = final_model.predict(xTrain[0].reshape(1,28,28,1))
    # probability results and the index
    print("Resulting output from model: ", final_result)
    print("estimated value ", np.argmax(final_result[0]))





