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

# tf.enable_eager_execution()

# Disable some troublesome logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

class Genetic():

    genCount = 0
    fittest = None
    secondFittest = None

    def __init__(self, inputSize, outputSize, neuronsPerLayer, data):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.population = Population()
        self.population.initializePopulation(data)
        self.mutate_factor = .1

        # Initial Population
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    #Selection
    def __selection(self):
        self.fittest = self.population.getFittest()
        self.secondFittest = self.population.getSecondFittest()

    #Crossover
    def __crossover(self):
        value = randint(0, self.population.individuals[0].geneLength)
        for i in range(0, value):
            temp = self.fittest.gene[i]
            self.fittest.gene[i] = self.secondFittest.gene[i]
            self.secondFittest.gene[i] = temp

    #Mutation
    def __mutation(self):
        mutationPoint = randint(0, self.population.individuals[0].geneLength)

        self.fittest.genes[mutationPoint] = random.random()

        mutationPoint = randint(0, self.population.individuals[0].geneLength)

        self.secondFittest.genes[mutationPoint] = random.random()

    #Get fittest offspring
    def __getFittestOffspring(self):
        if self.fittest.fitness > self.secondFittest.fitness:
            return self.fittest
        return self.secondFittest

    #Replace least fittest individual from most fittest offspring
    def __addFittestOffspring(self):

        #Update fitness values of offspring
        self.fittest.calcFitness()
        self.secondFittest.calcFitness()

        #Get index of least fit individual
        leastFittestIndex = self.population.getLeastFittestIndex()

        #Replace least fittest individual from most fittest offspring
        self.population.individuals[leastFittestIndex] = self.__getFittestOffspring()

    def train(self):

        loop = 0

        while loop < 1:

            self.__selection()
            self.__crossover()

            n = random.random()

            if n < self.mutate_factor:
                self.__mutation()

            self.__addFittestOffspring()
            self.population.calculateFitness()
            loop += 1

    def optimize(self):
        return self.fittest




#Individual class
class Individual():

    fitness = 0
    geneLength = 0
    preds = None

    def __init__(self, inputSize, outputSize, neuronsPerLayer, data):

        self.fitness = 0
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.data = data

        # Initial Population
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)
        # self.gene = self.W1.flatten() + self.W2.flatten()
        self.gene = np.concatenate((self.W1.flatten(), self.W2.flatten()))
        self.geneLength = len(self.gene)

    # Forward pass.
    def __forward(self, input):
        layer1 = keras.activations.sigmoid(np.dot(input, self.W1))
        layer2 = keras.activations.sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        self.preds = layer2.numpy()
        return layer2.numpy()

    def calcFitness(self):
        preds = self.__modPreds(self.predict(self.data[0]))
        fitness = 0
        xTest, yTest = self.data
        table = dict.fromkeys(range(10), 0)
        predsmod = np.zeros(preds.shape[0])
        yTestmod = np.zeros(yTest.shape[0])
        for i in range(preds.shape[0]):
            predsmod[i] = preds[i].argmax(axis=0)
            yTestmod[i] = yTest[i].argmax(axis=0)
            # Add to dictionary
            table[preds[i].argmax(axis=0)] += 1
            if np.array_equal(preds[i], yTest[i]):   fitness = fitness + 1
        self.fitness = fitness

    def __modPreds(self, preds):
        modout = np.zeros(preds.shape)
        for i in range(0, preds.shape[0]):
            modout[i][preds[i].argmax(axis=0)] = 1
        return modout

#Population class
class Population():

    popSize = 10
    individuals = [None] * 10
    fittest = 0

    def initializePopulation(self, data):
        self.data = data
        for i in range(0, self.popSize):
            # TODO only has static size for mnist
            self.individuals[i] = Individual(784, 10, 512, data)
            self.individuals[i].data = data[0]
        self.calculateFitness()

    #Get the fittest individual
    def getFittest(self):
        maxFit = -1
        maxFitIndex = 0
        for i in range(0, self.popSize):
            if maxFit <= self.individuals[i].fitness:
                maxFit = self.individuals[i].fitness
                maxFitIndex = i
        self.fittest = self.individuals[maxFitIndex].fitness
        return self.individuals[maxFitIndex]

    #Get the second most fittest individual
    def getSecondFittest(self):
        maxFit1 = 0
        maxFit2 = 0
        for i in range(0, self.popSize):
            if self.individuals[i].fitness > self.individuals[maxFit1].fitness:
                maxFit2 = maxFit1
                maxFit1 = i
            elif self.individuals[i].fitness > self.individuals[maxFit2].fitness:
                maxFit2 = i
        return self.individuals[maxFit2]

    #Get index of least fittest individual
    def getLeastFittestIndex(self):
        minFitVal = -1
        minFitIndex = 0
        for i in range(0, self.popSize):
            if minFitVal >= self.individuals[i].fitness:
                minFitVal = self.individuals[i].fitness
                minFitIndex = i

        return minFitIndex

    #Calculate fitness of each individual
    def calculateFitness(self):

        for i in range(0, self.popSize):
            self.individuals[i].calcFitness()

        self.getFittest()


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
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain, xTest = xTrain / 255, xTest / 255
    xTrain = np.ndarray.flatten(xTrain).reshape(xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2])
    xTest = np.ndarray.flatten(xTest).reshape(xTest.shape[0], xTest.shape[1] * xTest.shape[2])
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))

#=========================<Main>=================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    g = Genetic(784, 10, 512, data)
    g.train()
    fitest = g.optimize()
    print(fitest.fitness / 60000.0)

if __name__ == '__main__':
    main()
