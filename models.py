#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Logistic Regression classifier

   Brown CS142, Spring 2020
'''
import random
import math
import numpy as np


def softmax(x):
    '''
    Apply softmax to an array

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)

class LogisticRegression:
    '''
    Multiclass Logistic Regression that learns weights using 
    stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_classes, n_features + 1))  # An extra row added for the bias
        self.alpha = 0.03  # DO NOT TUNE THIS PARAMETER
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        '''
        Trains the model using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''

        w = self.weights 
        a = self.alpha
        b = self.batch_size
        epoch = 1
        converge = False 
        while not converge:
            ind = np.arange(Y.size)
            np.shuffle(ind)
            for i in rangenp.ceil((X.shape[0]/b)-1): 
                x_b = X[i*b: (i+1)*b,:]
                y_b = Y[i*b: (i+1)*b]
                L_w = np.zeros((self.n_classes, self.n_features + 1))
                
                for x,y in x_b, y_b: 
                    for j in [self.n_classes+1]:
                        if y == j:
                            L_w += (softmax(w @ x)*j) - 1 * x
                        else:
                            L_w += softmax(w @ x)*j * x
                w = w - (a*L_w)/len(x_b)
            if epoch ==1:
                if (abs(loss(x,y)- math.inf)< self.conv_threshold): converge = True
            else:
                if (abs(loss(x,y)- loss(x,y)) < self.conv_threshold): converge = True #epoch specific?



        pass

    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of examples.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the average loss of the model on the dataset
        '''
        # TODO
        pass

    def predict(self, X):
        '''
        Compute predictions based on the learned weigths and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        y = np.zeros(X.shape[0])
        y = X.transpose @ self.weights
        pass

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        # TODO
        pass
