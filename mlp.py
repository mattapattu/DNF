#!/usr/bin/bin/env python
# -*- coding: utf-8 -*-


# RASAMUEL Marino
# M2 Electronics, Systems and Telecommunications (ESTel)
# LEAT laboratory trainee - CNRS
# 07/18
#
# ______________________________________________________________________________
# Perceptron MultiLayer CODE (MLP)
# ______________________________________________________________________________
#
# NETWORK TOPOLOGY: 784 (+1 neuron bias) - 300 (+1 neuron bias) - 10
#
# MLP code in the inference phase. The learning was done separately
# on N2D2, the.txt files containing the weights were then retrieved.
#
# For the code to work, the weights of the two layers must be stored
# in 2 different text files: "weights_hiddenLayer.txt" and
# "weights_outputLayer.txt".
#
# This code is not modular (the topology cannot be changed).





import numpy as np
import matplotlib.pyplot as plt
from inputDNF_1D import *


class NeuronLayer():
    def __init__(self, name_file, number_of_neurons, number_of_outputs_per_neuron):
        '''
	  Weight matrix:
          each column groups the weights of the synapses from the layer
          to 1 neuron.
   	  A weight matrix per layer (hidden and output)
	'''

	# initalization of the weight matrix
        if number_of_outputs_per_neuron==1:
            weights = np.zeros(number_of_neurons,)
        else:
            weights = np.zeros((number_of_outputs_per_neuron, number_of_neurons))
            
        print(weights.shape)    
            

        # At the creation of the layer:
	# open the file containing all the weights of the layer
        # THIS CODE REQUIRES THE SEPARATION OF THE WEIGHTS OF EACH LAYER IN DIFFERENT FILES
        file = open(name_file, "r")
        # separation of the lines of the file
        lines = file.readlines()
#        print("length of lines is [%i]" % len(lines))
#        print(lines)
        print("number of rows = [%i]" % len(lines))           

        for i in range(0,(len(lines))) :
            # separation of each value in a line => in a tank chain !
           l = lines[i].split()
#            print("length of l is [%i]" % len(l))
           for j in range(0,(len(l))) :
                # filling of the weight matrix by converting the tanks
               # print("weights [%i][%i] is [%s]" % (j,i,l[j]))
                if number_of_outputs_per_neuron == 1:
                    weights[j]  = float(l[j])
                else:
                    weights[j][i] = float(l[j])
        #print("Size of the weight matrix lxc: %dx%d" % (weights.shape[0], weights.shape[1]))
        #print(weights)
        file.close()

        # attributes of each layer: list of weights and outputs
        self.synaptic_weights = weights
        self.layer_output = np.zeros(number_of_neurons)



class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # sigmoidal function through which the accumulation of each neuron is passed
    # and which normalizes their output between 0 and 1
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    # transfer function used in Lyes MLP C++ work
    def __transfertFunction(self, x):
        return np.tanh(1*x);

    # Propagation of the input in the network ==============> FEED FORWARD the INPUT
    def feedForward(self, inputs):

        
        #### f( Inputs)
        dnfNeuronActivation = self.__transfertFunction(inputs)
        print(dnfNeuronActivation)
        #### Dot product of DNF_TO_MOTOR weights &  dnfNeuronActivation
        #print("self.layer1.synaptic_weights", self.layer1.synaptic_weights)            
        dotproduct_layer1 = np.dot(self.layer1.synaptic_weights,dnfNeuronActivation)
        print("dotproduct_layer1: ",dotproduct_layer1)
        #####f(dot product)
        output_from_layer1 = self.__transfertFunction(dotproduct_layer1)
        self.layer1.layer_output = output_from_layer1
        print("output_from_layer1: ", output_from_layer1)
        

        
        output_from_layer2 = self.__transfertFunction(output_from_layer1)
        self.layer2.layer_output = output_from_layer2
        print("output_from_layer2: ",output_from_layer2)

        return output_from_layer2

    # display only of the network output
    def print_output(self):
        print (" Output of the output layer: ")
        print ([round(x,2) for x in self.layer2.layer_output])


    # search for the max, display of the recognized number with percentage of
    #Recognition # (output layer)
    def print_winner(self):
        Number = np.array([5, 0, 4, 1, 9, 2, 3, 6, 7, 8])
        index_max = np.argmax(self.layer2.layer2.layer_output)
        print ("Recognized number | TBR")
        print (" %d | %.2f%% " % (Number[index_max],
                                        self.layer2.layer2.layer_output[index_max]*100))




if __name__ == "__main__":

    # Creation layer 1 (300 neurons, each with 785 inputs) ==> HIDDEN LAYER
    layer1 = NeuronLayer("weightsDNFtoMotor.txt",320,2) #784+1 bias

    # Creation layer 2 (10 neurons, each with 301 inputs) ==> OUTPUT LAYER
    layer2 = NeuronLayer("motorneuronoutput.txt",2, 1) #300+1 bias

    # Combination of the two layers to form the network
    neural_network = NeuralNetwork(layer1, layer2)

    #print ("Step 1) INPUT RECOVERY")
    # we get the entries in the.txt file
    
    dnfNeuronActivations = np.random.beta(1,4,320)
    
    l = 320.0  
    n = 320
    m = 0.0         # mean
    sigma = 1.0
    x_inf, x_sup, cx, dx = -l/2, +l/2, 0, l/float(n)
    X = np.arange(x_inf, x_sup, dx)
    dnfNeuronActivations = 0.0001*gaussian(X,mu=-100,sigma=sigma)

    neural_network.feedForward(dnfNeuronActivations)
    neural_network.print_output()
    
    
    

