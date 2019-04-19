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
            
        #print(weights.shape)    
            

        # At the creation of the layer:
	# open the file containing all the weights of the layer
        # THIS CODE REQUIRES THE SEPARATION OF THE WEIGHTS OF EACH LAYER IN DIFFERENT FILES
        file = open(name_file, "r")
        # separation of the lines of the file
        lines = file.readlines()
#        print("length of lines is [%i]" % len(lines))
#        print(lines)
        #print("number of rows = [%i]" % len(lines))           

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
        return 1 / (1 + np.exp(-100*x))


    # transfer function used in Lyes MLP C++ work
    def __transfertFunction(self, x):
        return np.tanh(1*x/1000);
    
     # transfer function used in Lyes MLP C++ work
    def __transfertFunction1(self, x):
        return np.tanh(100*x);

    # Propagation of the input in the network ==============> FEED FORWARD the INPUT
    def feedForward(self, inputs):

        
        #### f( Inputs)
        dnfNeuronActivation = self.__transfertFunction(inputs)
        #print(dnfNeuronActivation)
        #### Dot product of DNF_TO_MOTOR weights &  dnfNeuronActivation
        #print("self.layer1.synaptic_weights", self.layer1.synaptic_weights)            
        dotproduct_layer1 = np.dot(self.layer1.synaptic_weights,dnfNeuronActivation)
        #print("dotproduct_layer1: ",dotproduct_layer1)
        #####f(dot product)
        output_from_layer1 = self.__transfertFunction(dotproduct_layer1)
        self.layer1.layer_output = output_from_layer1
        #print("output_from_layer1: ", output_from_layer1)
        

        
        output_from_layer2 = self.__transfertFunction1(output_from_layer1)
        self.layer2.layer_output = output_from_layer2
        #print("output_from_layer2: ",output_from_layer2)

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
    
#    dnfNeuronActivations = np.random.beta(1,4,320)
#    
#    l = 320.0  
#    n = 320
#    m = 0.0         # mean
#    sigma = 1.0
#    x_inf, x_sup, cx, dx = -l/2, +l/2, 0, l/float(n)
#    X = np.arange(x_inf, x_sup, dx)
#    dnfNeuronActivations = 0.0001*gaussian(X,mu=-100,sigma=sigma)
    
    dnfNeuronActivations = np.array([3.60508519e-07, 3.89733506e-07, 4.21728916e-07, 4.56775831e-07, 4.95185118e-07, 5.37300639e-07, 5.83502822e-07, 6.34212618e-07, 6.89895873e-07, 7.51068188e-07, 8.18300289e-07, 8.92223978e-07, 9.73538720e-07, 1.06301893e-06, 1.16152205e-06, 1.26999744e-06, 1.38949627e-06, 1.52118241e-06, 1.66634443e-06, 1.82640897e-06, 2.00295534e-06, 2.19773178e-06, 2.41267326e-06, 2.64992118e-06, 2.91184508e-06, 3.20106648e-06, 3.52048518e-06, 3.87330812e-06, 4.26308120e-06, 4.69372409e-06, 5.16956859e-06, 5.69540058e-06, 6.27650611e-06, 6.91872177e-06, 7.62848993e-06, 8.41291909e-06, 9.27984980e-06, 1.02379267e-05, 1.12966770e-05, 1.24665963e-05, 1.37592413e-05, 1.51873318e-05, 1.67648602e-05, 1.85072116e-05, 2.04312933e-05, 2.25556752e-05, 2.49007416e-05, 2.74888557e-05, 3.03445369e-05, 3.34946516e-05, 3.69686193e-05, 4.07986336e-05, 4.50199000e-05, 4.96708905e-05, 5.47936165e-05, 6.04339206e-05, 6.66417875e-05, 7.34716764e-05, 8.09828735e-05, 8.92398673e-05, 9.83127459e-05, 1.08277618e-04, 1.19217056e-04, 1.31220567e-04, 1.44385080e-04, 1.58815467e-04, 1.74625083e-04, 1.91936329e-04, 2.10881236e-04, 2.31602084e-04, 2.54252024e-04, 2.78995733e-04, 3.06010083e-04, 3.35484821e-04, 3.67623274e-04, 4.02643049e-04, 4.40776749e-04, 4.82272688e-04, 5.27395603e-04, 5.76427354e-04, 6.29667616e-04, 6.87434538e-04, 7.50065384e-04, 8.17917120e-04, 8.91366965e-04, 9.70812871e-04, 1.05667394e-03, 1.14939073e-03, 1.24942553e-03, 1.35726239e-03, 1.47340717e-03, 1.59838735e-03, 1.73275163e-03, 1.87706948e-03, 2.03193033e-03, 2.19794261e-03, 2.37573250e-03, 2.56594244e-03, 2.76922932e-03, 2.98628773e-03, 3.22164286e-03, 3.69984574e-03, 1.38347549e-02, 9.85795700e-01, 9.99999999e-01, 9.99999999e-01, 9.85795839e-01, 1.38349686e-02, 3.69992524e-03, 3.22173143e-03, 2.98638773e-03, 2.76933856e-03, 2.56605887e-03, 2.37585430e-03, 2.19806814e-03, 2.03205817e-03, 1.87719839e-03, 1.73288053e-03, 1.59851531e-03, 1.47353342e-03, 1.35738703e-03, 1.24956131e-03, 1.14960936e-03, 1.05703973e-03, 9.71152152e-04, 8.91545338e-04, 8.18019496e-04, 7.50151589e-04, 6.87513867e-04, 6.29740909e-04, 5.76495011e-04, 5.27457999e-04, 4.82330180e-04, 4.40829676e-04, 4.02691733e-04, 3.67668018e-04, 3.35525911e-04, 3.06047788e-04, 2.79030308e-04, 2.54283705e-04, 2.31631094e-04, 2.10907782e-04, 1.91960603e-04, 1.74647268e-04, 1.58835729e-04, 1.44403575e-04, 1.31237439e-04, 1.19232441e-04, 1.08291638e-04, 9.83255158e-05, 8.92514929e-05, 8.09934524e-05, 7.34812985e-05, 6.66505355e-05, 6.04418706e-05, 5.48008385e-05, 4.96774485e-05, 4.50258531e-05, 4.08040356e-05, 3.69735197e-05, 3.34990956e-05, 3.03485659e-05, 2.74925074e-05, 2.49040505e-05, 2.25586728e-05, 2.04340084e-05, 1.85096703e-05, 1.67670862e-05, 1.51893469e-05, 1.37610653e-05, 1.24682470e-05, 1.12981708e-05, 1.02392783e-05, 9.28107273e-06, 8.41402553e-06, 7.62949096e-06, 6.91962742e-06, 6.27732548e-06, 5.69614192e-06, 5.17023935e-06, 4.69433104e-06, 4.26363045e-06, 3.87380522e-06, 3.52093512e-06, 3.20147381e-06, 2.91221388e-06, 2.65025516e-06, 2.41297576e-06, 2.19800584e-06, 2.00320368e-06, 1.82663406e-06, 1.66654851e-06, 1.52136748e-06, 1.38966417e-06, 1.27014979e-06, 1.16166035e-06, 1.06314452e-06, 9.73652798e-07, 8.92327643e-07, 8.18394530e-07, 7.51153895e-07, 6.89973852e-07, 6.34283595e-07, 5.83567456e-07, 5.37359521e-07, 4.95238786e-07, 4.56824770e-07, 4.21773562e-07, 3.89774257e-07, 3.60545731e-07, 3.33833743e-07, 3.09408310e-07, 2.87061349e-07, 2.66604542e-07, 2.47867419e-07, 2.30695623e-07, 2.14949349e-07, 2.00501934e-07, 1.87238587e-07, 1.75055247e-07, 1.63857544e-07, 1.53559874e-07, 1.44084556e-07, 1.35361075e-07, 1.27325398e-07, 1.19919360e-07, 1.13090107e-07, 1.06789591e-07, 1.00974125e-07, 9.56039667e-08, 9.06429543e-08, 8.60581722e-08, 8.18196512e-08, 7.79000966e-08, 7.42746436e-08, 7.09206359e-08, 6.78174259e-08, 6.49461944e-08, 6.22897873e-08, 5.98325688e-08, 5.75602878e-08, 5.54599587e-08, 5.35197522e-08, 5.17288981e-08, 5.00775971e-08, 4.85569408e-08, 4.71588409e-08, 4.58759643e-08, 4.47016757e-08, 4.36299857e-08, 4.26555046e-08, 4.17734011e-08, 4.09793659e-08, 4.02695797e-08, 3.96406844e-08, 3.90897593e-08, 3.86142998e-08, 3.82122323e-08, 3.78836702e-08, 3.77118772e-08, 3.98962228e-08, 7.92312417e-08, 1.59878930e-06, 2.20820089e-04, 4.24618919e-03, 4.24622001e-03, 2.20823994e-04, 1.59881357e-06, 7.92317220e-08, 3.98962504e-08, 3.77118315e-08, 3.78835790e-08, 3.82120904e-08, 3.86140991e-08, 3.90894918e-08, 3.96403424e-08, 4.02691554e-08, 4.09788517e-08, 4.17727913e-08, 4.26548405e-08, 4.36297929e-08, 4.47043982e-08, 4.58858395e-08, 4.71762569e-08, 4.85746453e-08, 5.00878058e-08, 5.17312670e-08, 5.35185026e-08, 5.54577892e-08, 5.75578137e-08, 5.98298350e-08, 6.22867795e-08, 6.49428936e-08, 6.78138111e-08, 7.09166840e-08, 7.42703291e-08, 7.78953914e-08, 8.18145242e-08, 8.60525893e-08, 9.06368777e-08, 9.55973552e-08, 1.00966933e-07, 1.06781768e-07, 1.13081598e-07, 1.19910105e-07, 1.27315329e-07, 1.35350118e-07, 1.44072630e-07, 1.53546890e-07, 1.63843403e-07, 1.75039840e-07, 1.87221795e-07, 2.00483623e-07, 2.14929374e-07, 2.30673823e-07, 2.47843615e-07, 2.66578538e-07, 2.87032928e-07, 3.09377233e-07, 3.33799745e-07])
    neural_network.feedForward(dnfNeuronActivations)
    neural_network.print_output()
    
    
    

