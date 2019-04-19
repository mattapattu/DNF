#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fft2,ifft2,fftshift,ifftshift
from inputDNF_1D import *
from pylab import *
import operator
from collections import defaultdict
from time import sleep
import pixy.build.libpixyusb_swig.pixymodule as pmod
import pixy.build.libpixyusb_swig.pixy as pixy1
from mlp import *




'''

  ∂u(x,t)                 ⌠
τ ------- = -u(x,t) + h + ⎮  f(u(x',t)) ω(x-x') dx' + S(x,t)
    ∂t                    ⌡Ω

where # u(x,t) is the potential of a neural population at position x and time t
      # Ω is the domain of integration
      # ω(x) is the interaction kernel that determines connectivity between positions x and x' on the DNF
      # S(x) is the external input (e.g from sensor) ==> mostly saliency map
      # τ is the temporal decay of the synapse
      # f(x) is a sigmoidal non-linearity that shapes the output of the DNF
      # h is a negative resting level that sets values of u(x,t) to be below zero (output threshold) in absence of external input



'''
global peakOfActivation
global xmax
global totalInput
peakOfActivation=0.0
xmax = 0


def motorResponse(inputDNFActivation):
     # Creation layer 1 (300 neurons, each with 785 inputs) ==> HIDDEN LAYER
    print(inputDNFActivation)
    layer1 = NeuronLayer("weightsDNFtoMotor.txt",320,2) #784+1 bias

    # Creation layer 2 (10 neurons, each with 301 inputs) ==> OUTPUT LAYER
    layer2 = NeuronLayer("motorneuronoutput.txt",2, 2) #300+1 bias

    # Combination of the two layers to form the network
    neural_network = NeuralNetwork(layer1, layer2)

    #print ("Step 1) INPUT RECOVERY")
    # we get the entries in the.txt file
    
    dnfNeuronActivations = np.zeros(320)
    neural_network.feedForward(dnfNeuronActivations)
    neural_network.print_output()

    

def f(x, threshold_f):
    ''' sigmoidal function
        `x` : the matrix to be thresholded
        `threshold_f` : sigmoidal function threshold, the values below
                        it returns 0.0
    '''
    beta = 1.0    # slope of the sigmoidal function
    if np.all(x < threshold_f):          ##### !!!!!!!!!!!!!   change from "<=" to "<"
        return x*0.0
    else:
        fval = 1.0/(1 + np.exp(-beta*x)) # can be considered cases where x[i] = 0??????
    
    global peakOfActivation
    global xmax
    
    xmax1,peak1 = max(enumerate(fval), key=operator.itemgetter(1))
    if(peak1 >= peakOfActivation):
        xmax, peakOfActivation = xmax1,peak1
        xmax = xmax-159.5
        #print("New peak=%f xmax=%i" % (peakOfActivation,xmax))

    return fval   

def w(x):      # in the shape of a Mexican hat ! ;)
    ''' convolution nucleus: difference of Gaussian '''
            #print "mqslkdfjqmslfjkmslqfjkmslqfjkmslqfjslqfjsqlfmslqfjm \n"
#    c_exc =  5.0         # amplitude of the excitation part
#    c_inh =  4.65         # amplitude of the inhibition part
#    sigma_exc = 1.0      # width of the excitation part  ! >0
#    sigma_inh = 30.0      # width of the inhibition part  ! >0
            
    c_exc =  5.0         # amplitude of the excitation part
    c_inh =  4.0         # amplitude of the inhibition part
    sigma_exc = 1.0      # width of the excitation part  ! >0
    sigma_inh = 100.0      # width of the inhibition part  ! >0

    return c_exc * np.exp(-x**2/(2*sigma_exc**2)) - c_inh * np.exp(-x**2/(2*sigma_inh**2))


#def e(u,threshold):
#    exp=0
#    for i in range(1,320):
#       print("f = %f,x=%i" % (f(u[i-1],threshold),i))
#       exp = exp+i*f(u[i-1],threshold)
#    return exp       
#
#def ftotal(u,threshold):
#    ftot = 0
#    for i in range(1,320):
#       ftot = ftot+f(u[i-1],threshold)
#    return ftot 

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    #ion() # mode interaction on

    #Input Parameters
    l = 320.0       # window length

    n = 320        # discretization of space
#    tmax = 100       # simulation time
    dt = 0.1        # time step
    m = 0.0         # mean
    sigma = 1.0     # standard deviation

    #Paramètres DNF
    h = -0.8         # DNF rest threshold in the absence of an input
    tau = 2.0        # synaptic time constant
    threshold = 0.0  # sigmoid function activation threshold


    #Input 1D
    I0 = 1.0
    x_inf, x_sup, cx, dx = -l/2, +l/2, 0, l/float(n)
    X = np.arange(x_inf, x_sup, dx)

    
    #Generation of the KERNEL
    W = w(X)
    #print len(W)

    #Rest threshold, activation threshold and initialization of the population's potential
    H = np.ones(len(X)) * h
    thres = np.ones(len(X),dtype=np.float128) * threshold
    #u = np.zeros(len(X))
    global u
    u = np.ones(len(X)) * h
    
    global totalInput
    totalInput = 0.0001*gaussian(X,mu=0,sigma=sigma)
    #Initialization of the figure ##¡!!!!!!!!!!!!!!!!!!! HOW TO SUPERIMPOSE SEVERAL CURVES !!!!!!!!!!
    fig = plt.figure()
    #inp1, =  plt.plot(X, input1)
    inp, = plt.plot(X, totalInput,label='input')                # input
    #rest, = plot(X, H)                  # rest threshold ('resting level')
    act, = plt.plot(X, u,label = 'u')                    # neural population activity
    sig_act, = plt.plot(X, f(u,threshold),label = 'f(u,threshold)')   # DNF output
    thresh, = plt.plot(X, thres,label = 'threshold')             # population activity threshold
    kernel, = plt.plot(X, W, label = 'kernel')                # kernel

    #axis settings
    plt.ylim(-6,10)           # forces the min and max of y
    plt.legend()

# Assumption 1: All targets are of same size but at different distances #######################       
# Assumption 2: Currently activated target will always have the largest size in pixy o/p until target is brought ######
    

    # Calculation of the fft for the convolution part
    W_fft = fft(W)
    
    def updateDNF(inputs,u):
       print("Updating DNF\n") 
       for i in range(0,150):
           Fu_fft = fft(f(u, threshold))  # fft pour la convolution
           L = W_fft * Fu_fft             # convolution
           L = ifft(L).real
           du = dt/tau * (-u+L+H+inputs)
           inp.set_ydata(inputs) # modifies the input values !
#             input = shiftRight(input, shift)
           u += du
           updatedActivation = f(u,threshold)
           motorResponse(updatedActivation)
           sig_act.set_ydata(updatedActivation)
           act.set_ydata(u)
           plt.draw()
       plt.show()    
#           draw()
       pause(2)
#       show()    
           
    ## Correct the course by updating input to spiking neurons        
    #def courseCorrect():
        #print("New peak is at %i with activation of %f" % (xmax,peakOfActivation))
#    input1 = I0*gaussian(X, mu=m-20, sigma=sigma)
#    input2 = I0*gaussian(X, mu=m+20, sigma=sigma)
#    input = input1 + input2    
    blockcount=0    
    while(1):
       blocks = list() 
       blocks = pmod.getPixyBlocks()
       #print(blocks)
       totalInput = 0
       blockcount = blockcount+1
       if blocks is not None:
           count = 0
           for block in blocks:
               count = count+1
               print('[BLOCK_TYPE=%d SIG=%d X=%3d Y=%3d WIDTH=%3d HEIGHT=%3d]' % (block[0], block[1], block[2], block[3], block[4], block[5]))
               xpos = block[2]-159.5
#               print(xpos)
#               process = Popen(['pixy/src/host/snapshot',"block-"+str(blockcount)+"-frame-"+str(count),"2", str(block[4]),str(block[5]),str(block[2]),str(block[3]) ], stdout=PIPE, stderr=PIPE)
#               stdout, stderr = process.communicate()
#               print("Output:"+stdout)
               #print("Stopping pixy")
               #pmod.pixyStop()
               #time.sleep(1.0)
               
               #print("Starting pixy")
               #pmod.pixyStart()
               #time.sleep(0.5)
               totalInput = totalInput + (block[4]*block[5])/300* gaussian(X,mu=xpos,sigma=sigma)
           updateDNF(totalInput,u)    
        
