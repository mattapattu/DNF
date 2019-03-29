#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fft2,ifft2,fftshift,ifftshift
from inputDNF_1D import *
from pylab import *
import operator
import csv
from collections import defaultdict
from time import sleep
import math




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

def f(x, threshold_f):
    ''' sigmoidal function
        `x` : the matrix to be thresholded
        `threshold_f` : sigmoidal function threshold, the values below
                        it returns 0.0
    '''
    beta = 100.0    # slope of the sigmoidal function
    if np.all(x < threshold_f):          ##### !!!!!!!!!!!!!   change from "<=" to "<"
        return x*0.0
    else:
        fval = 1.0/(1 + np.exp(-beta*x)) # can be considered cases where x[i] = 0??????
    
    global peakOfActivation
    global xmax
    xmax1,peak1 = max(enumerate(fval), key=operator.itemgetter(1))
    if(peak1 >= peakOfActivation):
        xmax, peakOfActivation = xmax1,peak1
        xmax = xmax-90
        print("New peak=%f thetamax=%i" % (peakOfActivation,xmax))

    return fval   

def w(x):      # in the shape of a Mexican hat ! ;)
    ''' convolution nucleus: difference of Gaussian '''
            #print "mqslkdfjqmslfjkmslqfjkmslqfjkmslqfjslqfjsqlfmslqfjm \n"
    c_exc =  0.001         # amplitude of the excitation part
    c_inh =  0.311         # amplitude of the inhibition part
    sigma_exc = 1.0      # width of the excitation part  ! >0
    sigma_inh = 11.0      # width of the inhibition part  ! >0

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
    l = 180.0       # window length

    n = 180        # discretization of space
#    tmax = 100       # simulation time
    dt = 0.1        # time step
    shift = 0       # no shift of the input
    m = 0.0         # mean
    sigma = 1.0     # standard deviation

    #Paramètres DNF
    h = -0.8         # DNF rest threshold in the absence of an input
    tau = 1.0        # synaptic time constant
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
    thres = np.ones(len(X)) * threshold
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
    #kernel, = plot(X, W)                # kernel

    #axis settings
    plt.ylim(-6,10)           # forces the min and max of y
    plt.legend()
    
     
    
    
   
    ### Parameters for inputs
    ### Input to  DNF will be a Guassian shaped by the size & x-axis displacement of detected objects
    


# Assumption 1: All targets are of same size but at different distances #######################       
# Assumption 2: Currently activated target will always have the largest size in pixy o/p ######
# 3: Tracking of input is done manually - The object with largest size is assumed to be previously activated and will be tracked.    
    
#inputDict = defaultdict(list)

#

    # Calculation of the fft for the convolution part
    W_fft = fft(W)
    
    def updateDNF(inputs,u):
       for i in range(0,50):
           Fu_fft = fft(f(u, threshold))  # fft pour la convolution
           L = W_fft * Fu_fft             # convolution
           L = ifft(L).real
           du = dt/tau * (-u+L+H+inputs)
           inp.set_ydata(inputs) # modifies the input values !
#             input = shiftRight(input, shift)
           u += du
           sig_act.set_ydata(f(u,threshold))
           act.set_ydata(u)
           plt.draw()
       plt.show()    
#           draw()
       pause(10)
#       show()    
           
    ## Correct the course by updating input to spiking neurons        
    #def courseCorrect():
        #print("New peak is at %i with activation of %f" % (xmax,peakOfActivation))
        
    with open('inputs1') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV, None)   ## skip header
        
        currBlock=0
        actual_size = 10000
        for index,row in enumerate(readCSV):
            #print(row)
          #  inputDict[row[0]].append([row[1],row[2]])
           if(index==0):
               currBlock = row[0]
           xpos = int(row[1])-180
           size = int(row[2])
           ypos = (actual_size/size)**0.5
           d = (xpos**2+ypos**2)**0.5
#           arctan = np.arctan2(ypos,xpos)
#           deg = arctan * (180 / np.pi)
           arctan = math.atan2(ypos,xpos)/math.pi*180
           
           print("Processing row %i from  block %i" % (index,int(row[0])))
           
           #print("arctan is %f" % arctan)
           theta = 90-arctan
           print("xpos is %i, size is %i,ypos is %f,distance is %f,theta is %f" % (xpos,size,ypos,d,theta))
           
           
           if(row[0] == currBlock ) :    ### If i/p from same frame add to input
               print("Obj from same  block %i" % (int(row[0])))
               totalInput = totalInput + (100/d)* gaussian(X,mu=theta,sigma=sigma)
               #plt.plot(X, totalInput,label='input')         
               #plt.show()
               #sleep(10)
               
           else:
               print("Obj from new  block %i" % (int(row[0])))
               updateDNF(totalInput,u) ## update DNF with prev block
               #courseCorrect()
               sleep(0.02)   ##  wait 0.1s before processing new input block ( or new frame)
               ++currBlock
               totalInput = (100/d) * gaussian(X,mu=theta,sigma=sigma)
       
        
        ### Remove for actual program
        updateDNF(totalInput,u) 
               
               
               
       
           
                       
           
   
        
            
#    
#    input1 = I0*gaussian(X, mu=m-20, sigma=sigma)
#    input2 = 3*gaussian(X, mu=m+20, sigma=sigma)
#    input3 = 3*gaussian(X, mu=m+40, sigma=sigma)
#    input = input1 +input2 + input3




         


    #Calcul DNF
    #for i in range(0,int(tmax/dt)):
#    for i in range(0,100):
#        print ("t=%.3fs \t i=%d \t x_max=%.2f \t u_max=%.2f" % ((i*dt), i, X[u.argmax()], u.max()))
#        Fu_fft = fft(f(u, threshold))  # fft pour la convolution
#        L = W_fft * Fu_fft             # convolution
#        L = ifft(L).real
#        if (i<10):
#            du = dt/tau * (-u+L+H)
#        elif (i>=10 and i<150):
#            du = dt/tau * (-u+L+H+input)
#            inp.set_ydata(input) # modifies the input values !
#            input = shiftRight(input, shift)
#        else:
#            du = dt/tau * (-u+L+H+input2)
#        u += du
#        sig_act.set_ydata(f(u,threshold))
#        act.set_ydata(u)
#         
#        print("peak=%f xpeak=%i" % (peakOfActivation,xmax))
#
#       
#
#        plt.draw()
#        pause(0.1)
#        


