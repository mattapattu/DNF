#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

'''
Generation of a Gaussian 1D input that runs from left to right,
and from right to left the window. This entry will then be given to a 1D DNF.

Further: generation of a second input after a time t, in order to
check the behaviour of the TDR

'''



def gaussian(x, mu=0.0, sigma=1.0):
    ''' Gaussian function of the form exp(-x²/σ²)/(π.σ²) '''
    ''' Gaussian function of the form exp(-(x-μ)²/2σ²) '''
    #return 1.0/(sigma**2*np.pi)*np.exp(-x**2/(sigma**2))
    #return 1.0/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))
    return np.exp(-(x-mu)**2/(2*sigma**2))



def gaussian2D(x,y,mu_x=0.0, mu_y=0.0, sigma=1.0):   # no need for mu
    ''' Gaussian function of the form exp(-x²/σ²)/(π.σ²) '''
    return np.exp(-[(x-mu_x)^2+(y-mu_y)^2]/(sigma**2))





# I can possibly not use these functions and directly use the roll (Optim PLUS TARD)
def shiftLeft(X,shift):
    ''' Function that allows the curve to be shifted to the left '''
    X = np.roll(X,-shift)
    return X

def shiftRight(X,shift):
    ''' Function that allows the curve to be shifted to the right  '''
    X = np.roll(X,shift)
    return X

def generateInput(X, t_simul, shift, ):

    return 0


#def Input(length=100.0, mu=0.0, sigma):

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    plt.ion() # mode interaction on

    #Paramètres
    l = 100.0       # window length
    n = 50         # discretization of space
    tmax = 20       # input simulation time
    dt = 0.1        # time step
    shift = 2       # no shift of the input
    mu = 0.0        # hope
    sigma = 1.0     # standard deviation

    #Input
    x_inf, x_sup, cx, dx = -l/2, +l/2, 0, l/float(n)
    X = np.arange(x_inf, x_sup, dx)
    input = gaussian(X)  # Gaussian type input generation
    print ('x_inf=%.1f  x_sup=%.1f  len(X)=%.1f' % (x_inf, x_sup, len(X)))

    #Initialization of the figure
    print ("t=%.3fs \t i=%d \t x_max=%.2f" % (0, 0, X[input.argmax()]))
    line, = plt.plot(X, input)

    #moving curve
    for i in range(1,int(tmax/dt)):             # input makes round trips on the length window l
        if (i%n <= int((n/2)/shift)) :
            input = shiftRight(input,shift)
        elif (i%n > int((n/2)/shift) and i%n <= int((n/2)/shift)*3) :
            input = shiftLeft(input,shift)
        elif (i%n > int(((n/2)/shift))*3 and i%n < n) :
            input = shiftRight(input,shift)
        print ("t=%.2fs \t i=%d \t x_max=%.2f" % ((i*dt), i%n, X[input.argmax()]))
        line.set_ydata(input) # modifies the input values !
        plt.draw() # forces the drawing of the figure
        #pause(0.001)
        plt.pause(0) ## <===== problème

    plt.ioff()
    plt.show()
