#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:32:17 2019

@author: ajames
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fft2,ifft2,fftshift,ifftshift
from inputDNF_1D import *
from pylab import *

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
        return 1.0/(1 + np.exp(-beta*x))    # can be considered cases where x[i] = 0??????

def w(x):      # in the shape of a Mexican hat ! ;)
    ''' convolution nucleus: sum of two Gaussian '''
            #print "mqslkdfjqmslfjkmslqfjkmslqfjkmslqfjslqfjsqlfmslqfjm \n"
    c_exc =  0.001         # amplitude of the excitation part
    c_inh =  0.1         # amplitude of the inhibition part
    sigma_exc = 1.0      # width of the excitation part  ! >0
    sigma_inh = 11.0      # width of the inhibition part  ! >0

    return c_exc * np.exp(-x**2/(2*sigma_exc**2)) - c_inh * np.exp(-x**2/(2*sigma_inh**2))



# -----------------------------------------------------------------------------
if __name__ == '__main__':

    #ion() # mode interaction on

    #Input Parameters
    l = 100.0       # window length

    n = 1000        # discretization of space
    tmax = 100       # simulation time
    dt = 0.1        # time step
    shift = 2       # no shift of the input
    m = 0.0         # hope
    sigma = 3.0     # standard deviation

    #Paramètres DNF
    h = -0.8         # DNF rest threshold in the absence of an input
    tau = 1.0        # synaptic time constant
    threshold = 0.0  # sigmoid function activation threshold


    #Input 1D
    I0 = 1.0
    x_inf, x_sup, cx, dx = -l/2, +l/2, 0, l/float(n)
    X = np.arange(x_inf, x_sup, dx)
   
    input1 = I0*gaussian(X, mu=m-20, sigma=sigma)
    input2 = 3*gaussian(X, mu=m+20, sigma=sigma)
    input3 = 3*gaussian(X, mu=m+40, sigma=sigma)
    input = input1 + input2 + input3

    #Generation of the KERNEL
    W = w(X)
    #print len(W)

    #Rest threshold, activation threshold and initialization of the population's potential
    H = np.ones(len(X)) * h
    thres = np.ones(len(X)) * threshold
    #u = np.zeros(len(X))
    u = np.ones(len(X)) * h

    #Initialization of the figure ##¡!!!!!!!!!!!!!!!!!!! HOW TO SUPERIMPOSE SEVERAL CURVES !!!!!!!!!!
    fig = plt.figure()
    inp, = plt.plot(X, input)                # input
    #rest, = plot(X, H)                  # rest threshold ('resting level')
    act, = plt.plot(X, u)                    # neural population activity
    sig_act, = plt.plot(X, f(u,threshold))   # DNF output
    thresh, = plt.plot(X, thres)             # population activity threshold
    #kernel, = plot(X, W)                # kernel


    #réglages des axes
    plt.ylim(-6,10)           # forces the min and max of y


    # Calculation of the fft for the convolution part
    W_fft = fft(W)


    #Calcul DNF
    for i in range(0,int(tmax/dt)):
        print ("t=%.3fs \t i=%d \t x_max=%.2f \t u_max=%.2f" % ((i*dt), i, X[u.argmax()], u.max()))
        Fu_fft = fft(f(u, threshold))  # fft pour la convolution
        L = W_fft * Fu_fft             # convolution
        L = ifft(L).real
        if (i<10):
            du = dt/tau * (-u+L+H)
        elif (i>=10 and i<150):
            du = dt/tau * (-u+L+H+input)
            inp.set_ydata(input) # modifies the input values !
            input = shiftRight(input, shift)
        else:
            du = dt/tau * (-u+L+H+input2)
        u += du
        sig_act.set_ydata(f(u,threshold))
        act.set_ydata(u)

        plt.draw()
        pause(0.1)


    plt.show()
