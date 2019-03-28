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
    ''' fonction sigmoidale
        `x` : la matrice à seuiller
        `threshold_f` : seuil de la fonction sigmoidale, les valeurs en dessous
                        elle retourne 0.0
    '''
    beta = 100.0    # pente de la fonction sigmoidale
    if np.all(x < threshold_f):          ##### !!!!!!!!!!!!!   changement de "<=" en "<"
        return x*0.0
    else:
        return 1.0/(1 + np.exp(-beta*x))    # peut etre considéré les cas où x[i] = 0 ????

def w(x):      # de la forme d'un chapeau Mexicain ! ;)
    ''' noyau de convolution : somme de deux gaussiennes '''
            #print "mqslkdfjqmslfjkmslqfjkmslqfjkmslqfjslqfjsqlfmslqfjm \n"
    c_exc =  0.001         # amplitude de la partie excitatrice
    c_inh =  0.1         # amplitude de la partie inhibitrice
    sigma_exc = 1.0      # largeur de la partie excitatrice  ! >0
    sigma_inh = 11.0      # largeur de la partie inhibitrice  ! >0

    return c_exc * np.exp(-x**2/(2*sigma_exc**2)) - c_inh * np.exp(-x**2/(2*sigma_inh**2))



# -----------------------------------------------------------------------------
if __name__ == '__main__':

    #ion() # mode interaction on

    #Paramètres Entrées
    l = 100.0       # longueur de la fenêtre

    n = 1000        # discrétisation de l'espace
    tmax = 100       # temps de simulation
    dt = 0.1        # pas de temps
    shift = 1       # pas de décalage de l'entrée
    m = 0.0         # esperance
    sigma = 3.0     # ecart-type

    #Paramètres DNF
    h = -0.8         # seuil de repos du DNF en absence d'entrée
    tau = 1.0        # constante de temps synaptique
    threshold = 0.0  # seuil d'activatoin de la fonction sigmoid


    #Entrée 1D
    I0 = 1.0
    x_inf, x_sup, cx, dx = -l/2, +l/2, 0, l/float(n)
    X = np.arange(x_inf, x_sup, dx)

    input1 = I0*gaussian(X, mu=m-20, sigma=sigma)
    input2 = I0*gaussian(X, mu=m+20, sigma=sigma)
    input = input1 + input2

    #Génération du KERNEL
    W = w(X)
    #print len(W)

    #Seuil de repos, seuil d'activation et initialisation du potentiel de la population
    H = np.ones(len(X)) * h
    thres = np.ones(len(X)) * threshold
    #u = np.zeros(len(X))
    u = np.ones(len(X)) * h

    #Initialisation de la figure ##¡!!!!!!!!!!!!!!!!!!! COMMENT SUPERPOSER PLUSIEURS COURBSE !!!!!!!!!!
    fig = figure()
    inp, = plot(X, input,label='input')                # entrée
    #rest, = plot(X, H)                  # seuil de repos ('resting level')
    act, = plot(X, u,label = 'u')                    # activité de la population neuronal
    sig_act, = plot(X, f(u,threshold),label = 'f(u,threshold)')   # output du DNF
    thresh, = plot(X, thres,label = 'threshold')             # seuil de l'activité de la population
    #kernel, = plot(X, W)                # kernel
    plt.legend()

    #réglages des axes
    ylim(-6,10)           # force le min et max de y


    # Calcul de la fft pour la partie convolution
    W_fft = fft(W)


    #Calcul DNF
    for i in range(0,int(tmax/dt)):
        print( "t=%.3fs \t i=%d \t x_max=%.2f \t u_max=%.2f" % ((i*dt), i, X[u.argmax()], u.max()))
        Fu_fft = fft(f(u, threshold))  # fft pour la convolution
        L = W_fft * Fu_fft             # convolution
        L = ifft(L).real
        if (i<10):
            du = dt/tau * (-u+L+H)
        elif (i>=10 and i<150):
            du = dt/tau * (-u+L+H+input)
            inp.set_ydata(input) # modifie les valeurs de input !
            input = shiftRight(input, shift)
        else:
            du = dt/tau * (-u+L+H+input2)
        u += du
        sig_act.set_ydata(f(u,threshold))
        act.set_ydata(u)

        draw()
        pause(0.1)


    show()
