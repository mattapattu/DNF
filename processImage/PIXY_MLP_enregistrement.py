#!/usr/bin/env python
# -*- coding: utf-8 -*-
#=============================================
# LEAT Laboratory - Sophia Antipolis - France
# Marino Rasamuel & Benoit Miramond
#
#=============================================


# 23/08/18
# ------------------------------------------------------
# Version 4.0 : animation pixy_get_frame() + MLP
# ------------------------------------------------------
# Reconnaissance image pixy par MLP
# TRAITEMENT PRÉALABLE de la sortie de la caméra
# ===> enlever un maximum de bruits
#
#
#
# Sélectionner une image plus grande que 28x28 puis SOUS ECHANTILLONNER
# l'image:
# Convolution  : _ moyenneur
#                _ max
#
#
# Affichage de la simulation  :
#        _ image 200x320
#	     _ image selectionnée de dimension 28x28, 56x56, 112x112, 168x 168
#        _ image entrée du MLP 28x28 après traitement
#
#
# Version avec possibilité de controlé le seuillage
#


import sys
from pixy import *
import numpy as np
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons

from scipy import signal, misc  #convolve, imresize
import imageCompression as ic
from ctypes import *



#******************************************************************************#
#                     Initialisation de la thread Pixy                         #
#******************************************************************************#
pixy_init_status = pixy_init()

if pixy_init_status != 0:
    print ('Error: pixy_init() [%d] ' % pixy_init_status)
    pixy_error(pixy_init_status)
    sys.exit(1)

class Blocks (Structure):
  _fields_ = [ ("type", c_uint),
               ("signature", c_uint),
               ("x", c_uint),
               ("y", c_uint),
               ("width", c_uint),
               ("height", c_uint),
               ("angle", c_uint) ]

blocks = BlockArray(100)
frame = 0

def getPixyBlocks():
    count = pixy_get_blocks(100, blocks)

    if count > 0:
        # Blocks found #
        global frame	
        print('frame %3d:' % (frame))
        frame = frame + 1
        ret = list()	
        for index in range (0, count):
          print('[BLOCK_TYPE=%d SIG=%d X=%3d Y=%3d WIDTH=%3d HEIGHT=%3d]' % (blocks[index].type, blocks[index].signature, blocks[index].x, blocks[index].y, blocks[index].width, blocks[index].height))
          ret.append([blocks[index].type,blocks[index].signature, blocks[index].x, blocks[index].y, blocks[index].width, blocks[index].height])
        return ret  
    #else:
        #print("No block found")
    

size_selection = 168  # 28 / 56 / 112 / 168
seuillage  = 0.5
def acquisition(x,y):
    global winner, TBR_win
    #******************** INPUT ET FILTRAGE INPUT ************************#
    n, frame = pixy_get_frame(64000)  # 64000 nombre de pixels à récupérer
    frame.resize((200,320))
    #frame = misc.imresize(frame, (200/2,320/2)) # divise par deux la taille de l'image
    frame = misc.imresize(frame, (200,320))   # \\bon1\\
    frame = frame.astype(float)
    frame = frame/frame.max()

    # filtrage par convolution gaussien du l'image entière
    #frame = signal.fftconvolve(frame,gauss_frame,mode='same') # filtrage de l'image (gaussien)

    # on récupère une image d'une taille supérieure ou égale à 28x28
    demi_sel = size_selection/2 # :)
        
    if y-demi_sel<0 or y+demi_sel > 200 or  x-demi_sel <0 or x+demi_sel > 320 :
        raise ValueError('Unable to capture frame', -1)
        # on récupère la milieu de la frame
    inputbig = frame[y-demi_sel:y+demi_sel, x-demi_sel:x+demi_sel]
    # inversement des min et max, car chiffre en noir (donc noir doit être valeur max)
    inputbig = 1 - inputbig
    # Reduction de la taille de l'image par filtrage par MAX ou MOY pour obtenir une image 28x28
    # facteur de redimensionnement
    fact_redim = (size_selection * size_selection ) / 784 # 1 / 4 / 16 / 25
    input = ic.filtreMoy(inputbig, fact_redim)
    #input = ic.filtreMax(inputbig, fact_redim)


    #filtrage par convolution gaussien de l'image 28x28
    #input = signal.fftconvolve(input,gauss_input,mode='same') # \\28 x 28\\ convolution de l'image

    #filtre de seuillage
    threshold_image = seuillage  #0.5  # \\bon1 à 0.8\\
    #input =  np.where(input>threshold_image, 1.0, 0.0) # \\bon1 à 1.0 et 0.0\\
    input =  np.where(input>threshold_image, input, 0.0)

    #********************************  MLP  *******************************#
    inputMLP = input.flatten()      # remettre l'input en forme pour l'entrée MLP
    inputMLP = np.append(input,1)   # rajouter l'entrée bias supplémentaire
#        output = neural_network.feedForward(inputMLP)
#        output = [round(x,4) for x in output]
#
#        if output[np.argmax(output)]*100.0 > seuil_TBR :
#            neural_network.print_output()
#            neural_network.print_winner()
#            winner = win[np.argmax(output)]
#            TBR_win = output[np.argmax(output)]*100.0
#
#        tx = fig.suptitle("WINNER %d   TBR %2.2f%%  |   number %d   tbr %2.2f%% " %
#                         (winner, TBR_win, win[np.argmax(output)],
#                         output[np.argmax(output)]*100.0))
#


    return inputMLP
   

#------------------------------ VIDÉO -----------------------------------#

#Paramètres du format du fichier .mp4
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)


#******************************************************************************#
#                         initialisation de la figure                          #
#******************************************************************************#
        
if __name__ == "__main__":
    fig = plt.figure()
    # titre du graphique
    #tx = fig.suptitle('t = 0s', fontsize=14, fontweight='bold')
    
    # 3 subplots, matrices à tracer : input, activation, sigmoid
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4)
    # titres de chaque subplot
    ax1.set_title('Frame')
    ax2.set_title('Image %dx%d' % (size_selection, size_selection) )
    ax3.set_title('Input MLP 28x28')
    
    # initialisation de chaque subplot
    frame_im = ax1.imshow(np.zeros((200,320)),
    		              cmap='gray',
                          interpolation='nearest',
                          animated=True)
    
    inpbig_im = ax2.imshow(np.zeros((56,56)),
                            cmap='gray',
                            interpolation='nearest',
                            animated=True)
    
    
    inpMLP_im = ax3.imshow(np.zeros((28,28)),
                            cmap='gray',
                            interpolation='nearest',
                            animated=True)
    
    # colorbar légende
    cb1 = fig.colorbar(frame_im, ax=ax1) #ticks pour forcer la colorbar
    cb2 = fig.colorbar(inpbig_im, ax=ax2)
    cb3 = fig.colorbar(inpMLP_im, ax=ax3)
    
    
    #******************************************************************************#
    #                                   WIDGETS                                    #
    #******************************************************************************#
    
    # création des SLIDERS correpondants aux différents paramètres
    # plt.axes([x,y,longueur,largeur])
    x_wdgt = 0.56 # position du widget sur la longueur de la figure
    l_wdgt = 0.3  # longueur du widget
    w_wdgt = 0.03 # largeur du widget
    
    ax_seuil      = plt.axes([x_wdgt, 0.8, l_wdgt, w_wdgt])    # seuillage input MLP
    
    # paramétrage des "slider"
    s_seuil      = Slider(ax_seuil, 'Seuil', 0.0, 1.0, valinit=seuillage0)
    
    
    # fonction appelée lorsqu'un paramètre est modifié
    def update(val):
        global seuillage
    
        seuillage = s_seuil.val
    
    # mis à jour en cas de modification des paramètres
    s_seuil.on_changed(update)
    
    # bouton de sélection de la taille à récupérer en entrée avnt traitement
    axcolor = 'lightgoldenrodyellow'
    ax_sizeSelect = plt.axes([0.57, 0.54, 0.2, 0.2])
    buttonSizeSelect = RadioButtons(ax_sizeSelect, ('28x28', '56x56', '112x112', '168x168'), active=1)
    
    def sizeSelection(label):
        global size_selection, demi_sel
        selectDict = {'28x28': 28, '56x56' : 56, '112x112' : 112, '168x168' : 168 }
        size_selection = selectDict[label]
        ax2.set_title('Image %dx%d' % (size_selection, size_selection) )
        demi_sel = size_selection/2 # :)
    buttonSizeSelect.on_clicked(sizeSelection)
    
    #******************************************************************************#
    #                      FONCTION UTILISÉE POUR L'ANIMATION                      #
    #******************************************************************************#
    ani = animation.FuncAnimation(fig,
                              init_func=init_plots,
                              func=update_frames,
                              frames=acquisition,
                              repeat=False,
                              interval=50,
                              blit=False) #False
    
    #ani.save('MLP_classisfication.mp4', writer=writer)
    
    plt.show()
    
    pixy_close()
