#!/usr/bin/env python
# -*- coding: utf-8 -*-
#=============================================
# LEAT Laboratory - Sophia Antipolis - France
# Marino Rasamuel & Benoit Miramond
#
#=============================================

# 24/08/18
# CODE DE REDUCTION DE LA TAILLE D'IMAGE PAR FILTRE
# 2 fonctions :
#       _ moyenneur
#       _ max
#
# objectif : récupérer une taille d'image plus grande avec pixy et faire
# la convolution pour la réduire à la taille de l'image d'entrée du MLP
# ==> améliorer la qualité de l'entrée du MLP


import sys
import numpy as np




def filtreMoy(array, n) :
    '''
    Filtrage d'un array par convolution de type moyenneur.
    Il faut que Array soit une matrice carrée. Et la sortie aussi.
    Donc, bien choisir n en fonction de la taille de Array.

    Paramètres :
        Array : matrice à redimensionner
        n : facteur de redimensionnement
    Sortie :
        array de taille réduit
    '''
    # taille matrice à redimensionner
    xShape, yShape = array.shape
    print("Input array shape is ",array.shape)
    # erreur si la matrice d'entrée n'est pas carrée
    if xShape != yShape :
      print 'Error arguments : Matrix d\'input must be square'
      sys.exit(1)
    # nombre d'éléments dans la matrice de départ
    nbelt_beg = xShape * yShape
    # nombre d'éléments dans la matrice d'arrivé
    nbelt_end = nbelt_beg / n
    #erreur si le facteur n aboutit à une matrice non carrée
    if (int(np.sqrt(nbelt_end)) != np.sqrt(nbelt_end)) :
      print 'Error arguments: the resizing factor must result in a square matrix'
      sys.exit(1)
    # nombre d'element par bloc
    nbelt_block = nbelt_beg / nbelt_end
    # nombre de lignes et de colonnes par block
    nbl_b, nbc_b = int(np.sqrt(nbelt_block)), int(np.sqrt(nbelt_block))
    # nombre de lignes et de colones de la matrice d'arrivée qui est carré !
    nbl, nbc = int(np.sqrt(nbelt_end)), int(np.sqrt(nbelt_end))
    #print nbl, nbc
    # matrice de sortie
    newArray = np.zeros((nbl,nbc))

    for l in range(0,nbl):
        for c in range(0,nbl):
            #print array[l*nbl_b:l*nbl_b + nbl_b, c*nbc_b:c*nbc_b + nbc_b]  #--- debug
            #print "\n"                                                     #--- debug
            # moyenne des pixels d'un bloc
            newArray[l,c] = np.sum(array[l*nbl_b:l*nbl_b + nbl_b, c*nbc_b:c*nbc_b + nbc_b]) / float(n)

    return newArray


def filtreMax(array, n) :
    '''
    Filtrage d'un array par convolution de type moyenneur.
    Il faut que Array soit une matrice carrée. Et la sortie aussi.
    Donc, bien choisir n en fonction de la taille de Array.

    Paramètres :
        Array : matrice à redimensionner
        n : facteur de redimensionnement
    Sortie :
        array de taille réduit
    '''
    # taille matrice à redimensionner
    xShape, yShape = array.shape
    # erreur si la matrice d'entrée n'est pas carrée
    if xShape != yShape :
      print 'Erreur arguments : Matrice d\'entrée doit être carrée'
      sys.exit(1)
    # nombre d'éléments dans la matrice de départ
    nbelt_beg = xShape * yShape
    # nombre d'éléments dans la matrice d'arrivé
    nbelt_end = nbelt_beg / n
    #erreur si le facteur n aboutit à une matrice non carrée
    if (int(np.sqrt(nbelt_end)) != np.sqrt(nbelt_end)) :
      print 'Erreur arguments : le facteur de redimensionnement doit résulter une matrice carrée'
      sys.exit(1)
    # nombre d'element par bloc
    nbelt_block = nbelt_beg / nbelt_end
    # nombre de lignes et de colonnes par block
    nbl_b, nbc_b = int(np.sqrt(nbelt_block)), int(np.sqrt(nbelt_block))
    # nombre de lignes et de colones de la matrice d'arrivée qui est carré !
    nbl, nbc = int(np.sqrt(nbelt_end)), int(np.sqrt(nbelt_end))
    #print nbl, nbc
    # matrice de sortie
    newArray = np.zeros((nbl,nbc))

    for l in range(0,nbl):
        for c in range(0,nbl):
            #print array[l*nbl_b:l*nbl_b + nbl_b, c*nbc_b:c*nbc_b + nbc_b]
            #print "\n"
            # moyenne des pixels d'un bloc
            newArray[l,c] = (array[l*nbl_b:l*nbl_b + nbl_b, c*nbc_b:c*nbc_b + nbc_b]).max() # / float(n)

    return newArray




if __name__ == "__main__":
    x = np.arange(36)
    x.resize(6,6)
    #res = filtreMoy(x, 4)
    res = filtreMax(x, 9)
    print(res)
    print(res.shape)
