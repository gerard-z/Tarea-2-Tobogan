""" Módulo python que sirve únicamente para crear una matriz N x M x 4 que modelará la cueva, por esto su nombre
generador de mapas, donde en cada casilla (i,j) en N x M, tendrá 4 datos correspondientes a la altura del suelo,
la altura del techo, el índice de textura del suelo y el índice de textura del techo """

import numpy as np

def cuevaEjemplo(N):
    """ Una cueva sencilla, de altura tanto del suelo como del techo uniforme con una separación de 7 metros, y
    utiliza 2 índices de textura para el suelo y 2 para el techo. Utilizado para validar el programa, no es un escenario
    real. Será de tamaño N x N x 4"""
    Matriz = np.zeros((N,N,4))
    for i in range(N):
        for j in range(N):
            L = Matriz[i][j]
            L[0]=1
            L[1]=8
            if i+j<N:
                L[2]=0
                L[3]=2
            else:
                L[2]=1
                L[3]=3
    return Matriz

#np.save("map.npy",cuevaEjemplo(50))
Matriz = np.load("map.npy")
print (Matriz)