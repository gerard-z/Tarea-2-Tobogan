""" Módulo python que sirve únicamente para crear una matriz N x M x 4 que modelará la cueva, por esto su nombre
generador de mapas, donde en cada casilla (i,j) en N x M, tendrá 4 datos correspondientes a la altura del suelo,
la altura del techo, el índice de textura del suelo y el índice de textura del techo """

import numpy as np
import numpy.random as rd

def cuevaEjemplo(N):
    """ Una cueva sencilla, de altura variables y texturas randorizadas, sin embargo, no desarrollará situaciones como paredes,
    o pasadillos, ni la altura será lo suficientemente baja para que exista un conflicto con la altura del personaje.
     Será de tamaño N x N x 4"""
    Matriz = np.zeros((N,N,4), dtype=np.float32)

    Matriz[:, :, 0] = rd.rand(N, N) * 3 -1
    Matriz[:, :, 1] = rd.rand(N, N) * 3 +5
    Matriz[:, :, 2] = rd.randint(0, 4, (N, N))
    Matriz[:, :, 3] = rd.randint(5, 12, (N, N))
    return Matriz

def cuevaEjemplo2():
    """ Cueva más compleja, la altura variará con mayor notoriedad. Se escogerán índices de texturas aleatorios, entre otras cosas,
    implementará una generación de murallas predeterminadas y condiciones de altura que no permitan el paso del jugador.
    Será de tamañp 20 x 20 x 4"""
    Matriz = np.zeros((20, 20, 4), dtype=np.float32)

    Matriz[:, :, 0] = np.ones((20, 20))
    Matriz[:, :, 1] = np.ones((20, 20)) * 4
    Matriz[5:15, 5:15, 0] += np.ones((10, 10))
    Matriz[5:15, 9, 1] -= np.ones((10))  # Brecha de menor altura que el personaje en medio.
    Matriz[2, :, 1] -= np.ones(20) * 5
    Matriz[17, :, 0] += np.ones(20) * 5
    Matriz[: , :, 2:] = rd.randint(0, 12, (20, 20, 2))
    return Matriz


# Funciones que guardan y cargan un archivo npy, en este caso, donde se guarda la matriz que funcionará de mapa

np.save("map.npy",cuevaEjemplo(10))
np.save("prueba.npy", cuevaEjemplo2())
#Matriz = np.load("map.npy")
