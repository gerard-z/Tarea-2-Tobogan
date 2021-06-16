"""Funciones para crear distintas figuras y escenas en 3D """

import openmesh as om
import numpy as np
import numpy.random as rd
from OpenGL.GL import *
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.transformations as tr
import grafica.scene_graph as sg
import sys, os.path
from resources import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
thisFilePath = os.path.abspath(__file__)
thisFolderPath = os.path.dirname(thisFilePath)
assetsDirectory = os.path.join(thisFolderPath, "sprites")
waterPath = os.path.join(assetsDirectory, "water.png")
displacementPath = os.path.join(assetsDirectory, "displacement.png")
texturasPath = os.path.join(assetsDirectory, "textures.png")

# Convenience function to ease initialization
def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    return gpuShape

def createTextureGPUShape(shape, pipeline, path, sWrapMode=GL_CLAMP_TO_EDGE, tWrapMode=GL_CLAMP_TO_EDGE, minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST, mode=GL_STATIC_DRAW):
    # Funcion Conveniente para facilitar la inicializacion de un GPUShape con texturas
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, mode)
    gpuShape.texture = es.textureSimpleSetup(
        path, sWrapMode, tWrapMode, minFilterMode, maxFilterMode)
    return gpuShape

def createMultipleTextureGPUShape(shape, pipeline, paths, sWrapMode=GL_CLAMP_TO_EDGE, tWrapMode=GL_CLAMP_TO_EDGE, minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST, mode=GL_STATIC_DRAW):
    # Funcion Conveniente para facilitar la inicializacion de un GPUShape con texturas
    Cantidad = len(paths)
    gpuShape = es.GPUShapeMulti(Cantidad).initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, mode)
    for i in range(Cantidad):
        gpuShape.texture.append( es.textureSimpleSetup(
            paths[i], sWrapMode, tWrapMode, minFilterMode, maxFilterMode))
    return gpuShape

def generateT(t):
    "Entrega un vector tiempo"   
    return np.array([[1, t, t**2, t**3]]).T

def Curve(typeCurve, V1, V2, V3, V4, N):
    """ str np.array np.array np.array np.array int -> np.ndarray((N,3))
    Función que crea una curva con los 4 vectores claves para la parametrización y entrega la curva en cuestión.
    Las curvas compatibles son:
    "Hermite", curva que recibe P1, P2, T1, T2, punto inicial y final y sus tengentes.
    "Bezier", curva que recibe P1, P2, P3, P4, punto inicial, intermedio 1 y y2 y la final.
    "CatmullRom", curva que recibe P0, P1, P2, P3, punto anterior, inicial, final, y después"""
    # Se crean N puntos a evaluar entre 0 y 1
    ts = np.linspace(0.0, 1.0, N)

    # Genera la matriz que contiene los puntos y tangentes claves de la curva
    G = np.concatenate((V1, V2, V3, V4), axis=1)
    
    # Se crera la matriz de la curva constante que da la forma:
    if typeCurve == "Hermite":
        M = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])
    elif typeCurve == "Bezier":
        M = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    elif typeCurve == "CatmullRom":
        M = np.array([[0, -0.5, 1, -0.5], [1, 0, -2.5, 1.5], [0, 0.5, 2, -1.5], [0, 0, -0.5, 0.5]])
    else:
        print ("No se reconoce la curva para los vectores:", V1, V2, V3, V4)
        assert False

    #Se crea la matriz de la curva
    M = np.matmul(G, M)
    
    # Se crea la curva:
    curve = np.ndarray((N,3))

    # Se evalua cada punto de la curva
    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T

    return curve

def Curveposition(typeCurve, V1, V2, V3, V4, t):
    """ Función similar a la anterior, salvo que en vez de generar todas las posiciones de la curva, genera la posición del instante de tiempo pedido
    Las curvas compatibles son:
    "Hermite", curva que recibe P1, P2, T1, T2, punto inicial y final y sus tengentes.
    "Bezier", curva que recibe P1, P2, P3, P4, punto inicial, intermedio 1 y y2 y la final.
    "CatmullRom", curva que recibe P0, P1, P2, P3, punto anterior, inicial, final, y después"""
    # Genera la matriz que contiene los puntos y tangentes claves de la curva
    G = np.concatenate((V1, V2, V3, V4), axis=1)
    
    # Se crera la matriz de la curva constante que da la forma:
    if typeCurve == "Hermite":
        M = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])
    elif typeCurve == "Bezier":
        M = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    elif typeCurve == "CatmullRom":
        M = np.array([[0, -0.5, 1, -0.5], [1, 0, -2.5, 1.5], [0, 0.5, 2, -1.5], [0, 0, -0.5, 0.5]])
    else:
        print ("No se reconoce la curva para los vectores:", V1, V2, V3, V4)
        assert False
    
    # se evalua en el tiempo correspondiente
    T = generateT(t)
    curve = np.matmul(G, M)
    return np.matmul(curve, T).T



######################################################


######## CREANDO UNA MALLA FRACTAL #####
def fractalMesh(mesh, n):
    k = 0
    while k<n:
        newMesh = om.TriMesh()
        for face in mesh.faces():
            vertexs = list(mesh.fv(face))
            vertex1 = np.array(mesh.point(vertexs[0]))
            vertex2 = np.array(mesh.point(vertexs[1]))
            vertex3 = np.array(mesh.point(vertexs[2]))

            vertex12 = (vertex1 + vertex2)/2
            vertex23 = (vertex2 + vertex3)/2
            vertex13 = (vertex1 + vertex3)/2

            vertex12[2] = (rd.rand()-0.5)*0.1 + vertex12[2]
            vertex23[2] = (rd.rand()-0.5)*0.1 + vertex23[2]
            vertex13[2] = (rd.rand()-0.5)*0.1 + vertex13[2]

            v1 = newMesh.add_vertex(vertex1)
            v12 = newMesh.add_vertex(vertex12)
            v2 = newMesh.add_vertex(vertex2)
            v23 = newMesh.add_vertex(vertex23)
            v3 = newMesh.add_vertex(vertex3)
            v13 = newMesh.add_vertex(vertex13)

            newMesh.add_face(v1, v12, v13)
            newMesh.add_face(v12, v13, v23)
            newMesh.add_face(v13, v23, v3)
            newMesh.add_face(v12, v2, v23)
            #newMesh.add_face(v1, v2, v3)

        mesh = newMesh
        k+=1
        del newMesh

    return mesh


def caveMesh(matriz):
    """ Se crea las 2 mallas de polígonos correspondiente al suelo y el techo, por conveniencia, se utilizarán celdas
    de 5x5 metros cuadrados. (Considerando que los ejes se encontraran efectivamente en metros)
    De esta manera, Lara será capaz de moverse por la celda. """
    suelo = om.TriMesh()
    techo = om.TriMesh()
    # Se obtienen las dimensiones de la matriz
    (N, M, k) = matriz.shape
    n= N//2
    m= M//2
    # Se crean arreglos que corresponderan al eje x e y de la cueva, de N+1 y M+1 vértices cada uno, de modo que
    # cada celda de la matriz sea generada por un cuadrado de 4 vértices
    if N%2!=0:
        xs = np.linspace(-3*N-n, 3*N+n, N*7)
    else:
        xs = np.linspace(-3*N-n, 3*N+n-1, N*7)
    if M%2!=0:
        ys = np.linspace(-3*M-m, 3*M+m, M*7)
    else:
        ys = np.linspace(-3*M-m, 3*M+m-1, M*7)

    

    # largo de arregles
    lenXS = len(xs)-1
    lenYS = len(ys)-1

    # Se generan los vértices de la malla, utilizando las alturas dadas
    for i in range(lenXS):
        x = xs[i]
        im = i//7   # Transforma el índice en su correspondiente celda de la matriz
        a = False
        for j in range(lenYS):
            y = ys[j]
            jm = j//7 # Transforma el índice en su correspondiente celda de la matriz
            b = False
            z0 = matriz[im][jm][0]
            z1 = matriz[im][jm][1]
            if (i+1)//7 != im:
                Im = im+1
                z0 = (z0 + matriz[Im][jm][0])/2
                z1 = (z1 + matriz[Im][jm][1])/2
                a = True
            if (j+1)//7 != jm:
                Jm = jm+1
                z0 = (z0 + matriz[im][Jm][0])/2
                z1 = (z1 + matriz[im][Jm][1])/2
                b = True
            if a and b:
                z0 = (matriz[im][jm][0] + matriz[im][Jm][0] + matriz[Im][Jm][0] + matriz[Im][jm][0])/4
                z1 = (matriz[im][jm][1] + matriz[im][Jm][1] + matriz[Im][Jm][1] + matriz[Im][jm][1])/4
            
            # Condición borde, para cerrar las paredes y que no se pueda salir al vacío
            if i==0 or j==0 or i==lenXS-1 or j==lenYS-1:
                z1 = z0

            # Agregamos el vértice a la malla correspondiente
            suelo.add_vertex([x, y, z0])
            techo.add_vertex([x, y, z1])

    # Se calcula el índice de cada punto (i, j) de la forma:
    index = lambda i, j: i*lenYS + j
    # Obtenemos los vertices de cada malla, y agregamos las caras
    vertexsuelo = list(suelo.vertices())
    vertextecho = list(techo.vertices())

    # Creamos las caras para esta malla (Y usar esta orientación para los factoriales)
    for i in range(lenXS-1):
        for j in range(lenYS-1):
            # los índices:
            isw = index(i,j)
            ise = index(i+1,j)
            ine = index(i+1,j+1)
            inw = index(i,j+1)
            # Identificar vértices
            Vsw = vertexsuelo[isw]
            Vse = vertexsuelo[ise]
            Vne = vertexsuelo[ine]
            Vnw = vertexsuelo[inw]
            # Se agregan las caras
            suelo.add_face(Vsw, Vse, Vne)
            suelo.add_face(Vne, Vnw, Vsw)
            # Identificar vértices
            Vsw = vertextecho[isw]
            Vse = vertextecho[ise]
            Vne = vertextecho[ine]
            Vnw = vertextecho[inw]
            # Se agregan las caras
            techo.add_face(Vsw, Vse, Vne)
            techo.add_face(Vne, Vnw, Vsw)


    # Se aplican fractales a la malla
    fractal = 0
    sueloMesh = fractalMesh(suelo, fractal)
    techoMesh = fractalMesh(techo, fractal)
    lenXS += (lenXS-1)*(2**fractal -1)
    lenYS += (lenYS-1)*(2**fractal -1)

    index = lambda i, j: i*lenYS + j

    # Obtenemos los vertices de cada malla, y agregamos las caras
    vertexsuelo = list(sueloMesh.vertices())
    vertextecho = list(techoMesh.vertices())

    # Se generan los nuevos mesh que contienen las texturas (Se rehace ya que cada hay vértices que contienen
    # 4 coordenadas de texturas)
    sueloMeshtex = om.TriMesh()
    techoMeshtex = om.TriMesh()
    sueloMeshtex.request_vertex_texcoords2D()
    techoMeshtex.request_vertex_texcoords2D()

    indexMat = 7 + 6*(2**fractal -1)

    # Se crean las caras para cada cuadrado de la celda
    for i in range(lenXS-1):
        im = (i+1)//indexMat
        for j in range(lenYS-1):
            jm = (j+1)//indexMat
            # los índices:
            isw = index(i,j)
            ise = index(i+1,j)
            ine = index(i+1,j+1)
            inw = index(i,j+1)
            # Coordenadas de texturas
            indice = matriz[im][jm][2]
            tx = 1/12 * (indice)
            tX = 1/12 * (indice+1)
            tsw = [tx, 1]
            tse = [tX, 1]
            tne = [tX, 0]
            tnw = [tx, 0]
            # Identificar vértices
            vsw = vertexsuelo[isw]
            vse = vertexsuelo[ise]
            vne = vertexsuelo[ine]
            vnw = vertexsuelo[inw]
            # Agregar vertices a la nueva malla
            Vsw = sueloMeshtex.add_vertex(sueloMesh.point(vsw).tolist())
            Vse = sueloMeshtex.add_vertex(sueloMesh.point(vse).tolist())
            Vne = sueloMeshtex.add_vertex(sueloMesh.point(vne).tolist())
            Vnw = sueloMeshtex.add_vertex(sueloMesh.point(vnw).tolist())
            # Agregar las coordenadas de texturas a los vertices
            sueloMeshtex.set_texcoord2D(Vsw, tsw)
            sueloMeshtex.set_texcoord2D(Vse, tse)
            sueloMeshtex.set_texcoord2D(Vne, tne)
            sueloMeshtex.set_texcoord2D(Vnw, tnw)
            # Se agregan las caras
            sueloMeshtex.add_face(Vsw, Vse, Vne)
            sueloMeshtex.add_face(Vne, Vnw, Vsw)

            # Identificar vértices
            vsw = vertextecho[isw]
            vse = vertextecho[ise]
            vne = vertextecho[ine]
            vnw = vertextecho[inw]
            # Agregar vertices a la nueva malla
            Vsw = techoMeshtex.add_vertex(techoMesh.point(vsw).tolist())
            Vse = techoMeshtex.add_vertex(techoMesh.point(vse).tolist())
            Vne = techoMeshtex.add_vertex(techoMesh.point(vne).tolist())
            Vnw = techoMeshtex.add_vertex(techoMesh.point(vnw).tolist())
            # Coordenadas de texturas
            indice = matriz[im][jm][3]
            tx = 1/12 * (indice)
            tX = 1/12 * (indice+1)
            tsw = [tx, 1]
            tse = [tX, 1]
            tne = [tX, 0]
            tnw = [tx, 0]
            # Agregar las coordenadas de texturas a los vertices
            techoMeshtex.set_texcoord2D(Vsw, tsw)
            techoMeshtex.set_texcoord2D(Vse, tse)
            techoMeshtex.set_texcoord2D(Vne, tne)
            techoMeshtex.set_texcoord2D(Vnw, tnw)
            # Se agregan las caras
            techoMeshtex.add_face(Vsw, Vse, Vne)
            techoMeshtex.add_face(Vne, Vnw, Vsw)

    del sueloMesh
    del techoMesh
    # Se entregan las mallas
    return (sueloMeshtex, techoMeshtex, lenXS, lenYS)

def get_vertexs_and_indexes(mesh, orientation):
    # Obtenemos las caras de la malla
    faces = mesh.faces()

    # orientation indica si las normales deben apuntar abajo(-1) o arriba(1)
    assert orientation==1 or orientation==-1, "La orientación debe ser indicada con 1 o -1"

    # Creamos una lista para los vertices e indices
    vertexs = []
    indexes = []

    # Obtenemos los vertices y los recorremos
    for vertex in mesh.vertices():
        point = mesh.point(vertex).tolist()
        vertexs += point
        # Agregamos las coordenadas de a textura y su índice
        vertexs += mesh.texcoord2D(vertex).tolist()
        # Agregamos la norma
        normal = calculateNormal(mesh, vertex)
        normal = orientation * normal

        vertexs += [normal[0], normal[1], normal[2]]

    for face in faces:
        # Obtenemos los vertices de la cara
        face_indexes = mesh.fv(face)
        for vertex in face_indexes:
            # Obtenemos el numero de indice y lo agregamos a la lista
            indexes += [vertex.idx()]

    return vertexs, indexes

def createCave(pipeline, Matriz):
    # Creamos las mallas
    meshs = caveMesh(Matriz)
    # obtenemos los vértices e índices del suelo y del techo
    sVertices, sIndices = get_vertexs_and_indexes(meshs[0],1)
    tVertices, tIndices = get_vertexs_and_indexes(meshs[1],-1)
    sueloShape = bs.Shape(sVertices, sIndices)
    techoShape = bs.Shape(tVertices, tIndices)

    suelo = mallaTam(meshs[0], meshs[2], meshs[3])
    techo = meshs[1]

    gpuSuelo = createTextureGPUShape(sueloShape, pipeline, texturasPath)
    gpuTecho = createTextureGPUShape(techoShape, pipeline, texturasPath)

    return gpuSuelo, gpuTecho, suelo, techo

########## Curva Nonuniform splines ##############################
class CurvaNoUniforme:
    """ Crear una curva no uniforme de clase C2, para empezar, se debe dividir el tiempo de cada intervalo entre puntos en proporción al largo del segmento.
    De esta manera, el recorrido va tener una velocidad media. Para esto se utilizan intervalos de curvas de Hermite"""
    def __init__(self, posiciones, velocidad):
        """ Entregar el vector de posiciones donde deberá recorrer la curva y la velocidad promedio de recorrido """
        self.posiciones = posiciones
        self.velocidades = []
        self.Nodos = len(posiciones) # Número de nodos que tiene la curva
        self.distancias = []
        self.distanciaMax = 0
        for i in range(self.Nodos-1):
            pos0 = self.posiciones[i]
            pos1 = self.posiciones[i+1]
            distancia = np.linalg.norm(pos1-pos0)
            self.distancias.append(distancia)
            self.distanciaMax += distancia
        self.tiempo = self.distanciaMax/velocidad # Tiempo de la curva
        self.posActual = posiciones[0] # Donde comienza

    def getPosition(self, tiempo):
        """ Calcula la posición de la curva grande, estimando entre que nodos  está y calcular la curva de HERMITE que describe entre los nodos que se encuentra la posición en el tiempo"""
        assert tiempo<self.tiempo, "El tiempo a evaluar debe estar dentro del rango parametrizado del tiempo"
        distancia = tiempo * self.distanciaMax  # Tiempo entre 0 y 1
        distanciaActual = 0
        i=0
        while distanciaActual + self.distancias[i] < distancia and i<self.Nodos-1:
            distanciaActual += self.distancias[i]
            i += 1
        t = distancia - distanciaActual
        t /= self.distancias[i]     # Tiempo entre 0 y 1
        return Curveposition("Hermite", self.posiciones[i], self.posiciones[i+1], self.velocidad, self.velocidad, t)

    
# TODO: Agregar la igualdad en C2, es decir la aceleración final del camino anterior debe ser igual a la aceleración inicial del siguiente