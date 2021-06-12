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

def Curve(typeCurve, V1, V2, V3, V4, N):
    """ str np.array np.array np.array np.array int -> np.ndarray((N,3))
    Función que crea una curva con los 4 vectores claves para la parametrización y entrega la curva en cuestión.
    Las curvas compatibles son:
    "Hermite", curva que recibe P1, P2, T1, T2, punto inicial y final y sus tengentes.
    "Bezier", curva que recibe P1, P2, P3, P4, punto inicial, intermedio 1 y y2 y la final.
    "CatmullRom", curva que recibe P0, P1, P2, P3, punto anterior, inicial, final, y después"""


    # Se crean N puntos a evaluar entre 0 y 1
    ts = np.linspace(0.0, 1.0, N)

    # Se crea el vector tiempo
    def generateT(t):   
        return np.array([[1, t, t**2, t**3]]).T

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
    
    # Se crea la curva:
    curve = np.ndarray((N,3))

    # Se evalua cada punto de la curva
    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T

    return curve



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

def createScene(pipeline):

    gpuRedCube = createGPUShape(pipeline, bs.createColorNormalsCube(1, 0, 0))
    gpuGreenCube = createGPUShape(pipeline, bs.createColorNormalsCube(0, 1, 0))
    gpuGrayCube = createGPUShape(pipeline, bs.createColorNormalsCube(0.7, 0.7, 0.7))
    gpuWhiteCube = createGPUShape(pipeline, bs.createColorNormalsCube(1, 1, 1))

    redCubeNode = sg.SceneGraphNode("redCube")
    redCubeNode.childs = [gpuRedCube]

    greenCubeNode = sg.SceneGraphNode("greenCube")
    greenCubeNode.childs = [gpuGreenCube]

    grayCubeNode = sg.SceneGraphNode("grayCube")
    grayCubeNode.childs = [gpuGrayCube]

    whiteCubeNode = sg.SceneGraphNode("whiteCube")
    whiteCubeNode.childs = [gpuWhiteCube]

    rightWallNode = sg.SceneGraphNode("rightWall")
    rightWallNode.transform = tr.translate(1, 0, 0)
    rightWallNode.childs = [redCubeNode]

    leftWallNode = sg.SceneGraphNode("leftWall")
    leftWallNode.transform = tr.translate(-1, 0, 0)
    leftWallNode.childs = [greenCubeNode]

    backWallNode = sg.SceneGraphNode("backWall")
    backWallNode.transform = tr.translate(0,-1, 0)
    backWallNode.childs = [grayCubeNode]

    lightNode = sg.SceneGraphNode("lightSource")
    lightNode.transform = tr.matmul([tr.translate(0, 0, -0.4), tr.scale(0.12, 0.12, 0.12)])
    lightNode.childs = [grayCubeNode]

    ceilNode = sg.SceneGraphNode("ceil")
    ceilNode.transform = tr.translate(0, 0, 1)
    ceilNode.childs = [grayCubeNode, lightNode]

    floorNode = sg.SceneGraphNode("floor")
    floorNode.transform = tr.translate(0, 0, -1)
    floorNode.childs = [grayCubeNode]

    sceneNode = sg.SceneGraphNode("scene")
    sceneNode.transform = tr.matmul([tr.translate(0, 0, 0), tr.scale(5, 5, 5)])
    sceneNode.childs = [rightWallNode, leftWallNode, backWallNode, ceilNode, floorNode]

    trSceneNode = sg.SceneGraphNode("tr_scene")
    trSceneNode.childs = [sceneNode]

    return trSceneNode

def createCube1(pipeline):
    gpuGrayCube = createGPUShape(pipeline, bs.createColorNormalsCube(0.5, 0.5, 0.5))

    grayCubeNode = sg.SceneGraphNode("grayCube")
    grayCubeNode.childs = [gpuGrayCube]

    objectNode = sg.SceneGraphNode("object1")
    objectNode.transform = tr.matmul([
        tr.translate(0.25,-0.15,-0.25),
        tr.rotationZ(np.pi*0.15),
        tr.scale(0.2,0.2,0.5)
    ])
    objectNode.childs = [grayCubeNode]

    scaledObject = sg.SceneGraphNode("object1")
    scaledObject.transform = tr.scale(5, 5, 5)
    scaledObject.childs = [objectNode]

    return scaledObject

def createCube2(pipeline):
    gpuGrayCube = createGPUShape(pipeline, bs.createColorNormalsCube(0.5, 0.5, 0.5))

    grayCubeNode = sg.SceneGraphNode("grayCube")
    grayCubeNode.childs = [gpuGrayCube]

    objectNode = sg.SceneGraphNode("object1")
    objectNode.transform = tr.matmul([
        tr.translate(-0.25,-0.15,-0.35),
        tr.rotationZ(np.pi*-0.2),
        tr.scale(0.3,0.3,0.3)
    ])
    objectNode.childs = [grayCubeNode]

    scaledObject = sg.SceneGraphNode("object1")
    scaledObject.transform = tr.scale(5, 5, 5)
    scaledObject.childs = [objectNode]

    return scaledObject

def createColorNormalSphere(N, r, g, b):

    vertices = []
    indices = []
    dTheta = 2 * np.pi /N
    dPhi = 2 * np.pi /N
    r = 0.5
    c = 0

    for i in range(N - 1):
        theta = i * dTheta
        theta1 = (i + 1) * dTheta
        for j in range(N):
            phi = j*dPhi
            phi1 = (j+1)*dPhi
            v0 = [r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)]
            v1 = [r*np.sin(theta1)*np.cos(phi), r*np.sin(theta1)*np.sin(phi), r*np.cos(theta1)]
            v2 = [r*np.sin(theta1)*np.cos(phi1), r*np.sin(theta1)*np.sin(phi1), r*np.cos(theta1)]
            v3 = [r*np.sin(theta)*np.cos(phi1), r*np.sin(theta)*np.sin(phi1), r*np.cos(theta)]
            n0 = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
            n1 = [np.sin(theta1)*np.cos(phi), np.sin(theta1)*np.sin(phi), np.cos(theta1)]
            n2 = [np.sin(theta1)*np.cos(phi1), np.sin(theta1)*np.sin(phi1), np.cos(theta1)]
            n3 = [np.sin(theta)*np.cos(phi1), np.sin(theta)*np.sin(phi1), np.cos(theta)]


            # Creamos los quad superiores
            if i == 0:
                vertices += [v0[0], v0[1], v0[2], r, g, b, n0[0], n0[1], n0[2]]
                vertices += [v1[0], v1[1], v1[2], r, g, b, n1[0], n1[1], n1[2]]
                vertices += [v2[0], v2[1], v2[2], r, g, b, n2[0], n2[1], n2[2]]
                indices += [ c + 0, c + 1, c +2 ]
                c += 3
            # Creamos los quads inferiores
            elif i == (N-2):
                vertices += [v0[0], v0[1], v0[2], r, g, b, n0[0], n0[1], n0[2]]
                vertices += [v1[0], v1[1], v1[2], r, g, b, n1[0], n1[1], n1[2]]
                vertices += [v3[0], v3[1], v3[2], r, g, b, n3[0], n3[1], n3[2]]
                indices += [ c + 0, c + 1, c +2 ]
                c += 3
            
            # Creamos los quads intermedios
            else: 
                vertices += [v0[0], v0[1], v0[2], r, g, b, n0[0], n0[1], n0[2]]
                vertices += [v1[0], v1[1], v1[2], r, g, b, n1[0], n1[1], n1[2]]
                vertices += [v2[0], v2[1], v2[2], r, g, b, n2[0], n2[1], n2[2]]
                vertices += [v3[0], v3[1], v3[2], r, g, b, n3[0], n3[1], n3[2]]
                indices += [ c + 0, c + 1, c +2 ]
                indices += [ c + 2, c + 3, c + 0 ]
                c += 4
    return bs.Shape(vertices, indices)

def createTextureNormalSphere(N):
    vertices = []
    indices = []
    dTheta = 2 * np.pi /N
    dPhi = 2 * np.pi /N
    r = 0.5
    c = 0

    for i in range(N - 1):
        theta = i * dTheta
        theta1 = (i + 1) * dTheta
        for j in range(N):
            phi = j*dPhi
            phi1 = (j+1)*dPhi
            v0 = [r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)]
            v1 = [r*np.sin(theta1)*np.cos(phi), r*np.sin(theta1)*np.sin(phi), r*np.cos(theta1)]
            v2 = [r*np.sin(theta1)*np.cos(phi1), r*np.sin(theta1)*np.sin(phi1), r*np.cos(theta1)]
            v3 = [r*np.sin(theta)*np.cos(phi1), r*np.sin(theta)*np.sin(phi1), r*np.cos(theta)]
            n0 = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
            n1 = [np.sin(theta1)*np.cos(phi), np.sin(theta1)*np.sin(phi), np.cos(theta1)]
            n2 = [np.sin(theta1)*np.cos(phi1), np.sin(theta1)*np.sin(phi1), np.cos(theta1)]
            n3 = [np.sin(theta)*np.cos(phi1), np.sin(theta)*np.sin(phi1), np.cos(theta)]


            # Creamos los quad superiores
            if i == 0:
                vertices += [v0[0], v0[1], v0[2], 0, 1, n0[0], n0[1], n0[2]]
                vertices += [v1[0], v1[1], v1[2], 1, 1, n1[0], n1[1], n1[2]]
                vertices += [v2[0], v2[1], v2[2], 0.5, 0, n2[0], n2[1], n2[2]]
                indices += [ c + 0, c + 1, c +2 ]
                c += 3
            # Creamos los quads inferiores
            elif i == (N-2):
                vertices += [v0[0], v0[1], v0[2], 0, 0, n0[0], n0[1], n0[2]]
                vertices += [v1[0], v1[1], v1[2], 0.5, 1, n1[0], n1[1], n1[2]]
                vertices += [v3[0], v3[1], v3[2], 1, 0, n3[0], n3[1], n3[2]]
                indices += [ c + 0, c + 1, c +2 ]
                c += 3
            
            # Creamos los quads intermedios
            else: 
                vertices += [v0[0], v0[1], v0[2], 0, 0, n0[0], n0[1], n0[2]]
                vertices += [v1[0], v1[1], v1[2], 0, 1, n1[0], n1[1], n1[2]]
                vertices += [v2[0], v2[1], v2[2], 1, 1, n2[0], n2[1], n2[2]]
                vertices += [v3[0], v3[1], v3[2], 0, 1, n3[0], n3[1], n3[2]]
                indices += [ c + 0, c + 1, c +2 ]
                indices += [ c + 2, c + 3, c + 0 ]
                c += 4
    return bs.Shape(vertices, indices)


def createSphereNode(r, g, b, pipeline):
    sphere = createGPUShape(pipeline, createColorNormalSphere(20, r,g,b))

    sphereNode = sg.SceneGraphNode("sphere")
    sphereNode.transform =tr.matmul([
        tr.translate(0.25,0.15,-0.35),
        tr.scale(0.3,0.3,0.3)
    ])
    sphereNode.childs = [sphere]

    scaledSphere = sg.SceneGraphNode("sc_sphere")
    scaledSphere.transform = tr.scale(5, 5, 5)
    scaledSphere.childs = [sphereNode]

    return scaledSphere

def createTexSphereNode(pipeline):
    sphere = createTextureGPUShape(createTextureNormalSphere(20), pipeline, stonePath)

    sphereNode = sg.SceneGraphNode("sphere")
    sphereNode.transform =tr.matmul([
        tr.translate(-0.25,0.25,-0.35),
        tr.scale(0.3,0.3,0.3)
    ])
    sphereNode.childs = [sphere]

    scaledSphere = sg.SceneGraphNode("sc_sphere")
    scaledSphere.transform = tr.scale(5, 5, 5)
    scaledSphere.childs = [sphereNode]

    return scaledSphere

def createTexureNormalToroid(N):
    vertices = []
    indices = []

    dalpha = 2 * np.pi /(N-1)
    dbeta = 2 * np.pi /(N-1)
    R=0.3
    r = 0.2
    c = 0
    for i in range(N-1):
        beta = i * dbeta
        beta2= (i+1) * dbeta
        for j in range(N-1):
            alpha = j * dalpha
            alpha2 = (j+1) * dalpha

            v0 = [(R + r*np.cos(alpha))*np.cos(beta), (R+r*np.cos(alpha))*np.sin(beta), r*np.sin(alpha)]
            v1 = [(R + r*np.cos(alpha2))*np.cos(beta), (R+r*np.cos(alpha2))*np.sin(beta), r*np.sin(alpha2)]
            v2 = [(R + r*np.cos(alpha2))*np.cos(beta2), (R+r*np.cos(alpha2))*np.sin(beta2), r*np.sin(alpha2)]
            v3 = [(R + r*np.cos(alpha))*np.cos(beta2), (R+r*np.cos(alpha))*np.sin(beta2), r*np.sin(alpha)]

            n0 = [np.cos(alpha) * np.cos(beta), np.cos(alpha) * np.sin(beta), np.sin(alpha)]
            n1 = [np.cos(alpha2) * np.cos(beta), np.cos(alpha2) * np.sin(beta), np.sin(alpha2)]
            n2 = [np.cos(alpha2) * np.cos(beta2), np.cos(alpha2) * np.sin(beta2), np.sin(alpha2)]
            n3 = [np.cos(alpha) * np.cos(beta2), np.cos(alpha) * np.sin(beta2), np.sin(alpha)]

            vertices += [v0[0], v0[1], v0[2], 0, 0, n0[0], n0[1], n0[2]]
            vertices += [v1[0], v1[1], v1[2], 0, 1, n1[0], n1[1], n1[2]]
            vertices += [v2[0], v2[1], v2[2], 1, 1, n2[0], n2[1], n2[2]]
            vertices += [v3[0], v3[1], v3[2], 0, 1, n3[0], n3[1], n3[2]]
            indices += [ c + 0, c + 1, c +2 ]
            indices += [ c + 2, c + 3, c + 0 ]
            c += 4
    return bs.Shape(vertices, indices)

def createTexToroidNode(pipeline, path):
    toroid = createTextureGPUShape(createTexureNormalToroid(20), pipeline, path)

    toroidNode = sg.SceneGraphNode("toroid")
    toroidNode.transform =tr.matmul([
        tr.translate(-0.25,0.25,-0.44),
        tr.scale(0.3,0.3,0.3)
    ])
    toroidNode.childs = [toroid]

    scaledToroid = sg.SceneGraphNode("sc_toroid")
    scaledToroid.transform = tr.scale(5, 5, 5)
    scaledToroid.childs = [toroidNode]

    return scaledToroid
