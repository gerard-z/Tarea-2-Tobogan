"""Funciones para crear distintas figuras y escenas en 3D """

from numpy.core.arrayprint import format_float_scientific
import openmesh as om
import numpy as np
import numpy.random as rd
from OpenGL.GL import *
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.transformations as tr
import grafica.scene_graph as sg
import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
thisFilePath = os.path.abspath(__file__)
thisFolderPath = os.path.dirname(thisFilePath)
assetsDirectory = os.path.join(thisFolderPath, "sprites")
stonePath = os.path.join(assetsDirectory, "stone.png")
dirtPath = os.path.join(assetsDirectory, "dirt.png")

# Convenience function to ease initialization
def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    return gpuShape

def createTextureGPUShape(shape, pipeline, path):
    # Funcion Conveniente para facilitar la inicializacion de un GPUShape con texturas
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    gpuShape.texture = es.textureSimpleSetup(
        path, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_NEAREST, GL_NEAREST)
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

def calculateNormal(mesh, vertex):
    """ om.mesh() vertex -> np.array3
    Recibe el mesh y un vértice, utilizando la estructura halfedge calcula la normal de las caras
    adyacentes y los promedia para conseguir su norma.
    Por ahora no tiene en cuenta los cálculos previamente hechos :s"""
    normal = np.array([0, 0, 0])            # vector que promediará las normales de las caras adyacentes

    # Antiguo cálculo de normales, sin considerar atributo en la cara
    #point0 = np.array(mesh.point(vertex))   #Coordenadas del vértice original
    #for outhEdge in mesh.voh(vertex):
        #nexthEdge = mesh.next_halfedge_handle(outhEdge)                    # Obtiene el siguiente puntero
        #if mesh.to_vertex_handle(mesh.next_halfedge_handle(nexthEdge))== vertex: # Se reviza que el puntero sea de la misma cara
            #point1= np.array(list(mesh.point(mesh.to_vertex_handle(outhEdge))))      # Obtiene otro vértice de la cara 
            #point2 = np.array(list(mesh.point(mesh.to_vertex_handle(nexthEdge))))    # Obtiene el último vértice de la cara
            #dir1 = point1 - point0          # Calcula el vector que va desde el primer vértice al segundo
            #dir2 = point1 - point2          # Calcula el vector que va desde el tercer vértice al segundo
            #cruz = np.cross(dir2, dir1)     # Obtiene la normal de la cara adyacente
            #normal = normal +  cruz/np.linalg.norm(cruz)   # Se suma la normal de la cara normalizada  
    
    # Cálculo de la normal considerando que cada cara tiene guardada su normal
    outHalfEdge = mesh.halfedge_handle(vertex)  #Se obtiene el half edge de salida
    OutHalfEdge = outHalfEdge
    k = True # Se crea una variable que sirve para indicar si seguimos dentro de las caras vecinas
    while k:
        face = mesh.face_handle(outHalfEdge)        # Obtiene la cara ligada al half edge
        nextHalfEdge = mesh.next_halfedge_handle(outHalfEdge)   # Obtiene el siguiente half edge 
        if mesh.face_handle(nextHalfEdge) != face:    # Revisa que el siguiente half edge está ligado a la misma cara
            k = False   # No lo está
        else:
            inHalfEdge = mesh.next_halfedge_handle(nextHalfEdge)    # Obtiene el siguiente half edge que apuntará al vértice nuevamente
            outHalfEdge = mesh.opposite_halfedge_handle(inHalfEdge) # Se pasa al half edge opuesto que va en salida
            if outHalfEdge == OutHalfEdge: k = False    # Volvemos al half edge del inicio
            Normal = np.array(list(mesh.normal(face))) # Se obtiene la normal calculada en la cara
            normal = normal + Normal    # Se suman las normales

    normal = normal/np.linalg.norm(normal)    # Se obtiene el promedio de las normales
    return normal

######## CREANDO UNA MALLA FRACTAL #####
def fractalMesh(mesh, n):
    k = 0
    newMesh = om.TriMesh()
    while k<n:
        faces = list(mesh.faces())
        for i in range(len(faces)-1):
            vertexs = list(mesh.fv(faces[i]))
            vertex1 = np.array(list(mesh.point(vertexs[0])))
            vertex2 = np.array(list(mesh.point(vertexs[1])))
            vertex3 = np.array(list(mesh.point(vertexs[2])))

            vertex12 = (vertex1 + vertex2)/2
            vertex23 = (vertex2 + vertex3)/2
            vertex13 = (vertex1 + vertex3)/2

            vertex12[2] = (0.5-rd.rand())*0.1 + vertex12[2]
            vertex23[2] = (0.5-rd.rand())*0.1 + vertex23[2]
            vertex13[2] = (0.5-rd.rand())*0.1 + vertex13[2]

            v1 = newMesh.add_vertex(vertex1)
            v2 = newMesh.add_vertex(vertex2)
            v3 = newMesh.add_vertex(vertex3)
            v12 = newMesh.add_vertex(vertex12)
            v23 = newMesh.add_vertex(vertex23)
            v13 = newMesh.add_vertex(vertex13)

            #newMesh.add_face(v12, v13, v23)
            newMesh.add_face(v1, v12, v13)
            newMesh.add_face(v13, v23, v3)
            newMesh.add_face(v12, v2, v23)
            #newMesh.add_face(v1, v2, v3)

        mesh = newMesh
        k+=1
        newMesh = om.TriMesh()

    return mesh


def caveMesh(matriz):
    """ Se crea las 2 mallas de polígonos correspondiente al suelo y el techo, por conveniencia, se utilizarán celdas
    de 5x5 metros cuadrados. (Considerando que los ejes se encontraran efectivamente en metros)
    De esta manera, Lara será capaz de moverse por la celda. """
    sueloMesh = om.TriMesh()
    techoMesh = om.TriMesh()
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

    # Se sabe que los puntos medios de los cuadrados comparten altura con los sus vecinos y por lo tanto son eliminados
    a,b = 0,0
    Xs = np.zeros(N*3)
    Ys = np.zeros(M*3)

    # largo de arregles
    lenXS = len(Xs)-1
    lenYS = len(Ys)-1

    A, B = 0,0
    while a<len(xs):
        if (a-1)%7 !=0 and (a-2)%7 !=0 and (a-3)%7 !=0 and (a-4)%7 !=0:
            Xs[A] = xs[a]
            A += 1
        a += 1
    while b<len(ys):
        if (b-1)%7 !=0 and (b-2)%7 !=0 and (b-3)%7 !=0 and (b-4)%7 !=0:
            Ys[B] = ys[b]
            B += 1
        b += 1
    
    # Se generan los vértices de la malla, utilizando las alturas dadas
    for i in range(lenXS):
        x = Xs[i]
        im = i//3   # Transforma el índice en su correspondiente celda de la matriz
        a = False
        for j in range(lenYS):
            y = Ys[j]
            jm = j//3 # Transforma el índice en su correspondiente celda de la matriz
            b = False
            z0 = matriz[im][jm][0]
            z1 = matriz[im][jm][1]
            if (i+1)//3 != im:
                Im = im+1
                z0 = (z0 + matriz[Im][jm][0])/2
                a = True
            if (j+1)//3 != jm:
                Jm = jm+1
                z0 = (z0 + matriz[im][Jm][0])/2
                b = True
            if a and b:
                z0 = (matriz[im][jm][0] + matriz[im][Jm][0] + matriz[Im][Jm][0] + matriz[Im][jm][0])/4
            
            

            # Agregamos el vértice a la malla correspondiente
            sueloMesh.add_vertex([x, y, z0])
            techoMesh.add_vertex([x, y, z1])

    # Se calcula el índice de cada punto (i, j) de la forma:
    index = lambda i, j: i*lenYS + j

    # Obtenemos los vertices de cada malla, y agregamos las caras
    vertexsuelo = list(sueloMesh.vertices())
    vertextecho = list(techoMesh.vertices())

    sueloMesh.request_vertex_texcoords2D()
    techoMesh.request_vertex_texcoords2D()
    # Agregamos texturas
    for j in range(lenYS):      # en los bordes laterales de la cueva
        k=j-2 
        t = index(lenXS-1, j)
        if j==0: # primer vértice
            sueloMesh.set_texcoord2D(vertexsuelo[j], np.array([0, 1]))
            sueloMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 1]))
            techoMesh.set_texcoord2D(vertexsuelo[j], np.array([0, 1]))
            techoMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 1]))
        elif j==1:
            sueloMesh.set_texcoord2D(vertexsuelo[j], np.array([0, 1/6]))
            sueloMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 1/6]))
            techoMesh.set_texcoord2D(vertexsuelo[j], np.array([0, 1/6]))
            techoMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 1/6]))
        elif (k)%3==1:
            sueloMesh.set_texcoord2D(vertexsuelo[j], np.array([0, 5/6]))
            sueloMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 5/6]))
            techoMesh.set_texcoord2D(vertexsuelo[j], np.array([0, 5/6]))
            techoMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 5/6]))
        elif (k)%3==0:
            sueloMesh.set_texcoord2D(vertexsuelo[j], np.array([0, 0]))
            sueloMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 0]))
            techoMesh.set_texcoord2D(vertexsuelo[j], np.array([0, 0]))
            techoMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 0]))
        elif k%3==2:
            sueloMesh.set_texcoord2D(vertexsuelo[k], np.array([0, 1/6]))
            sueloMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 1/6]))
            techoMesh.set_texcoord2D(vertexsuelo[k], np.array([0, 1/6]))
            techoMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 1/6]))

    for i in range(1, lenXS):      # en los bordes horizontales de la cueva
        k=i-2 
        t = index(i, lenYS-1)
        j = index(i, 0)
        if i==1:
            sueloMesh.set_texcoord2D(vertexsuelo[j], np.array([5/6, 1]))
            sueloMesh.set_texcoord2D(vertexsuelo[t], np.array([5/6, 0]))
            techoMesh.set_texcoord2D(vertexsuelo[j], np.array([5/6, 1]))
            techoMesh.set_texcoord2D(vertexsuelo[t], np.array([5/6, 0]))
        elif (k)%3==1:
            sueloMesh.set_texcoord2D(vertexsuelo[j], np.array([1/6, 1]))
            sueloMesh.set_texcoord2D(vertexsuelo[t], np.array([1/6, 0]))
            techoMesh.set_texcoord2D(vertexsuelo[j], np.array([1/6, 1]))
            techoMesh.set_texcoord2D(vertexsuelo[t], np.array([1/6, 0]))
        elif (k)%3==0:
            sueloMesh.set_texcoord2D(vertexsuelo[j], np.array([1, 1]))
            sueloMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 0]))
            techoMesh.set_texcoord2D(vertexsuelo[j], np.array([1, 1]))
            techoMesh.set_texcoord2D(vertexsuelo[t], np.array([1, 0]))
        elif k%3==2:
            sueloMesh.set_texcoord2D(vertexsuelo[j], np.array([5/6, 1]))
            sueloMesh.set_texcoord2D(vertexsuelo[t], np.array([5/6, 0]))
            techoMesh.set_texcoord2D(vertexsuelo[j], np.array([5/6, 1]))
            techoMesh.set_texcoord2D(vertexsuelo[t], np.array([5/6, 0]))
    """# El vértice esquina superior derecha
    sueloMesh.set_texcoord2D(vertexsuelo[index(lenXS-1,lenYS-1)], np.array([1, 0]))
    techoMesh.set_texcoord2D(vertexsuelo[index(lenXS-1,lenYS-1)], np.array([1, 0]))
    """
            
    # Vértices en medio
    for i in range(1,lenXS-1):
        I = (i-2)%3
        for j in range(1,lenYS-1):
            k = index(i,j)
            J = (j-2)%3
            tex = np.array([0,0])
            if I==0:
                tex[0] = 1
            elif I==1:
                tex[0] = 1/6
            else:
                tex[0] = 5/6
            if J==0:
                tex[1] = 0
            elif J==1:
                tex[1] = 5/6
            else:
                tex[1] = 1/6
            sueloMesh.set_texcoord2D(vertexsuelo[k], tex)
            techoMesh.set_texcoord2D(vertexsuelo[k], tex)
    # Notar que siempre falta una coordenada para cerrar la celda, esto es debido a que algunos vértices corresponderán
    # al cierre de una celda y la apertura de otra. Se corregirá en el shader específico para la textura en la cueva

    # Se crean las caras para cada cuadrado de la celda
    for i in range(lenXS-1):
        for j in range(lenYS-1):
            # los índices:
            isw = index(i,j)
            ise = index(i+1,j)
            ine = index(i+1,j+1)
            inw = index(i,j+1)
            # Se agregan las caras
            sueloMesh.add_face(vertexsuelo[isw], vertexsuelo[ise], vertexsuelo[ine])
            sueloMesh.add_face(vertexsuelo[ine], vertexsuelo[inw], vertexsuelo[isw])

            techoMesh.add_face(vertextecho[isw], vertextecho[ise], vertextecho[ine])
            techoMesh.add_face(vertextecho[ine], vertextecho[inw], vertextecho[isw])

    

    # Se entregan las mallas
    return (sueloMesh, techoMesh)

def get_vertexs_and_indexes(mesh, orientation):
    # Obtenemos las caras de la malla
    faces = mesh.faces()

    # orientation indica si las normales deben apuntar abajo(-1) o arriba(1)
    assert orientation==1 or orientation==-1, "La orientación debe ser indicada con 1 o -1"

    # Creamos una lista para los vertices e indices
    vertexs = []

    # Se activa la propiedad de agregar normales en las caras, sin embargo, no se utilizará el método de openmesh para
    # calcular dichas normales, sino se implementará una función propia para utilizar la estructura half-edge y simplemente
    # utiizar dicho espacio para guardar el vector normal resultante.
    mesh.request_face_normals()
    # Se calcula la normal de cada cara
    for face in mesh.faces():
        vertices = list(mesh.fv(face)) # Se obtiene los vértices de la cara
        P0 = np.array(mesh.point(vertices[0]).tolist())    # Se obtiene la coordenada del vértice 1
        P1 = np.array(mesh.point(vertices[1]).tolist())    # Se obtiene la coordenada del vértice 2
        P2 = np.array(mesh.point(vertices[2]).tolist())    # Se obtiene la coordenada del vértice 3
        dir1 = P1 - P0          # Calcula el vector que va desde el primer vértice al segundo
        dir2 = P1 - P2          # Calcula el vector que va desde el tercer vértice al segundo
        cruz = np.cross(dir2, dir1)     # Obtiene la normal de la cara
        mesh.set_normal(face, cruz/np.linalg.norm(cruz))    # Se guarda la normal normalizada como atributo en la cara

    # Obtenemos los vertices y los recorremos
    for vertex in mesh.vertices():
        vertexs += mesh.point(vertex).tolist()
        # Agregamos un color al azar
        vertexs += mesh.texcoord2D(vertex).tolist()
        # Agregamos la norma
        normal = calculateNormal(mesh, vertex)
        normal = orientation * normal

        vertexs += [normal[0], normal[1], normal[2]]


        

    indexes = []

    for face in faces:
        # Obtenemos los vertices de la cara
        face_indexes = mesh.fv(face)
        for vertex in face_indexes:
            # Obtenemos el numero de indice y lo agregamos a la lista
            indexes += [vertex.idx()]

    return vertexs, indexes

def createCave(pipeline, meshs):
    # obtenemos los vértices e índices del suelo y del techo
    sVertices, sIndices = get_vertexs_and_indexes(meshs[0],1)
    tVertices, tIndices = get_vertexs_and_indexes(meshs[1],-1)
    sueloShape = bs.Shape(sVertices, sIndices)
    techoShape = bs.Shape(tVertices, tIndices)

    gpuSuelo = createGPUShape(pipeline, sueloShape)
    gpuTecho = createGPUShape(pipeline, techoShape)

    return gpuSuelo, gpuTecho

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
