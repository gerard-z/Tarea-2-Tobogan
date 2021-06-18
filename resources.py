# coding=utf-8
""" Módulo que contiene las clases y objetos relacionados al jugador, entidades, cámara y
mecánicas de la aplicación, en resumen, todo lo que no tiene que ver con el apartado de modelos
geométricos ni la parte gráfica """

from sys import path_importer_cache
import glfw
import numpy as np
import grafica.transformations as tr
import grafica.easy_shaders as es
from shapes3d import *
from numpy import random as rd

# Cámara en tercera persona
class ThirdCamera:
    def __init__(self, x, y , z):
        self.at = np.array([x, y, z+0.1])
        self.theta = -np.pi/2
        self.eye = np.array([x, y - 3.0, z + 0.1])
        self.up = np.array([0, 0, 1])

    # Determina el ángulo theta
    def set_theta(self, theta):
        self.theta = theta

    # Actualiza la matriz de vista y la retorna
    def update_view(self):
        self.eye[0] = 3 * np.cos(self.theta) + self.at[0]
        self.eye[1] = 3 * np.sin(self.theta) + self.at[1]
        self.eye[2] = self.at[2]

        viewMatrix = tr.lookAt(
            self.eye,
            self.at,
            self.up
        )
        return viewMatrix

class FirstCamera:
    def __init__(self, x, y, z):
        self.at = np.array([x, y + 3.0, z + 0.0])
        self.theta = -np.pi/2
        self.phi = np.pi/2
        self.eye = np.array([x, y, z + 0.0])
        self.up = np.array([0, 0, 1])

    # Determina el ángulo theta
    def set_theta(self, theta):
        self.theta = theta + np.pi
    
    def set_phi(self, phi):
        self.phi = phi

    # Actualiza la matriz de vista y la retorna
    def update_view(self):
        self.at[0] = 3*np.cos(self.theta) * np.sin(self.phi) + self.eye[0]
        self.at[1] = 3*np.sin(self.theta) * np.sin(self.phi) + self.eye[1]
        self.at[2] = 3*np.cos(self.phi) + self.eye[2]

        viewMatrix = tr.lookAt(
            self.eye,
            self.at,
            self.up
        )
        return viewMatrix
        

    
# Clase del controlador, tiene distintos parámetros que son utilizados para albergar la información de lo que ocurre
# en la aplicación.
class Controller:
    def __init__(self, width, height):
        self.fillPolygon = True
        self.waterEffect = False
        self.width = width
        self.height = height

        self.is_a_pressed = False

        self.camera = ThirdCamera(0, 0, 2.5)
        self.camara = 3

        self.light = 1

        self.suelo = None
        self.techo = None

        self.leftClickOn = False
        self.rightClickOn = False
        self.mousePos = (0.0, 0.0)

    # Función que retorna la cámara que se está utilizando
    def get_camera(self):
        return self.camera

    # Función que entrega la posición del vector at
    def getAtCamera(self):
        return self.camera.at
    
    # Función que obtiene la posición del vector eye
    def getEyeCamera(self):
        return self.camera.eye

    # Función que entrega el ángulo theta
    def getThetaCamera(self):
        return self.camera.theta

    # Función que le entrega el mapa al controlador
    def setMap(self, suelo, techo):
        # Se calcula si es posible avanzar en ciertas coordenadas
        (N, M) = (suelo.N, suelo.M)
        sueloV = np.array(suelo.mesh.points())
        techoV = np.array(techo.points())
        x = sueloV[: , 0]
        y = sueloV[: , 1]
        zs = sueloV[:, 2]
        zt = techoV[:, 2]

        
        self.suelo = suelo
        self.techo = techo

    # Función que detecta que tecla se está presionando
    def on_key(self, window, key, scancode, action, mods):
        
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self.fillPolygon = not self.fillPolygon

            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

            if key == glfw.KEY_LEFT_CONTROL:
                self.waterEffect = not self.waterEffect

            if key == glfw.KEY_A:
                self.is_a_pressed = not self.is_a_pressed
            
            if key == glfw.KEY_1:
                self.light = 1
        
            if key == glfw.KEY_2:
                self.light = 2

            if key == glfw.KEY_3:
                self.light = 3

            if key == glfw.KEY_4:
                self.light = 4

    # Función que obtiene las coordenadas de la posición del mouse y las traduce en coordenadas de openGL
    def cursor_pos_callback(self, window, x, y):
        mousePosX = 2 * (x - self.width/2) / self.width

        if y<0:
            glfw.set_cursor_pos(window, x, 0)
        elif y>self.height:
            glfw.set_cursor_pos(window, x, self.height)


        mousePosY = 2 * (y - self.height/2) / self.height

        self.mousePos = (mousePosX, mousePosY)

    # Función que identifica si los botones del mouse son presionados
    def mouse_button_callback(self, window, button, action, mods):
        """
        glfw.MOUSE_BUTTON_1: left click
        glfw.MOUSE_BUTTON_2: right click
        glfw.MOUSE_BUTTON_3: scroll click
        """

        if (action == glfw.PRESS or action == glfw.REPEAT):
            if (button == glfw.MOUSE_BUTTON_1):
                self.leftClickOn = True

            if (button == glfw.MOUSE_BUTTON_2):
                self.rightClickOn = True

            if (button == glfw.MOUSE_BUTTON_3):
                pass

        elif (action ==glfw.RELEASE):
            if (button == glfw.MOUSE_BUTTON_1):
                self.leftClickOn = False
            if (button == glfw.MOUSE_BUTTON_2):
                self.rightClickOn = False

    #Funcion que recibe el input para manejar la camara y el tipo de esta, incluye ek movimiento del personaje
    def update_camera(self, delta):
        # Selecciona la cámara a utilizar
        if self.is_a_pressed and self.camara != 1:
            x = self.camera.at[0]
            y = self.camera.at[1]
            z = self.camera.at[2]-0.1
            self.camera = FirstCamera(x, y, z)
            self.camara = 1
        elif not self.is_a_pressed and self.camara != 3:
            x = self.camera.eye[0]
            y = self.camera.eye[1]
            z = self.camera.eye[2]
            self.camera = ThirdCamera(x, y, z)
            self.camara = 3

        #suelo = self.suelo
        #(N, M) = suelo.shape
        #techo = self.techo
        #n = np.ceil(N/2)
        #m = np.ceil(M/2)

        direction = self.camera.at - self.camera.eye
        dx, dy = direction[0]/3, direction[1]/3
        theta = -self.mousePos[0] * 2 * np.pi - np.pi/2

        mouseY = self.mousePos[1]
        phi = mouseY * (np.pi/2-0.01) + np.pi/2

        if self.camara == 3:
            #x = self.camera.at[0]+n
            #y = self.camera.at[1]+m
            if self.leftClickOn:# and techo[int(np.round(x+dx))][int(np.round(y+dy))]>=self.camera.at[2]:
                self.camera.at += direction * delta

            if self.rightClickOn:# and techo[int(np.round(x-dx))][int(np.round(y-dy))]>=self.camera.at[2]:
                self.camera.at -= direction * delta
            
            #x = int(self.camera.at[0]+n)
            #y = int(self.camera.at[1]+m)
            #self.camera.at[2] = suelo[x][y]+1.2
            
        elif self.camara == 1:
            #x = self.camera.at[0]+n
            #y = self.camera.at[1]+m
            if self.leftClickOn:# and techo[int(np.round(x+dx))][int(np.round(y+dy))]>=self.camera.eye[2]+0.5:
                self.camera.eye += direction * delta

            if self.rightClickOn:# and techo[int(np.round(x-dx))][int(np.round(y-dy))]>=self.camera.eye[2]+0.5:
                self.camera.eye -= direction * delta
            self.camera.set_phi(phi)

            #x = int(self.camera.eye[0]+n)
            #y = int(self.camera.eye[1]+m)
            #self.camera.eye[2] = suelo[x][y]+0.7


        self.camera.set_theta(theta)

# Clase iluminación, crea los parámetros y las funciones para inicializar los shaders con normales.
class Iluminacion:
    def __init__(self):
        # Características de la luz por defecto
        self.LightPower = 0.8
        self.lightConcentration =30
        self.lightShininess = 1
        self.constantAttenuation = 0.01
        self.linearAttenuation = 0.03
        self.quadraticAttenuation = 0.05

    # fija los nuevos datos para la luz
    def setLight(self, Power, Concentration, Shininess, Attenuation):
        self.LightPower = Power
        self.lightConcentration = Concentration
        self.lightShininess = Shininess
        self.constantAttenuation = Attenuation[0]
        self.linearAttenuation = Attenuation[1]
        self.quadraticAttenuation = Attenuation[2]

    # Actualiza los parámetros de la luz al shader
    def updateLight(self, Pipeline, Pos, Direction, Camera):
        # Se guardan los variables
        LightPower = self.LightPower
        lightShininess = self.lightShininess
        lightConcentration = self.lightConcentration
        constantAttenuation = self.constantAttenuation
        linearAttenuation = self.linearAttenuation
        quadraticAttenuation = self.quadraticAttenuation
        # Se activa el program
        glUseProgram(Pipeline.shaderProgram)
        # Se envían los uniforms
        glUniform3fv(glGetUniformLocation(Pipeline.shaderProgram, "lightPos"), 1, Pos)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "La"), 0.3, 0.3, 0.3)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Ld"), LightPower, LightPower, LightPower)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Ls"), lightShininess, lightShininess, lightShininess)

        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "viewPosition"), Camera[0], Camera[1], Camera[2])
        glUniform1ui(glGetUniformLocation(Pipeline.shaderProgram, "shininess"), 100)
        glUniform1ui(glGetUniformLocation(Pipeline.shaderProgram, "concentration"), lightConcentration)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "lightDirection"), Direction[0], Direction[1], Direction[2])

        glUniform1f(glGetUniformLocation(Pipeline.shaderProgram, "constantAttenuation"), constantAttenuation)
        glUniform1f(glGetUniformLocation(Pipeline.shaderProgram, "linearAttenuation"), linearAttenuation)
        glUniform1f(glGetUniformLocation(Pipeline.shaderProgram, "quadraticAttenuation"), quadraticAttenuation)

# Clase para guardar datos
class mallaTam:
    def __init__(self, mesh, N, M):
        self.mesh = mesh
        self.N = N
        self.M = M
        


# Funciones

def UseProgram2D(Pipeline):
    glUseProgram(Pipeline.shaderProgram)
    # Se envían los uniforms


def calculateNormal(mesh):
    """ om.mesh() -> 
    Recibe el mesh, utilizando la estructura halfedge calcula la normal de las caras
    adyacentes y los promedia para conseguir su norma.
    Por ahora no tiene en cuenta los cálculos previamente hechos :s"""

    # Se activa la propiedad de agregar normales en las caras, sin embargo, no se utilizará el método de openmesh para
    # calcular dichas normales, sino se implementará una función propia para utilizar la estructura half-edge y simplemente
    # utiizar dicho espacio para guardar el vector normal resultante.
    mesh.request_face_normals()
    # Se calcula la normal de cada cara
    for face in mesh.faces():
        vertices = list(mesh.fv(face)) # Se obtiene los vértices de la cara
        P0 = np.array(mesh.point(vertices[0]))    # Se obtiene la coordenada del vértice 1
        P1 = np.array(mesh.point(vertices[1]))    # Se obtiene la coordenada del vértice 2
        P2 = np.array(mesh.point(vertices[2]))    # Se obtiene la coordenada del vértice 3
        dir1 = P1 - P0          # Calcula el vector que va desde el primer vértice al segundo
        dir2 = P1 - P2          # Calcula el vector que va desde el tercer vértice al segundo
        cruz = np.cross(dir2, dir1)     # Obtiene la normal de la cara
        mesh.set_normal(face, cruz/np.linalg.norm(cruz))    # Se guarda la normal normalizada como atributo en la cara 
    
    # Cálculo de la normal considerando que cada cara tiene guardada su normal
    mesh.request_vertex_normals()
    for vertex in mesh.vertices():
        normal = np.array([0, 0, 0])            # vector que promediará las normales de las caras adyacentes
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
                Normal = np.array(mesh.normal(face)) # Se obtiene la normal calculada en la cara
                normal = normal + Normal    # Se suman las normales
        normal = normal/np.linalg.norm(normal)    # Se obtiene el promedio de las normales
        mesh.set_normal(vertex, normal)