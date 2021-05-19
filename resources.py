# coding=utf-8
""" Módulo que contiene las clases y objetos relacionados al jugador, entidades, cámara y
mecánicas de la aplicación, en resumen, todo lo que no tiene que ver con el apartado de modelos
geométricos ni la parte gráfica """

import glfw
import numpy as np
import grafica.transformations as tr
import grafica.easy_shaders as es
from shapes3d import *
from numpy import random as rd

# Cámara en tercera persona
class ThirdCamera:
    def __init__(self, x, y):
        self.at = np.array([x, y, 0.1])
        self.theta = -np.pi/2
        self.eye = np.array([x, y - 3.0, 0.1])
        self.up = np.array([0, 0, 1])

    # Determina el ángulo theta
    def set_theta(self, theta):
        self.theta = theta

    # Actualiza la matriz de vista y la retorna
    def update_view(self):
        self.eye[0] = 3 * np.cos(self.theta) + self.at[0]
        self.eye[1] = 3 * np.sin(self.theta) + self.at[1]
        #self.eye[2] = self.at[2]

        viewMatrix = tr.lookAt(
            self.eye,
            self.at,
            self.up
        )
        return viewMatrix

class FirstCamera:
    def __init__(self, x, y):
        self.at = np.array([x, y + 1.0, 0.0])
        self.theta = -np.pi/2
        self.eye = np.array([x, y, 0.0])
        self.up = np.array([0, 0, 1])

    # Determina el ángulo theta
    def set_theta(self, theta):
        self.theta = theta + np.pi

    # Actualiza la matriz de vista y la retorna
    def update_view(self):
        self.at[0] = np.cos(self.theta) + self.eye[0]
        self.at[1] = np.sin(self.theta) + self.eye[1]

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
        self.showAxis = True
        self.width = width
        self.height = height

        self.is_up_pressed = False
        self.is_down_pressed = False
        self.is_left_pressed = False
        self.is_right_pressed = False
        self.is_a_pressed = False
        self.is_z_pressed = False

        self.camera = ThirdCamera(0, 0)
        self.camara = 3

        self.leftClickOn = False
        self.rightClickOn = False
        self.mousePos = (0.0, 0.0)

    # Función que retorna la cámara que se está utilizando
    def get_camera(self):
        return self.camera

    # Función que detecta que tecla se está presionando
    def on_key(self, window, key, scancode, action, mods):
        
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self.fillPolygon = not self.fillPolygon

            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

            if key == glfw.KEY_LEFT_CONTROL:
                self.showAxis = not self.showAxis

            if key == glfw.KEY_A:
                self.is_a_pressed = not self.is_a_pressed
        

    # Función que obtiene las coordenadas de la posición del mouse y las traduce en coordenadas de openGL
    def cursor_pos_callback(self, window, x, y):
        mousePosX = 2 * (x - self.width/2) / self.width
        mousePosY = 2 * (self.height/2 - y) / self.height
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

    # Función que obtiene el scroll del mouse
    def scroll_callback(window, x, y):
        pass



    #Funcion que recibe el input para manejar la camara y el tipo de esta
    def update_camera(self, delta):
        # Selecciona la cámara a utilizar
        if self.is_a_pressed and self.camara != 1:
            x = self.camera.at[0]
            y = self.camera.at[1]
            self.camera = FirstCamera(x, y)
            self.camara = 1
        elif not self.is_a_pressed and self.camara != 3:
            x = self.camera.eye[0]
            y = self.camera.eye[1]
            self.camera = ThirdCamera(x, y)
            self.camara = 3


        direction = self.camera.at - self.camera.eye
        theta = self.mousePos[0] * 2 * np.pi - np.pi/2

        if self.camara == 3:
            if self.leftClickOn:
                self.camera.at += direction * delta

            if self.rightClickOn:
                self.camera.at -= direction * delta
        elif self.camara == 1:
            if self.leftClickOn:
                self.camera.eye += direction * delta

            if self.rightClickOn:
                self.camera.eye -= direction * delta


        self.camera.set_theta(theta)

    # Función que entrega la posición del vector at
    def getAtCamera(self):
        return self.camera.at
    
    # Función que obtiene la posición del vector eye
    def getEyeCamera(self):
        return self.camera.eye

    # Función que entrega el ángulo theta
    def getThetaCamera(self):
        return self.camera.theta