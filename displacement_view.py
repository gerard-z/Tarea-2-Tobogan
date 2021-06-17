""" Primera parte, implementación de la técnica Displacement mapping para generar el efecto de corriente de agua.
1: La textura de agua
2: La textura de ruido
3: Agua con efectos
Gerard Cathalifaud Salazar"""

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.performance_monitor as pm
from shapes3d import *
from resources import *


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Displacement Map"

    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    controller = Controller(width, height)
    # Conectando las funciones: on_key del controlador al teclado
    glfw.set_key_callback(window, controller.on_key)

    # Diferentes shader 3D que consideran la iluminación de la linterna
    waterShader = es.WaterTextureTransformShaderProgram()

    # Setting up the clear screen color
    glClearColor(0.65, 0.65, 0.65, 1.0)

    # Creating shapes on GPU memory
    waterShape = bs.createTextureQuad(1,1)
    bs.scaleVertices(waterShape, 5, [2, 2, 2])
    gpuWater = createMultipleTextureGPUShape(waterShape, waterShader, [waterPath, displacementPath], minFilterMode=GL_LINEAR, maxFilterMode=GL_LINEAR)

    # Activar el shader
    glUseProgram(waterShader.shaderProgram)

    glUniform1i(glGetUniformLocation(waterShader.shaderProgram, "TexWater"), 0)
    glUniform1i(glGetUniformLocation(waterShader.shaderProgram, "TexDisplacement"), 1)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)
    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)
    
    # Variables últies
    t0 = glfw.get_time()
    waterEffect = 0
    T=t0

    # Application loop
    while not glfw.window_should_close(window):
        # Rendimiento en la pantalla
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))
        
        # Variables del tiempo
        t1 = glfw.get_time()
        delta = t1 -t0
        t0 = t1

        T += delta/8

        # Using GLFW to check for input events
        glfw.poll_events()

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Dibujar el cuadrado
        # Binding samplers to both texture units
        
        glUniform1i(glGetUniformLocation(waterShader.shaderProgram, "waterEffect"), controller.light)
        glUniform1f(glGetUniformLocation(waterShader.shaderProgram, "time"), T)

        waterShader.drawCall(gpuWater)

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    gpuWater.clear()
    glfw.terminate()