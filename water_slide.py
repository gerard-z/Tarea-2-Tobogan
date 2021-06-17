""" Segunda parte, creación del tobogán, modelando un bote con OBJ, donde el tobogán tiene agua y obstáculos de por medio """

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.performance_monitor as pm
import grafica.scene_graph as sg
import grafica.newLightShaders as nl
from shapes3d import *
from resources import *

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tobogán"

    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    controller = Controller(width, height)
    # Conectando las funciones: on_key, cursor_pos_callback, mouse_button_callback y scroll_callback del controlador al teclado y mouse
    glfw.set_key_callback(window, controller.on_key)

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)  # Deja la cámara centrada en la ventana con movimiento ilimitado
    glfw.set_cursor_pos(window, width/2, height/2)                  # Fija el mouse en el centro de la ventana
    glfw.set_cursor_pos_callback(window, controller.cursor_pos_callback)
    glfw.set_mouse_button_callback(window, controller.mouse_button_callback)


    # Diferentes shader 3D que consideran la iluminación de la linterna
    phongPipeline = nl.SimplePhongSpotlightShaderProgram()
    phongTexPipeline = nl.SimplePhongTextureSpotlightShaderProgram()

    # Este shader 3D no considera la iluminación de la linterna
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()

    # Este shader es uno en 2D
    pipeline2D = es.SimpleTransformShaderProgram()

    # Setting up the clear screen color
    glClearColor(0.65, 0.65, 0.65, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creando curva
    pos = np.zeros((7,3))
    pos[0]= np.array([-0.204, 0.404, 0])
    pos[1]= np.array([-0.5, 0.102, 0])
    pos[2]= np.array([-0.303, -0.053, 0])
    pos[3]= np.array([-0.0, 0.009, 0])
    pos[4]= np.array([0.225, -0.095, 0])
    pos[5]= np.array([0.49, -0.155, 0])
    pos[6]= np.array([0.497, -0.01, 0])


    # Creating shapes on GPU memory
    curva = CatmullRom(pos)
    gpuGraph = curva.drawGraph(pipeline2D, 500)


    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)
    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)
    
    # Variables últies
    t0 = glfw.get_time()
    light = Iluminacion()

    # Application loop
    while not glfw.window_should_close(window):
        # Rendimiento en la pantalla
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))
        
        # Variables del tiempo
        t1 = glfw.get_time()
        delta = t1 -t0
        t0 = t1

        # Using GLFW to check for input events
        glfw.poll_events()

        # Definimos la cámara de la aplicación
        controller.update_camera(delta)
        camera = controller.get_camera()
        viewMatrix = camera.update_view()

        # Setting up the projection transform
        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glUseProgram(pipeline2D.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline2D.shaderProgram, "transform"), 1, GL_TRUE, tr.identity())
        pipeline2D.drawCall(gpuGraph, GL_LINE_STRIP)
        
        
        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    gpuGraph.clear()

    glfw.terminate()