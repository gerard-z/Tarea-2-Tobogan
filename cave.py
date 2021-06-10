""" Módulo principal que genera el escenario del juego, la cueva y a Lara Jones, con todas las mecánicas
integradas y en general, la main scene. """

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

    width = 600
    height = 600
    title = "Lara Jones: Cave"

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

    # Creating shapes on GPU memory
    gpuAxis = createGPUShape(mvpPipeline, bs.createAxis(4))
    # Personaje
    gpuRedCube = createGPUShape(mvpPipeline, bs.createColorCube(1,0,0))
    gpuRedQuad = createGPUShape(pipeline2D, bs.createColorQuad(1, 0, 0))

    # Cueva
    Matriz = np.load("map.npy")
    gpuSuelo, gpuTecho = createCave(phongPipeline, caveMesh(Matriz))

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

        # Configuración de la cámara en tercera persona
        if not controller.is_a_pressed:
            glUseProgram(mvpPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)
            # Ubicación de la personaje
            LaraPos = controller.getAtCamera()
            rotation = tr.rotationZ(controller.getThetaCamera())
            position = tr.matmul([tr.translate(LaraPos[0], LaraPos[1], LaraPos[2]-0.7), rotation, tr.uniformScale(0.6)])
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, position)
            mvpPipeline.drawCall(gpuRedCube)
            lightPos = LaraPos
        else:
            lightPos = controller.getEyeCamera()

        lightDirection = camera.at - camera.eye

        # Setting all uniform shader variables
        if controller.light==1:
            light.setLight(0.6, 50, 1, [0.01, 0.03, 0.05])

        elif controller.light==2:
            light.setLight(0.8, 30, 1, [0.01, 0.02, 0.04])

        elif controller.light==4:
            light.setLight(0, 1, 0, [0.01, 0.03, 0.05])

        else:
            light.setLight(1, 10, 1, [0.01, 0.01, 0.02])
        
        # Shader de colores
        light.updateLight(phongPipeline, lightPos, lightDirection, camera.eye)

        # Object is barely visible at only ambient. Diffuse behavior is slightly red. Sparkles are white
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        # Enviar matrices de transformaciones
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.translate(0,0,-2))

        # Drawing
        #CUEVA
        phongPipeline.drawCall(gpuSuelo)
        phongPipeline.drawCall(gpuTecho)

        # Shaders de texturas
        light.updateLight(phongTexPipeline, lightPos, lightDirection, camera.eye)
        
        # Object is barely visible at only ambient. Diffuse behavior is slightly red. Sparkles are white
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        # Dibuja la linterna para la visión en primera persona
        if controller.is_a_pressed:
            glUseProgram(pipeline2D.shaderProgram)
            transform = tr.matmul([tr.translate(0.75, -0.75, 0), tr.uniformScale(0.5)])
            glUniformMatrix4fv(glGetUniformLocation(pipeline2D.shaderProgram, "transform"), 1, GL_TRUE, transform)
            pipeline2D.drawCall(gpuRedQuad)
        
        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    gpuAxis.clear()
    gpuRedCube.clear()
    gpuRedQuad.clear()
    gpuSuelo.clear()
    gpuTecho.clear()

    glfw.terminate()