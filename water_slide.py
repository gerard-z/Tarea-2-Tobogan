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


velocidad = 1
N = 10

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
    phongTexPipeline = nl.DoublePhongTextureSpotlightShaderProgram()

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
    pos = np.zeros((10,3))
    pos[0]= np.array([0, 0, 0])
    pos[1]= np.array([4, 0, 0])
    pos[2]= np.array([8, 0, 0])
    pos[3]= np.array([12, 0, -1])
    pos[4]= np.array([16, 0, -3])
    pos[5]= np.array([20, 0, -5])
    pos[6]= np.array([24, 4, -5])
    pos[7]= np.array([28, 8, -5])
    pos[8]= np.array([32, 12, -5])
    pos[9]= np.array([36, 16, -5])

    pos2 = np.zeros((10,3))
    pos2[0]= np.array([0, 0, 0])
    pos2[1]= np.array([4, 4, -5])
    pos2[2]= np.array([8, 8, -10])
    pos2[3]= np.array([12, 12, -15])
    pos2[4]= np.array([16, 16, -20])
    pos2[5]= np.array([20, 20, -30])
    pos2[6]= np.array([24, 24, -40])
    pos2[7]= np.array([28, 28, -60])
    pos2[8]= np.array([32, 32, -80])
    pos2[9]= np.array([36, 36, -140])

    pos3 = np.zeros((10,3))
    pos3[0]= np.array([0, 0, 0])
    pos3[1]= np.array([0, 4, 0])
    pos3[2]= np.array([0, 8, 0])
    pos3[3]= np.array([0, 12, -1])
    pos3[4]= np.array([0, 16, -3])
    pos3[5]= np.array([0, 20, -5])
    pos3[6]= np.array([0, 24, -5])
    pos3[7]= np.array([0, 28, -5])
    pos3[8]= np.array([0, 32, -5])
    pos3[9]= np.array([0, 36, -5])

    pos1 = np.zeros((25,3))
    pos1[0] = np.array([-20, 10, 0])
    pos1[1] = np.array([-10, 5, 0])
    pos1[2] = np.array([0,0,0])
    pos1[3] = np.array([10,-5,0])
    pos1[4] = np.array([20,0,-5])
    pos1[5] = np.array([25,10,-10])
    pos1[6] = np.array([20, 20, -15])
    pos1[7] = np.array([10, 25, -20])
    pos1[8] = np.array([0, 20, -25])
    pos1[9] = np.array([-5, 10, -25])
    pos1[10] = np.array([-5, 0, -30])
    pos1[11] = np.array([-10, -10, -35])
    pos1[12] = np.array([-20, -15, -35])
    pos1[13] = np.array([-30, -10, -40])
    pos1[14] = np.array([-35, 0, -45])
    pos1[15] = np.array([-30, 10, -50])
    pos1[16] = np.array([-20, 15, -55])
    pos1[17] = np.array([-10, 15, -60])
    pos1[18] = np.array([0, 10, -60])
    pos1[19] = np.array([5, 0, -60])
    pos1[20] = np.array([5, -10, -60])
    pos1[21] = np.array([5, -20, -60])
    pos1[22] = np.array([0, -30, -60])
    pos1[23] = np.array([-10, -35, -60])
    pos1[24] = np.array([-20, -35, -60])




    # Creating shapes on GPU memory
    curva = CatmullRom(pos1, velocidad)
    tobogan, toboganTex = createSlide(curva, 100)
    gpuTobogan = createTobogan(phongPipeline, tobogan)
    gpuToboganTex = createTexTobogan(phongTexPipeline, toboganTex)


    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)
    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)
    
    # Variables últies
    t0 = glfw.get_time()
    light = Iluminacion()
    T = t0
    Lpos = curva.puntos//4
    Lpos1 = curva.getvertice(Lpos)
    Lpos2 = curva.getvertice(Lpos*2)
    Lpos3 = curva.getvertice(Lpos*3)
    Lpos1[2] += curva.radio*0.9
    Lpos2[2] += curva.radio*0.9
    Lpos3[2] += curva.radio*0.9
    Lpos4 = curva.getvertice(curva.puntos-16)

    # Application loop
    while not glfw.window_should_close(window):
        # Rendimiento en la pantalla
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))
        
        # Variables del tiempo
        t1 = glfw.get_time()
        delta = t1 -t0
        t0 = t1

        T += 3*delta * velocidad

        c1 = np.abs(((0.5*t1+0.00) % 2)-1)
        c2 = np.abs(((0.5*t1+0.66) % 2)-1)
        c3 = np.abs(((0.5*t1+1.32) % 2)-1)

        # Using GLFW to check for input events
        glfw.poll_events()

        # ALgunos parámetros de movimiento
        if (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS):
            delta *= 4
        if (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS):
            delta /= 4

        # Definimos la cámara de la aplicación
        controller.update_camera(delta, curva)
        camera = controller.get_camera()
        # Actualizamos eye y at de la cámara en el tobogán
        if controller.camara == 2:
            slideEye, slideAt = curva.camera(delta, controller)
            camera.eye = slideEye
            camera.at = slideAt

        viewMatrix = camera.update_view()

        # iluminación
        lightPos = controller.getEyeCamera()
        lightDirection = controller.getAtCamera() - controller.getEyeCamera()

        # definiendo parámetros del foco
        if controller.light==1:
            light.setLight(0.6, 30, 1, [0.01, 0.03, 0.04])

        elif controller.light==2:
            light.setLight(0.8, 15, 1, [0.01, 0.02, 0.03])

        elif controller.light==4:
            light.setLight(0, 1, 0, [0.01, 0.03, 0.05])

        else:
            light.setLight(1, 6, 1, [0.01, 0.01, 0.01])

        # Setting up the projection transform
        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        # Shader de texturas
        light.updateLight(phongTexPipeline, lightPos, lightDirection, lightPos)
        light.addLight(0, Lpos1, 1, 1, 1)
        light.addLight(1, Lpos2, 1, 1, 1)
        light.addLight(2, Lpos3, 1, 1, 1)
        light.addLight(3, Lpos4, c1, c2, c3)
        # Enviando información de las texturas
        glUniform1i(glGetUniformLocation(phongTexPipeline.shaderProgram, "TexWater"), 0)
        glUniform1i(glGetUniformLocation(phongTexPipeline.shaderProgram, "TexDisplacement"), 1)
        glUniform1f(glGetUniformLocation(phongTexPipeline.shaderProgram, "time"), T)
        # Enviar matrices de transformaciones
        glUniformMatrix4fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)

        # Iluminación del material
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        # Transformación del modelo
        glUniformMatrix4fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())

        # Drawing
        phongTexPipeline.drawCall(gpuToboganTex)

        # Shader de iluminación
        light.updateLight(phongPipeline, lightPos, lightDirection, lightPos)
        light.addLight(0, Lpos1, 1, 1, 1)
        light.addLight(1, Lpos2, 1, 1, 1)
        light.addLight(2, Lpos3, 1, 1, 1)
        light.addLight(3, Lpos4, c1, c2, c3)
        # Enviar matrices de transformaciones
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)

        # Iluminación del material
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        # Transformación del modelo
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())

        # Drawing
        phongPipeline.drawCall(gpuTobogan)
        
        
        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    gpuTobogan.clear()
    gpuToboganTex.clear()

    glfw.terminate()