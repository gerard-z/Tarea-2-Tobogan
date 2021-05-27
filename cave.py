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

    glfw.set_cursor_pos_callback(window, controller.cursor_pos_callback)
    glfw.set_mouse_button_callback(window, controller.mouse_button_callback)

     # Different shader programs for different lighting strategies
    #phongPipeline = nl.MultiplePhongShaderProgram()
    #phongTexPipeline = nl.MultipleTexturePhongShaderProgram()

    phongPipeline = nl.SimplePhongSpotlightShaderProgram()
    phongTexPipeline = nl.SimplePhongTextureSpotlightShaderProgram()

    # This shader program does not consider lighting
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

    gpuRedCube = createGPUShape(mvpPipeline, bs.createColorCube(1,0,0))

    gpuRedQuad = createGPUShape(pipeline2D, bs.createColorQuad(1, 0, 0))

    scene = createScene(phongPipeline)
    cube1 = createCube1(phongPipeline)
    cube2 = createCube2(phongPipeline)
    tex_toroid = createTexToroidNode(phongTexPipeline, stonePath)
    tex_toroid2 = createTexToroidNode(phongTexPipeline, dirtPath)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)
    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)
    t0 = glfw.get_time()
    r = 0.5
    g = 0
    b = 0.25

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
        lightingPipeline = phongPipeline
        pi = np.pi
        #lightposition0 = [1.7*np.cos(t1), 1.7*np.sin(t1), 2.3]
        #lightposition1 = [1.7*np.cos(t1 + 2*pi/3), 1.7*np.sin(t1 + 2*pi/3), 2.3]
        #lightposition2 = [1.7*np.cos(t1 + 4*pi/3), 1.7*np.sin(t1 + 4*pi/3), 2.3]
        #lightposition0 = [0, 0, 2.3]

        #c1 = np.abs(((0.5*t1+0.00) % 2)-1)
        #c2 = np.abs(((0.5*t1+0.66) % 2)-1)
        #c3 = np.abs(((0.5*t1+1.32) % 2)-1)

        # Setting all uniform shader variables
        LightPower = 0.9
        
        glUseProgram(lightingPipeline.shaderProgram)
        # Position of all light
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "lightPos0"), 1,lightposition0)
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "lightPos1"), 1,lightposition1)
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "lightPos2"), 1,lightposition2)
        glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "lightPos"), 1,lightPos)

        # White light in all components: ambient, diffuse and specular.
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "La0"), 1, [c1, c2, c3])
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "La1"), 1, [c2, c3, c1])
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "La2"), 1, [c3, c1, c2])
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "Ld0"), 1, [c1, c2, c3])
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "Ld1"), 1, [c2, c3, c1])
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "Ld2"), 1, [c3, c1, c2])
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "Ls0"), 1, [c1, c2, c3])
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "Ls1"), 1, [c2, c3, c1])
        #glUniform3fv(glGetUniformLocation(lightingPipeline.shaderProgram, "Ls2"), 1, [c3, c1, c2])
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "La"), 0.1, 0.1, 0.1)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ld"), LightPower, LightPower, LightPower)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        # Object is barely visible at only ambient. Diffuse behavior is slightly red. Sparkles are white
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "viewPosition"), camera.eye[0], camera.eye[1], camera.eye[2])
        glUniform1ui(glGetUniformLocation(lightingPipeline.shaderProgram, "shininess"), 100)
        glUniform1ui(glGetUniformLocation(lightingPipeline.shaderProgram, "concentration"), 50)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "lightDirection"), lightDirection[0], lightDirection[1], lightDirection[2])
        
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "constantAttenuation"), 0.01)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "quadraticAttenuation"), 0.05)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())

        # Drawing
        sg.drawSceneGraphNode(scene, lightingPipeline, "model")
        sg.drawSceneGraphNode(cube1, lightingPipeline, "model")
        sg.drawSceneGraphNode(cube2, lightingPipeline, "model")
        
        glUseProgram(phongTexPipeline.shaderProgram)
        # Position of all light
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "lightPos0"), 1,lightposition0)
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "lightPos1"), 1,lightposition1)
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "lightPos2"), 1,lightposition2)
        glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "lightPos"), 1,lightPos)

        # White light in all components: ambient, diffuse and specular.
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "La0"), 1, [c1, c2, c3])
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "La1"), 1, [c2, c3, c1])
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "La2"), 1, [c3, c1, c2])
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ld0"), 1, [c1, c2, c3])
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ld1"), 1, [c2, c3, c1])
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ld2"), 1, [c3, c1, c2])
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ls0"), 1, [c1, c2, c3])
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ls1"), 1, [c2, c3, c1])
        #glUniform3fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ls2"), 1, [c3, c1, c2])
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "La"), 0.1, 0.1, 0.1)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ld"), LightPower, LightPower, LightPower)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        # Object is barely visible at only ambient. Diffuse behavior is slightly red. Sparkles are white
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "viewPosition"), camera.eye[0], camera.eye[1], camera.eye[2])
        glUniform1ui(glGetUniformLocation(phongTexPipeline.shaderProgram, "shininess"), 100)
        glUniform1ui(glGetUniformLocation(phongTexPipeline.shaderProgram, "concentration"), 50)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "lightDirection"), lightDirection[0], lightDirection[1], lightDirection[2])
        
        glUniform1f(glGetUniformLocation(phongTexPipeline.shaderProgram, "constantAttenuation"), 0.001)
        glUniform1f(glGetUniformLocation(phongTexPipeline.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(phongTexPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)
        glUniformMatrix4fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())

        sg.drawSceneGraphNode(tex_toroid, phongTexPipeline, "model")
        # Características del otro toroide (como es tierra es más opaco)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ka"), 0.1, 0.1, 0.1)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Kd"), 0.4, 0.4, 0.4)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ks"), 0.0, 0.0, 0.0)
        sg.drawSceneGraphNode(tex_toroid2, phongTexPipeline, "model", tr.matmul([tr.translate(1.21, 0.5, 0.55),tr.rotationY(-pi/4)]))

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

    glfw.terminate()