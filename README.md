# Tarea-2-Lara-Jones

Tarea 2a del curso de Modelación y Computación Gráfica para Ingenieros CC3501.
La tarea consiste en la creación de un entorno 3D de una cueva con un personaje controlable que debe recorrer estas cuevas en busca de unos tesoros




IDEAS:
- Agregar una vista aerea que muestre la cueva, la posición actual de lara y quite de la vista el techo de la cueva


TODO:
- Debes terminar la creación de la cueva básica, falta crear los gpuShape, como leerá las texturas y otros


# Eliminamos los puntos con altura similar
    for i in range(1,lenXS-1):
        for j in range(1, lenYS-1):
            k = index(i, j)

            neighbour = list(sueloMesh.vv(vertexsuelo[k]))
            if len(neighbour)==8:   # No es un borde
                point = list(sueloMesh.point(vertexsuelo[k]))
                conservar = False
                for vh in neighbour:
                    pointvh = list(sueloMesh.point(vh))
                    if point[2] != pointvh[2]:
                        conservar = True
                        break
                if not conservar:
                    sueloMesh.delete_vertex(vertexsuelo[k])
    sueloMesh.garbage_collection()

    for i in range(1,lenXS-1):
        for j in range(1, lenYS-1):
            k = index(i, j)
            neighbour = list(techoMesh.vv(vertextecho[k]))
            if len(neighbour)==8:   # No es un borde
                point = list(techoMesh.point(vertextecho[k]))
                conservar = False
                for vh in neighbour:
                    pointvh = list(techoMesh.point(vh))
                    if point[2] != pointvh[2]:
                        conservar = True
                        break
                if not conservar:
                    techoMesh.delete_vertex(vertextecho[k])
    techoMesh.garbage_collection()
