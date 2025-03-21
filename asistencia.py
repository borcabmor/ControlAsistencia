import cv2
import face_recognition as fr
import os

lista_empleados = []
nombres_empleados = []

for imagen in os.listdir('Empleados'):
    foto_cargada = cv2.imread('Empleados/' + imagen)

    # pasar foto a RGB
    foto_cargada = cv2.cvtColor(foto_cargada, cv2.COLOR_BGR2RGB)

    # localizar cara en foto
    lista_empleados.append(fr.face_encodings(foto_cargada)[0])

    #introducir nombre empleado
    nombres_empleados.append(os.path.splitext(imagen)[0])

print(nombres_empleados)

#capturar imágen de cámara web
captura = cv2.VideoCapture(0)

#leer imágen de la cámara
exito, imagen_captura = captura.read()

if not exito:
    print('La captura no ha sido correcta')
else:
    lugar_cara_prueba = fr.face_locations(imagen_captura)
    cara_codificada = fr.face_encodings(imagen_captura, lugar_cara_prueba)

    #busca coincidencias
    encontrado = False

    for caracodif, caraubic in zip(cara_codificada, lugar_cara_prueba):
        coincidencias = fr.compare_faces(lista_empleados, caracodif)
        distancias = fr.face_distance(lista_empleados, caracodif)

        iterador = 0

        for comparacion in coincidencias:
            if comparacion:
                print(f'Usted es {nombres_empleados[iterador]}')
                encontrado = True

                # Mostrar imágen
                cv2.imshow('Imágen Cam', imagen_captura)

                break

            iterador += 1

    if not encontrado:
        print('No es un trabajador de la empresa')

    cv2.waitKey(0)