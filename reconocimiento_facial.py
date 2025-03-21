import cv2
import face_recognition as fr

#cargar imágenes
foto_control = fr.load_image_file('BorjaJR.jpg')
foto_prueba = fr.load_image_file('FotoC.jpg')

#pasar imágenes a RGB
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

#localizar cara en foto_control
lugar_cara_A = fr.face_locations(foto_control)[0]
cara_codificada_A = fr.face_encodings(foto_control)[0]

#mostrar rectángulo en cara A
cv2.rectangle(foto_control, (lugar_cara_A[3], lugar_cara_A[0]), (lugar_cara_A[1], lugar_cara_A[2]), (0, 255, 0), 2)

#localizar cara en foto_prueba
lugar_cara_B = fr.face_locations(foto_prueba)[0]
cara_codificada_B = fr.face_encodings(foto_prueba)[0]

#mostrar rectángulo en cara B
cv2.rectangle(foto_prueba, (lugar_cara_B[3], lugar_cara_B[0]), (lugar_cara_B[1], lugar_cara_B[2]), (0, 255, 0), 2)

#realizar comparación
resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B)

print(resultado[0])

#medida de la distancia
distancia = fr.face_distance([cara_codificada_A], cara_codificada_B)

print(distancia)

#mostrar distancia entre imágenes
cv2.putText(foto_prueba, f'{resultado[0]} {distancia.round(2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

#Mostrar imágenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)

#mantener programa abierto
cv2.waitKey(0)

