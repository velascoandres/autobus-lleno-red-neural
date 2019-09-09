from builtins import ord
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2


MODEL_PATH = "red_v3.h5"

# cargar el modelo
print("Cargando la red neural...")
model = load_model(MODEL_PATH)
# inicializar erl video
print("Cargando Video")
video = cv2.VideoCapture('videos_prueba/video1.mp4')

while True:
    q, cuadro = video.read()
    # preparamos la imagen para que pueda ser compatible con la red neural
    imagen = cv2.resize(cuadro, (28, 28))
    imagen = imagen.astype("float") / 255.0
    imagen = img_to_array(imagen)
    imagen = np.expand_dims(imagen, axis=0)
    # clasificamos la imagen obtenemos las probabilidades
    (probabilidadVacio, probabilidadLleno) = model.predict(imagen)[0]
    etiqueta = "No lleno"
    probabilidad = probabilidadVacio
    # etiquetamos la imagen conforme a la prediccion mas alta
    if probabilidadLleno > probabilidadVacio:
        probabilidad = probabilidadLleno
        etiqueta = "Lleno"

    # armamos la etiqueta con su respectiva probabilidad y la ponemos en la imagen
    etiqueta = "{}: {:.2f}%".format(etiqueta, probabilidad * 100)
    cuadro = cv2.putText(cuadro, etiqueta, (10, 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # mostrar la imagen
    cv2.imshow("Video", cuadro)
    teclaPresionada = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if teclaPresionada == ord("q"):
        break

cv2.destroyAllWindows()
video.stop()

