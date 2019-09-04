import matplotlib

matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# Definicion de la estrucutra de los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


def mezclar_orden_imagenes(directorioDataset):
    listaRutaImagenes = list(paths.list_images(directorioDataset))
    listaRutaImagenesOrdenada = sorted(listaRutaImagenes)
    random.seed(42)
    random.shuffle(listaRutaImagenesOrdenada)
    return listaRutaImagenesOrdenada


def preparar_entremaiento(listaRutaImagenes, dimensionImagen, etiquetaReferencia):
    datos_entremamiento = []
    lista_etiquetas = []
    for rutaImagen in listaRutaImagenes:
        # Cargamos la imagen y la preprocesamos
        imagen = cv2.imread(rutaImagen)
        imagen = cv2.resize(imagen, (dimensionImagen, dimensionImagen))
        imagen = img_to_array(imagen)  # Convertimos la imagen a un arreglo Numpy
        datos_entremamiento.append(imagen)

        nombreEtiqueta = rutaImagen.split(os.path.sep)[-2]
        nombreEtiqueta = 1 if nombreEtiqueta == etiquetaReferencia else 0
        lista_etiquetas.append(nombreEtiqueta)

    # Escalamos cada pixel
    datos_entremamiento = np.array(datos_entremamiento, dtype="float") / 255.0
    # Convertimos la lista de etiquetas a un arreglo numpy
    lista_etiquetas = np.array(lista_etiquetas)
    return datos_entremamiento, lista_etiquetas


ITERACIONES = 50  # numero de iteraciones
TASA_APRENDIZAJE = 1e-3  # Tasa de aprendizaje
BATCH_SIZE = 32  # Batch size

lista_rutas_imagenes_entremamiento = mezclar_orden_imagenes(args["dataset"])

datos_a_entrenar, etiquetas = preparar_entremaiento(lista_rutas_imagenes_entremamiento, 28, 'lleno')

# Particinar los datos en datos para entranar 75% y datos para pruebas 25%
(trainX, testX, trainY, testY) = train_test_split(datos_a_entrenar, etiquetas, test_size=0.25, random_state=42)

# Convertimos las etiquetas de enteros a vectores
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# Configuramos el generador de datos, esto se hace para aumentar la cantidad de datos de entramiento
aumentador = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                horizontal_flip=True, fill_mode="nearest")

# Inicializamos el modelo
print("Iniciando modelo...")
model = LeNet.build(width=28, height=28, depth=3, classes=2)
optimizador = Adam(lr=TASA_APRENDIZAJE, decay=TASA_APRENDIZAJE / ITERACIONES)
model.compile(loss="binary_crossentropy", optimizer=optimizador, metrics=["accuracy"])

# Entrenar la red neural
print("Entrenando red...")
resultadosHistoricos = model.fit_generator(aumentador.flow(trainX, trainY, batch_size=BATCH_SIZE),
                                           validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
                                           epochs=ITERACIONES, verbose=1)

# Guardar la red neural entrenada en el disco
print("Guardando red neural...")
model.save(args["model"])

# Graficamos la precision y la perdida que hubo durante el entrenamiento

plt.style.use("ggplot")
plt.figure()
N = ITERACIONES
plt.plot(np.arange(0, N), resultadosHistoricos.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), resultadosHistoricos.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), resultadosHistoricos.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), resultadosHistoricos.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on bus lleno/bus no lleno")
plt.xlabel("Iteracion #")
plt.ylabel("Perdida/Precision")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
