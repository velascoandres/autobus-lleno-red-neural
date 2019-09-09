from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import os


MODEL_PATH = "red_v3.h5"

SANTA = False
# load the model
print("Cargando la red neural...")
model = load_model(MODEL_PATH)
# initialize the video stream and allow the camera sensor to warm up
print("Cargando Video")
video = cv2.VideoCapture('videos_prueba/video3.mp4')

while True:
    q, frame = video.read()
    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # classify the input image and initialize the label and
    # probability of the prediction
    (vacio, lleno) = model.predict(image)[0]
    print(model.predict(image))
    label = "No lleno"
    proba = vacio
    if lleno > vacio:
        proba = lleno
        label = "Lleno"

    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Cuadro", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video.stop()

