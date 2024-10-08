from PIL import Image
import base64
from io import BytesIO
import json
import random
import cv2
import numpy as np
from deepface import DeepFace
from keras.models import load_model
from keras.preprocessing import image

model = load_model("model.h5")  # load your model trained on your dataset from your system 

face_cas = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")  # load haarcascade_frontalface_default.xml file form your system as given in git hub


def face_fun(img):
    faces = face_cas.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        crop_face = img[y:y+h, x:x+w]

    return crop_face


font = cv2.FONT_HERSHEY_COMPLEX

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    _, frame = video_capture.read()

    if frame is None:
        print("Error: Could not read frame from the video capture.")
        break

    face = face_fun(frame)
    if isinstance(face, np.ndarray):
        try:
            result = DeepFace.analyze(
                face, actions=["emotion"], enforce_detection=False)
            print(result)

            emotion = result[0]["dominant_emotion"]
            if result is not None and "dominant_emotion" in result[0]:
                emotion = result[0]["dominant_emotion"]
            else:
                emotion = "No Emotion found"
        except Exception as e:
            print("An error occurred during emotion analysis:", str(e))
            emotion = "Error"

        face = cv2.resize(face, (150, 150))
        im = Image.fromarray(face, "RGB")
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)

        name = "None Match"

        if pred[0][0] == 1:  # Adjust the threshold as per your model's requirement
            name = "Hitesh"
        elif pred[0][0] == 0 :
            name = "Aditya"
        else:
            name = "None Match"

        cv2.putText(frame, f"Emotion: {emotion}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, f"Name: {name}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()