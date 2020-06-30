import config as cfg
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

class predict_emotions():
    def __init__(self):
        # cargo modelo de deteccion de emociones
        self.model = load_model(cfg.path_model)

    def preprocess_img(self,face_image,rgb=True,w=48,h=48):
        face_image = cv2.resize(face_image, (w,h))
        if rgb == False:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = face_image.astype("float") / 255.0
        face_image= img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        return face_image

    def get_emotion(self,img,boxes_face):
        emotions = []
        if len(boxes_face)!=0:
            for box in boxes_face:
                y0,x0,y1,x1 = box
                face_image = img[x0:x1,y0:y1]
                # preprocesar data
                face_image = self.preprocess_img(face_image ,cfg.rgb, cfg.w, cfg.h)
                # predecir imagen
                prediction = self.model.predict(face_image)
                emotion = cfg.labels[prediction.argmax()]
                emotions.append(emotion)
        else:
            emotions = []
            boxes_face = []
        return boxes_face,emotions

