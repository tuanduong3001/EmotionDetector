import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
classifier = load_model('models/model_emotions.h5')

class ClassficateEmotion():
    def GetEmotion(self, img):
        emotions = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                preds = classifier.predict(roi)[0]
                emotion = class_labels[preds.argmax()]
                emotions.append(emotion)
            else:
                emotions.append('Unknown')                                    
        return emotions

# ahihi = ClassficateEmotion()
# img = cv2.imread('images/ahihi.jpg')
# label = ahihi.GetEmotion(img)
# print(label)