import time
import cv2
import imutils
from kafka import KafkaProducer
  
def publish_camera(producer,topic):
    camera = cv2.VideoCapture(0)
    try:
        print('Sending frame...')
        while(True):
            success, frame = camera.read()
            if not success:
                break
            producer.send(topic, value=frame)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
            time.sleep(0.5)
    except:
        print('Exiting...')
    camera.release()

def encoding_img(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    im_bytes = buffer.tobytes()
    return im_bytes


producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: encoding_img(v)
)
publish_camera(producer,'video')





