# %% 
import cv2
import json
import numpy as np
from pyspark.streaming.kafka import KafkaUtils
from pyspark.streaming import StreamingContext
from pyspark import SparkContext, SparkConf
import EmotionDetection
from kafka import KafkaProducer

def deserializer(im_byte):
    decoByte = np.frombuffer(im_byte, dtype=np.uint8)
    decoJpg = cv2.imdecode(decoByte, cv2.IMREAD_COLOR)
    return decoJpg

detector = EmotionDetection.ClassficateEmotion()
def get_emotions(m):
    try: 
        img = m[1]
        emotion  = detector.GetEmotion(img)
        return emotion
    except:
        return 'error in get_emotions'


def message_sender(m):
    my_producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    my_producer.send('result',m)
    return m


conf = SparkConf().setMaster('local[2]').setAppName('video')
sc = SparkContext.getOrCreate(conf=conf)

n_secs = 1
ssc = StreamingContext(sparkContext=sc, batchDuration=n_secs)

kafkaParams = {'bootstrap.servers':'localhost:9092'}
stream = KafkaUtils.createDirectStream(ssc=ssc,
                topics=['video'],
                kafkaParams=kafkaParams,
                valueDecoder=lambda v: deserializer(v))  
                                 
stream.map(
        get_emotions
    ).map(
        message_sender
    )
    
ssc.start()
ssc.awaitTermination(timeout=100000)

# %%
