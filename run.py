import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import os

def get_model():
    model = tf.keras.models.load_model('my_model.h5')
    print(model.summary())
    return model

def make_prediction(image_path, model):
    image_string = tf.io.read_file(os.path.join(os.getcwd(),image_path))
    image_decoder = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image_decoder, (150,150))
    image = tf.reshape(image, (1,150,150,3))
    pred_int = model.predict_classes(image)
    return pred_int
    
    

if __name__ == "__main__":
    model = get_model()
    print(make_prediction(sys.argv[1],model))
