from tensorflow.python.keras.utils import np_utils
import keras
import numpy as np
import tensorflow as tf
from os import path, listdir
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow. keras.layers import Input,GlobalMaxPooling2D,Dense
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from PIL import Image
model = load_model("soil_typemodel.h5")
label_map = {'Black Soil': 0,
 'Cinder Soil': 1,
 'Laterite Soil': 2,
 'Peat Soil': 3,
 'Yellow Soil': 4}
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image and converts  into an object 
        that can be used as input to a trained model, returns an Numpy array.

        Arguments
        ---------
        image_path: string, path of the image.
    '''
    
    im = Image.open(image_path)
    im=im.resize((224, 224))
    im =img_to_array(im)
    im = np.expand_dims(im, axis=0)
    im= preprocess_input(im)
    
    return im
def get_key(val): 
    for key, value in label_map.items(): 
         if val == value: 
             return key 
def process(predict):
    print("Path==22===",path)
    #image = process_image(path)
    #print("Path==",path)

    prediction = model.predict(predict)
    print("prediction",prediction)
    
    return get_key(np.argmax(prediction))
#print("Predicted=",process("./FinalDataset/Trigloporus lastoviza/trigloporus_lastoviza-1704.jpg"))
    
