import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from geopy.geocoders import Nominatim
from src.config.__init__ import model
from src.config.settings import BASE_DIR
from .constants import Label, Area


labels = Label.to_list()

areas = Area.to_list()

geolocator = Nominatim(user_agent="geoapiExercises")


def load_pic(path):
    file_path = os.path.join(BASE_DIR, ''.join(path.split('/', 1)))
    loaded_image = image.load_img(file_path, target_size=(300,300))
    matrix = image.img_to_array(loaded_image)/255
    matrix = np.reshape(matrix,(1, 300, 300, 3))

    return matrix


def classify_image(img_path):
    data = load_pic(img_path)
    prediction = model.predict(data)
    label = labels[np.argmax(prediction)]
    
    probabilities = prediction[0][np.argmax(prediction)]
    response = {
        'label': label,
        'prob': probabilities
    }

    return response


def get_area_from_lat_long(lat, long):
    location = geolocator.reverse(lat+","+long)
    address = location.raw['address']
    area = address['city']

    return area


def is_in_gta(area):
    return area in areas


