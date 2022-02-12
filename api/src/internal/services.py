import numpy as np
from keras.preprocessing import image
from geopy.geocoders import Nominatim
from src.config.__init__ import model
from src.config.settings import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
from PIL import Image
from .constants import Label, Area
import io
import boto3

s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,region_name='ca-central-1')
bucket = s3.Bucket('2022hacks')


labels = Label.to_list()

areas = Area.to_list()

geolocator = Nominatim(user_agent="geoapiExercises")


def load_pic(path):
    s3_path = path.split('.com/')
    print(s3_path[1])
    print(s3_path)
    obj = bucket.Object(s3_path[1])
    print(obj)
    file_stream = io.BytesIO()
    obj.download_fileobj(file_stream)
    img = Image.open(file_stream)
    
    resized_img = np.resize(img,(300,300,3))
    print("HELLO")
    matrix = image.img_to_array(resized_img)/255
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


