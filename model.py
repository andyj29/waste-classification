import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

import glob, os, random

base_path = r'/Users/phongnguyen/Documents/waste-classifier/dataset'

img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))

