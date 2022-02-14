import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from IPython.display import display


base_path = r'/Users/phongnguyen/Documents/2022-hackathon/dataset'

img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))

print(len(img_list))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)

train_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    seed=0
)

validation_generator = test_datagen.flow_from_directory(
    base_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=0
)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

print(labels)
model = Sequential([
    Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(224, 224, 3)),

    BatchNormalization(),

    MaxPooling2D(pool_size=(2,2), strides=(2,2)),

    BatchNormalization(),

    Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),

    BatchNormalization(),

    MaxPooling2D(pool_size=(2,2), strides=(2,2)),

    BatchNormalization(),

    Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),

    BatchNormalization(),

    Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),

    BatchNormalization(),

    MaxPooling2D(pool_size=(2,2), strides=(2,2)),

    BatchNormalization(),

    Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'),

    BatchNormalization(),

    Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'),

    BatchNormalization(),

    MaxPooling2D(pool_size=(2,2), strides=(2,2)),

    BatchNormalization(),

    Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'),

    BatchNormalization(),

    Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'),

    BatchNormalization(),

    MaxPooling2D(pool_size=(2,2), strides=(2,2)),

    BatchNormalization(),

    Flatten(),

    Dense(4096, activation='relu'),

    Dropout(0.3),
    BatchNormalization(),

    Dense(4096, activation='relu'),

    Dropout(0.3),
    BatchNormalization(),

    Dense(12, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=20)

checkpoint = ModelCheckpoint('saved_model.h5', monitor='val_acc', mode='max', save_best_only=True, save_weights_only=False)

model.fit(train_generator, epochs=1000, validation_data=validation_generator, callbacks=[checkpoint,es])

test_x, test_y = validation_generator.__getitem__(1)

preds = model.predict(test_x)

plt.figure(figsize=(16, 16))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])
    plt.show()

predicted = []
actual = []
for i in range(16):
    predicted.append(labels[np.argmax(preds[i])])
    actual.append(labels[np.argmax(test_y[i])])

df = pd.DataFrame(predicted, columns=["predicted"])
df["actual"] = actual
display(df)

