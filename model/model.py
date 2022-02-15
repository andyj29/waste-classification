from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import Generator, Utils

train_data_flow = Generator.generate_train_data_flow()
test_data_flow = Generator.generate_test_data_flow()

labels = (train_data_flow.class_indices)
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

model.fit(train_data_flow, epochs=1000, validation_data=test_data_flow, callbacks=[checkpoint,es])

test_x, test_y = test_data_flow.__getitem__(1)

preds = model.predict(test_x)

Utils.plot_results((16,16), 16, labels, preds, test_x, test_y)