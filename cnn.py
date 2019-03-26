import keras
import random, os
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten
from keras.models import Model, Sequential
from keras import backend as K
from glob import glob
from PIL import Image
import numpy as np
import re

batch_size = 128
num_classes = 6
epochs = 1000

img_rows, img_cols = 28, 28
label = ["circle", "square", "triangle", "horizontal", "vertical", "diagonal"]

def load_data():
    path = 'trains/'
    dataset = []
    filenames = os.listdir(path)
    index = 0
    labels = []
    for file in filenames:
        pic = Image.open(path + file)
        lb = file.replace(re.findall("\d+", file)[0], '').replace('.png', '')
        labels.append(label.index(lb))
        pic = np.array(pic)
        dataset.append(pic)
    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels

(x_train, y_train) = load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_train /= 255

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)