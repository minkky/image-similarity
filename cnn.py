import keras
import random, os
from keras.layers import Activation, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten
from keras.models import Model, Sequential
from keras import backend as K
from glob import glob
from PIL import Image
import numpy as np
import re
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


batch_size = 128
num_classes = 6
epochs = 500

img_rows, img_cols = 28, 28
label = ["circle", "square", "triangle", "horizontal", "vertical", "diagonal"]

def get_datasets_labels(path):
	files = os.listdir(path)
	labels = []
	sets = []
	for file in files:
		pic = Image.open(path + file)
		lb = file.replace(re.findall("\d+", file)[0], '').replace('.png', '')
		labels.append(label.index(lb))
		pic = np.array(pic)
		sets.append(pic)
	return sets, labels

def load_data():
    tr_path = 'trains/'
    te_path = 'tests/'

    (tr_sets, tr_labels) = get_datasets_labels(tr_path)
    (te_sets, te_labels) = get_datasets_labels(te_path)

    tr_sets = np.array(tr_sets)
    te_sets = np.array(te_sets)
    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    return tr_sets, tr_labels, te_sets, te_labels

(x_train, y_train, x_test, y_test) = load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))