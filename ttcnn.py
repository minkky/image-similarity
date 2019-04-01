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
import matplotlib as mpl
import matplotlib.pylab as plt
from pylab import figure, axes, pie, title, savefig

batch_size = 128
num_classes = 6
epochs = 1000

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
    tr_path = 'dataset/trains/'
    te_path = 'dataset/tests/'
    va_path = 'dataset/vals/'

    (tr_sets, tr_labels) = get_datasets_labels(tr_path)
    (te_sets, te_labels) = get_datasets_labels(te_path)
    (va_sets, va_labels) = get_datasets_labels(va_path)

    tr_sets = np.array(tr_sets)
    te_sets = np.array(te_sets)
    va_sets = np.array(va_sets)
    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    va_labels = np.array(va_labels)
    return tr_sets, tr_labels, te_sets, te_labels, va_sets, va_labels

(x_train, y_train, x_test, y_test, x_val, y_val) = load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255.
x_test /= 255.
x_val /= 255.

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding='same'))
encoder = model

# decoder
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(1, (3, 3), padding='same'))
model.add(Activation('sigmoid'))


model.compile(optimizer='adadelta', loss='binary_crossentropy')#, metrics=['accuracy'])
model.fit(x_train, x_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, validation_data=(x_val, x_val))

decoded_imgs = model.predict(x_test)

encoder_to_json = encoder.to_json()
model_to_json = model.to_json()
with open("encoder.json", "w") as json_file:
	json_file.write(encoder_to_json)

with open("model.json", "w") as json_file:
	json_file.write(model_to_json)

encoder.save_weights("encoder.h5")
model.save_weights("model.h5")

model.summary()
#print("weight : " + encoder.get_weights())
'''
input_img = Input(shape=(28, 28, 1)) 
x = Conv2D(32, (3, 3), padding='same')(input_img)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)
model.add(Dropout(0.25))

x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)



x = MaxPooling2D(2, 2)(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
x = Dense(num_classes)(x)

encoded = MaxPooling2D(2, 2)(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(16, (3, 3), activation='relu')(x)
#x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')#, metrics=['accuracy'])

print(x_train.shape)

autoencoder.fit(x_train, x_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, validation_data=(x_test, x_test))

#score = model.evaluate(x_test, x_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

decoded_imgs = autoencoder.predict(x_test)
'''
m = 10
n = 30
plt.figure(figsize=(60, 40))
for j in range(0, m):
	for i in range(0, n):
	    # display original
	    fi = ax = plt.subplot(2*m, n, 60*j+i+1)
	    plt.imshow(x_test[n*(j)+i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)

	    # display reconstruction
	    fi = ax = plt.subplot(2*m, n, 60*j+30+i+1)
	    plt.imshow(decoded_imgs[n*(j)+i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
plt.show()
