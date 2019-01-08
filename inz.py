import os
import cv2
from glob import glob
from scipy import misc
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras import backend as K

data = glob('database/*')
len(data)

images = []

class_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fingerprints_number = 8

with open('fingerprints_classes.txt', 'w') as file:
    counter = 1
    file.write("Class\n")
    while counter <= fingerprints_number:
        for klasa in class_number:
                file.write("%i\n" % counter)
        counter += 1


def read_images(data, file_class_name):
    readed_classes = pd.read_csv(file_class_name, sep="\t")

    for i in range(79):
        img = misc.imread(data[i])
        img = misc.imresize(img, (224, 224))  # 480 x 640
        images.append(img)
    if len(file_class_name) > 0:
        return np.asarray(images), readed_classes["Class"].values
    else:
        return np.asarray(images)


images, y = read_images(data, "fingerprints_classes.txt")

train_set_x = data[:60]
val_set_x = data[60:]
train_set_y = y[:60]
val_set_y = y[60:]

(X_train, y_train), (X_test, y_test) = (train_set_x, train_set_y), (val_set_x, val_set_y)

# input image dimensions
img_rows, img_cols = 224, 224
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# Checking if the backend is Theano or Tensorflow
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


#images = read_images(data)
images_arr = np.asarray(images)
# images_arr = images_arr.astype('float32')

# images_arr.shape


# Display the first image in training data
# for i in range(2):
#     plt.figure(figsize=[5, 5])
#     curr_img = np.reshape(images_arr[i], (224, 224))
#     plt.imshow(curr_img, cmap='gray')
#     plt.show()
#
# print("Dataset (images) shape: {shape}".format(shape=images_arr.shape))


# data preprocessing
# konwersja każdego obrazu o wymiarach 244x244 do macierzy 244 x 244 x 1
images_arr = images_arr.reshape(-1, 224, 224, 1)
images_arr.shape

# przeskalowanie
images_arr.dtype
np.max(images_arr)
images_arr = images_arr / np.max(images_arr)

# weryfikacja
np.max(images_arr)  # 1
np.min(images_arr)  # 0







##################################################################################
from keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator()
image_gen.flow_from_directory('database/train')
input_shape= (150, 150, 3)
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense


model = Sequential()

# kernel size and  filters shoot!! ;)

model.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

# helps reduce overfitting by randomly turning neurons off during training (wybieram 50%)
model.add(Dropout(0.5))

model.add(Dense(10))  # bo 10 klas
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# must define batch size ( good start is 16)
batch_size = 16
train_image_gen = image_gen.flow_from_directory('database/train', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')

test_image_gen = image_gen.flow_from_directory('database/test', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')


train_image_gen.class_indices

# epoka to przejscie przez caly zbior treningowy        # steps per epoch to cos w rodzaju kroku - jesli nie damy limitu to bedzie sie dlugo trenowal
# validation steps ????
results = model.fit_generator(train_image_gen, epochs=1, steps_per_epoch=150, validation_data=test_image_gen, validation_steps=12)



