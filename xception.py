from __future__ import print_function 
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications.xception import Xception
import os #foldery i pliki
import re #regular expressions
import cv2 #opencv
import random # random
import time # czas
import pickle # serializacja obiektów do pliku
import numpy as np #biblioteka numeryczna, tablice wielowymiarowe i utils
import matplotlib.pyplot as plt #rysowanie wykresów
from sklearn.cluster import KMeans # algorytm k średnich
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

root_dir = os.path.join('C:\\', 'Users', 'Wojciech', 'Desktop', 'SNR') # ścieżka do katalogu głównego gdzie opisane są boundingbox. Należy zmienić w zależności od środowiska uruchomienia.
train_img_dir = os.path.join(root_dir, 'train') # ścieżka do katalogu głównej z folderami ptaków zbioru trenującego
test_img_dir = os.path.join(root_dir, 'test') # ścieżka do katalogu głównej z folderami ptaków zbioru testującego
xception_file_name = os.path.join(root_dir, "xception.best.{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5") # plik w którym serializowane są wagi xception po procesie uczenia
history_file_name = os.path.join(root_dir, "xception.history.p") # plik w którym serializowana jest historia nauki xception
network_input_size = 192
batch_size = 32
epochs = 10000
num_classes = 50

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_img_dir,  # this is the target directory
        target_size=(network_input_size, network_input_size),  # all images will be resized to this size
        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        test_img_dir,
        target_size=(network_input_size, network_input_size),
        batch_size=batch_size)

model = Xception(weights = None, classes = num_classes)

model.compile(loss='categorical_crossentropy',
            optimizer=RMSprop(),
            metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=0, mode='max')
mcp_save = ModelCheckpoint(xception_file_name, monitor='val_categorical_accuracy', save_best_only=True, mode='max')

history = model.fit_generator(
        train_generator,
        epochs=epochs,
        verbose=1,
        validation_data=validation_generator,
        callbacks=[early_stop, mcp_save])
pickle.dump(history.history, open(history_file_name, "wb"))