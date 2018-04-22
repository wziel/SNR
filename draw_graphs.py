from __future__ import print_function 
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
history_file_name = os.path.join(root_dir, "perceptron.history.p") # plik w którym serializowana jest historia nauki # xception.history.p
history = pickle.load( open( history_file_name, "rb" ) )
# dostępne metryki print(history.keys()) dict_keys(['val_loss', 'val_categorical_accuracy', 'val_top_k_categorical_accuracy', 'loss', 'categorical_accuracy', 'top_k_categorical_accuracy'])
# summarize history for categorical_accuracy
plt.plot(history['categorical_accuracy'])
plt.plot(history['val_categorical_accuracy'])
plt.title('model categorical accuracy')
plt.ylabel('categorical accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for acc top_k_categorical_accuracy
plt.plot(history['top_k_categorical_accuracy'])
plt.plot(history['val_top_k_categorical_accuracy'])
plt.title('model top 5 categorical accuracy')
plt.ylabel('top 5 categorical accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()