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

root_dir = os.path.join('C:\\', 'Informatyka', 'SNR') # ścieżka do katalogu głównego gdzie opisane są boundingbox. Należy zmienić w zależności od środowiska uruchomienia.
history_file_name = os.path.join(root_dir, "perceptron_50.history.p") # plik w którym serializowana jest historia nauki # xception.history.p
history_50 = pickle.load( open( history_file_name, "rb" ) )
history_file_name = os.path.join(root_dir, "perceptron_100.history.p") # plik w którym serializowana jest historia nauki # xception.history.p
history_100 = pickle.load( open( history_file_name, "rb" ) )
history_file_name = os.path.join(root_dir, "perceptron_200.history.p") # plik w którym serializowana jest historia nauki # xception.history.p
history_200 = pickle.load( open( history_file_name, "rb" ) )
cat_acc=os.path.join(root_dir, "categorical_accuracy.png")
top_acc=os.path.join(root_dir, "top_k_categorical_accuracy.png")
loss=os.path.join(root_dir, "loss.png")
# dostępne metryki print(history.keys()) dict_keys(['val_loss', 'val_categorical_accuracy', 'val_top_k_categorical_accuracy', 'loss', 'categorical_accuracy', 'top_k_categorical_accuracy'])
# summarize history for categorical_accuracy
plt.plot(history_50['categorical_accuracy'], color = 'blue')
plt.plot(history_50['val_categorical_accuracy'], color = 'blue')
plt.plot(history_100['categorical_accuracy'], color = 'red')
plt.plot(history_100['val_categorical_accuracy'], color = 'red')
plt.plot(history_200['categorical_accuracy'], color = 'green')
plt.plot(history_200['val_categorical_accuracy'], color = 'green')
plt.title('model categorical accuracy')
plt.ylabel('categorical accuracy')
plt.xlabel('epoch')
plt.legend(['train_50', 'test_50','train_100', 'test_100', 'train_200', 'test_200'], loc='upper left')
plt.savefig(cat_acc)
plt.show()
# summarize history for acc top_k_categorical_accuracy
plt.plot(history_50['top_k_categorical_accuracy'], color = 'blue')
plt.plot(history_50['val_top_k_categorical_accuracy'], color = 'blue')
plt.plot(history_100['top_k_categorical_accuracy'], color = 'red')
plt.plot(history_100['val_top_k_categorical_accuracy'], color = 'red')
plt.plot(history_200['top_k_categorical_accuracy'], color = 'green')
plt.plot(history_200['val_top_k_categorical_accuracy'], color = 'green')
plt.title('model top 5 categorical accuracy')
plt.ylabel('top 5 categorical accuracy')
plt.xlabel('epoch')
plt.legend(['train_50', 'test_50','train_100', 'test_100', 'train_200', 'test_200'], loc='upper left')
plt.savefig(top_acc)
plt.show()
# summarize history for loss
plt.plot(history_50['loss'], color = 'blue')
plt.plot(history_50['val_loss'], color = 'blue')
plt.plot(history_100['loss'], color = 'red')
plt.plot(history_100['val_loss'], color = 'red')
plt.plot(history_200['loss'], color = 'green')
plt.plot(history_200['val_loss'], color = 'green')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_50', 'test_50','train_100', 'test_100', 'train_200', 'test_200'], loc='lower left')
plt.savefig(loss)
plt.show()
