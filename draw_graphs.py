from __future__ import print_function
import keras
import os #foldery i pliki
import re #regular expressions
import cv2 #opencv
import random # random
import time # czas
import pickle # serializacja obiektów do pliku
import numpy as np #biblioteka numeryczna, tablice wielowymiarowe i utils
import matplotlib.pyplot as plt #rysowanie wykresów
from PIL import Image

root_dir = os.path.join('C:\\', 'Informatyka', 'SNR', 'conv') # ścieżka do katalogu głównego gdzie opisane są boundingbox. Należy zmienić w zależności od środowiska uruchomienia.
history_file_name = os.path.join(root_dir, "conv_final.history.p") # plik w którym serializowana jest historia nauki # xception.history.p
history_50 = pickle.load( open( history_file_name, "rb" ) )
cat_acc=os.path.join(root_dir, "1categorical_accuracy.png")
top_acc=os.path.join(root_dir, "1top_k_categorical_accuracy.png")
loss=os.path.join(root_dir, "1loss.png")
# dostępne metryki print(history.keys()) dict_keys(['val_loss', 'val_categorical_accuracy', 'val_top_k_categorical_accuracy', 'loss', 'categorical_accuracy', 'top_k_categorical_accuracy'])
# summarize history for categorical_accuracy
plt.figure(figsize=(8, 4))
plt.plot(history_50['categorical_accuracy'], color = 'blue')
plt.plot(history_50['val_categorical_accuracy'], color = 'green')
plt.plot(history_50['top_k_categorical_accuracy'], color = 'blue', linestyle=':')
plt.plot(history_50['val_top_k_categorical_accuracy'], color = 'green', linestyle=':')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc','train_top5_acc', 'test_top5_acc'], loc='upper left')
plt.savefig(cat_acc)
plt.show()
# summarize history for loss
plt.figure(figsize=(8, 4))
plt.plot(history_50['loss'], color = 'red')
plt.plot(history_50['val_loss'], color = 'red', linestyle=':')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'], loc='upper left')
plt.savefig(loss)
plt.show()


'''
max(history_50['val_top_k_categorical_accuracy'])
max(history_50['val_categorical_accuracy'])
history_50['categorical_accuracy'] = history_50['categorical_accuracy'][:20]
history_50['val_categorical_accuracy'] = history_50['val_categorical_accuracy'][:20]
history_50['top_k_categorical_accuracy'] = history_50['top_k_categorical_accuracy'][:20]
history_50['val_top_k_categorical_accuracy'] = history_50['val_top_k_categorical_accuracy'][:20]
history_50['loss'] = history_50['loss'][:20]
history_50['val_loss'] = history_50['val_loss'][:20]
'''
