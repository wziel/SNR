from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from sklearn import preprocessing
import os #foldery i pliki
import re #regular expressions
import numpy as np #biblioteka numeryczna, tablice wielowymiarowe i utils
from sklearn import datasets, svm, metrics
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from keras.models import Model

root_dir = os.path.join('D:\\', 'DYSK', 'SNR') # ścieżka do katalogu głównego gdzie opisane są boundingbox. Należy zmienić w zależności od środowiska uruchomienia.
train_img_dir = os.path.join(root_dir, 'train') # ścieżka do katalogu głównej z folderami ptaków zbioru trenującego
test_img_dir = os.path.join(root_dir, 'test') # ścieżka do katalogu głównej z folderami ptaków zbioru testującego
network_input_size = 192
conv_file_name = os.path.join(root_dir, "conv_final_2.best.43-0.3800.hdf5") # plik w którym serializowane są wagi po procesie uczenia
history_file_name = os.path.join(root_dir, "conv_final.history.p") # plik w którym serializowana jest historia nauki

model = keras.models.load_model(conv_file_name)
layer_name = 'dense_1'
to_svm_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


def load_data(model, img_dir, img_size):
	X_set = []
	Y_set = []
	for root, subdirs, files in os.walk(img_dir):	
		for directory in subdirs:
			class_name = directory
			for root, subdirs, files in os.walk(os.path.join(img_dir, directory)):
				for file in files:
					file_path = os.path.join(img_dir, directory, file)
					img = Image.open(file_path)
					img = img.resize((img_size,img_size))
					rgb = Image.new("RGB", img.size, (255, 255, 255))
					rgb.paste(img)
					
					arr = img_to_array(rgb)
					image_features =  model.predict(np.array([arr]))
					image_features = image_features.flatten()
					image_features = np.array(image_features.tolist())
					X_set.append(image_features)
					Y_set.append(directory)
	Y_set = np.array(Y_set)
	Y_set = Y_set.astype(np.int32)
	X_set = np.array(X_set)
	X_set = preprocessing.scale(X_set)
	return	X_set, Y_set

X_train, y_train = load_data(to_svm_layer_model, train_img_dir, 192)
X_test, y_test = load_data(to_svm_layer_model, test_img_dir, 192)

C_parameters =[0.1,0.2,0.5,1,2,5,10,20,50,100,200,500]
print('SVM - polynominal kernel - tests')
for x in C_parameters:
    print(x)
    clf = svm.SVC(kernel='poly', degree=2, C=x)
    clf.fit(X_train, y_train)
    print("Train: {0:.4f}".format(clf.score(X_train, y_train)))
    print("Test: {0:.4f}".format(clf.score(X_test, y_test)))
	
print('SVM - rbf kernel - tests')
for x in C_parameters:
    print(x)
    clf = svm.SVC(C=x)
    clf.fit(X_train, y_train)
    print("Train: {0:.4f}".format(clf.score(X_train, y_train)))
    print("Test: {0:.4f}".format(clf.score(X_test, y_test)))
