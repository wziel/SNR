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
train_img_dir = os.path.join(root_dir, 'train') # ścieżka do katalogu głównej z folderami ptaków zbioru trenującego
test_img_dir = os.path.join(root_dir, 'test') # ścieżka do katalogu głównej z folderami ptaków zbioru testującego
cachedDataFileName = os.path.join(root_dir, "perceptron.data.p") # plik w którym serializowane będą dane imageTuples
perceptronFileName = os.path.join(root_dir, "perceptron.best.{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5") # plik w którym serializowane są wagi perceptronu po procesie uczenia
historyFileName = os.path.join(root_dir, "perceptron.history.p") # plik w którym serializowana jest historia nauki perceptronu
batch_size = 16
epochs = 10000
num_classes = 50 # liczba klas (ptaków) do rozpoznawania
num_kmeans_descriptors = num_classes * 60 * 10 # Liczba losowych deskrptorów branych pod uwagę podczas obliczania cech k-średnich, obecnie średnio 10 na obrazek, musi być dobrana tak aby kmeans wykonywał się wystarczajaco szybko - docelowo mozna zwiększyć
num_features = num_classes * 10 # Liczba grup (cech) równa num_classes * 10 to podobno dobra praktyka
imgMultiply = 10 # ile obrazków zbioru trenującego należy losowo wygenerowac z istniejących obrazków
ignoreCache = False # czy należy ignorować cache wartości i przeprowadzić wszystkie obliczenia na nowo

## Pobranie danych wejściowych i obliczenie deskryptorów SIFT dla każdego obrazka i zwrócenie listy tuple (lista deskryptorów tego obrazka, klasa obrazka)
def getSiftData(setDir, multiply):
    # weź wszystkie podkatalogi które mają pliki jpg, (w sumie jest 50 podkatalogów (rodzajów ptaków), 60 zdjęć każdy (60 obrazków danego ptaka))
    siftData = []
    class_no = 0
    for dirName in [name for name in os.listdir(setDir) if os.path.isdir(os.path.join(setDir, name))]:
        print("Processing SIFT folder " + dirName)
        dirPath = os.path.join(setDir, dirName)
        fileNames = [name for name in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, name))]
        for fileName in fileNames:
            filePath = os.path.join(dirPath, fileName)
            img = cv2.imread(filePath)
            siftData.append((getSiftDescriptors(img), class_no))
            if(multiply):
                for newImg in multiplyImage(img):
                    siftData.append((getSiftDescriptors(newImg), class_no))
        class_no = class_no + 1
    random.seed(111) # losowść za każdym razem taka sama
    random.shuffle(siftData)
    return siftData

## Obliczenie deskryptorów SIFT dla każdego obrazka ze zbioru treningowego lub testowego i zwrócenie tuple (lista deskryptorów tego obrazka, klasa obrazka)
def getSiftDescriptors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ##convert image to gray scale
    sift = cv2.xfeatures2d.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

def multiplyImage(img):
    datagen = ImageDataGenerator(
        rotation_range=20,
        height_shift_range=0.2,
        width_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    x = img_to_array(img_pil)
    x = x.reshape((1,) + x.shape)

    imgs = []
    i = 0
    for batch in datagen.flow(x, batch_size=imgMultiply, seed=111):
        imgs.extend(batch)
        i += 1
        if(i >= imgMultiply):
            break

    return np.asarray(imgs).astype('uint8')

# Obliczenie klasyfikatora k-średnich dla deskryptorów SIFT, innymi słowy stworzenie funkcji przypisującej deskryptory do grup (cech)
def getKMeans(trainSet):
    print("KMeans started...") # wyświetl postęp w konsoli
    flatDescriptors = [desc for imgTuple in trainSet for desc in imgTuple[0]] # weź wszystkie deskryptory wszystkich obrazów
    random.seed(111) # losowść za każdym razem taka sama
    random.shuffle(flatDescriptors)
    flatDescriptorsLimited = flatDescriptors[:num_kmeans_descriptors] ## ogranicz liczbę deskryptorów do obliczania k-średnich, tak aby algorytm wykonywał się wystarczająco szybko
    kmeans = KMeans(n_clusters=num_features) # Liczba k grup na które będą podzielone deskryptory
    kmeans.fit(flatDescriptorsLimited)
    return kmeans

#  Obliczenie cech dla każdego obrazka i zwrócenie tuple (histogram cech obrazka, klasa obrazka)
def getFeatureData(imgSet, kmeans):
    imgFTuples = [] # wynik funkcji
    for descriptors, class_no in imgSet:
        img_features = kmeans.predict(descriptors)
        img_feature_histogram = [0] * num_features
        for img_feature in img_features:
            img_feature_histogram[img_feature] += 1
        img_feature_histogram_normalized = [float(i)/sum(img_feature_histogram) for i in img_feature_histogram]
        imgFTuples.append((img_feature_histogram_normalized, class_no))
    return imgFTuples

# Pobranie list danych wejściowych i oczekiwanych wyjść sieci
def getXYData(fSet):
    xSet = [img[0] for img in fSet]
    ySet = keras.utils.to_categorical([img[1] for img in fSet], num_classes)
    return (np.asarray(xSet).astype('float32'), np.asarray(ySet).astype('float32'))

#główne ciało skryptu
# jeśli istnienie cache danych to zwróć go bez ponownego obliczania wartości
if(os.path.isfile(cachedDataFileName) and not ignoreCache):
    print("Using cached data") # wyświetl postęp w konsoli
    x_train, y_train, x_test, y_test = pickle.load( open( cachedDataFileName, "rb" ) )
else:
    trainSet = getSiftData(train_img_dir, True) # pobierz listę par (lista deskryptorów SIFT obrazka, klasa obrazka)
    testSet = getSiftData(test_img_dir, False) # pobierz listę par (lista deskryptorów SIFT obrazka, klasa obrazka)
    kmeans = getKMeans(trainSet) # oblicz grupowanie k-średnich dla deskryptorów zbioru treningowego, będą to cechy przypisywane do obrazków
    trainFSet = getFeatureData(trainSet, kmeans) # pobierz listę par (histogram cech obrazka, klasa obrazka)
    testFSet = getFeatureData(testSet, kmeans) # pobierz listę par  (histogram cech obrazka, klasa obrazka)
    x_train, y_train = getXYData(trainFSet) # Pobranie list danych wejściowych i oczekiwanych wyjść sieci
    x_test, y_test = getXYData(testFSet) # Pobranie list danych wejściowych i oczekiwanych wyjść sieci
    print("Caching network input data...") # wyświetl postęp w konsoli
    pickle.dump((x_train, y_train, x_test, y_test), open(cachedDataFileName, "wb"))

model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(num_features,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=0, mode='max')
mcp_save = ModelCheckpoint(perceptronFileName, monitor='val_categorical_accuracy', save_best_only=True, mode='max')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop, mcp_save])
pickle.dump(history.history, open(historyFileName, "wb"))