from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import os #foldery i pliki
import re #regular expressions
import cv2 #opencv
import pickle # serializacja obiektów do pliku
import numpy as np #biblioteka numeryczna, tablice wielowymiarowe i utils
import matplotlib.pyplot as plt #rysowanie wykresów
from sklearn.cluster import KMeans # algorytm k średnich

root_img_dir = 'C:/Users/Wojciech/Desktop/SNR/SET_B/' #ścieżka do katalogu głównej z folderami ptaków
precalculatedImagesTuplesFileName = root_img_dir + "cachedImagesTuples.p" # plik w którym serializowane będą dane imageTuples
precalculatedKMeansFileName = root_img_dir + "cachedKMeans.p" # plik w którym serializowane będą dane imageTuples
num_classes = 50 # liczba klas (ptaków) do rozpoznawania
num_descriptors = 100 # TODO dla każdego obrazka pobierane tylko 100 pierwszych deskryptorów. czy takie ograniczenie jest ok? czy na początku są najważniejsze? czy nie powinno być np. 1000?
num_features = num_classes * 10 # Liczba grup (cech) równa num_classes * 10 to podobno dobra praktyka
clearCache = False # czy należy wyczyścić cache wartości i rozpocząć wszystkie obliczenia na nowo

## Obliczenie deskryptorów SIF dla każdego obrazka i zwrócenie tuple (ścieżka do pliku obrazka, klasa obrazka, lista deskryptorów tego obrazka)
def getImageTuples():
    # jeśli istnienie cache danych to zwróć go bez ponownego obliczania wartości
    if(os.path.isfile(precalculatedImagesTuplesFileName)):
        print("Using precalculated SIFT values") # wyświetl postęp w konsoli
        return pickle.load( open( precalculatedImagesTuplesFileName, "rb" ) )
    # oblicz od początku wszystkie wartości SIFT
    imgTuples = [] # wynik funkcji
    class_no = 0 # wyświetlanie postepu w konsoli
    # weź wszystkie podkatalogi które mają pliki jpg, (w sumie jest 50 podkatalogów (rodzajów ptaków), 60 zdjęć każdy (60 obrazków danego ptaka))
    for dirPath, dirNames, fileNames in os.walk(root_img_dir):
        print("SIFT processing bird class " + str(class_no) + " from folder " + os.path.basename(dirPath) + "...") # wyświetl postęp w konsoli
        if(fileNames):
            for fileName in fileNames:
                filePath = dirPath + "/" + fileName
                descriptors = getSIFTDescriptors(filePath)[:num_descriptors]
                imgTuple = (filePath, class_no, descriptors)
                imgTuples.append(imgTuple)
                break #TODO odkomentować, obecnie pobierany tylko 1 obrazek dla każdego ptaka, tylko ze wzgleu na wydajność (SIFT zajmuje dużo czasu)
            class_no = class_no + 1
    print("SIFT done. Caching result.") # wyświetl postęp w konsoli
    pickle.dump(imgTuples, open(precalculatedImagesTuplesFileName, "wb"))
    return imgTuples

# Obliczenie deskryptorów SIFT dla obrazka o zadanej ścieżce na dysku
def getSIFTDescriptors(imgPath):
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ##convert image to gray scale
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

# Obliczenie klasyfikatora k-średnich dla deskryptorów, innymi słowy stworzenie funkcji przypisującej deskryptory do grup (cech)
def getKMeansOfDescriptors(imgTuples):
    if(os.path.isfile(precalculatedKMeansFileName)):
        print("Using precalculated KMeans") # wyświetl postęp w konsoli
        return pickle.load( open( precalculatedKMeansFileName, "rb" ) )
    print("KMeans started...") # wyświetl postęp w konsoli
    flatDescriptors = [desc for imgTuple in imgTuples for desc in imgTuple[2]] # weź wszystkie deskryptory wszystkich obrazów
    kmeans = KMeans(n_clusters=num_features) # Liczna grup równa num_classes * 10 to podobno dobra praktyka
    kmeans.fit(flatDescriptors)
    print("KMeans Done. Caching result.") # wyświetl postęp w konsoli
    pickle.dump(kmeans, open(precalculatedKMeansFileName, "wb"))
    return kmeans

#wyczyszczenie cache'a obliczeń
def clearCacheIfRequested():
    if(clearCache):
        os.remove(precalculatedImagesTuplesFileName)

#  Obliczenie cech dla każdego obrazka i zwrócenie tuple (ścieżka do pliku obrazka, klasa obrazka, histogram cech obrazka)
def getImageFeatureTuples(imgTuples, kmeans):
    imgFTuples = [] # wynik funkcji
    for imgTuple in imgTuples:
        descriptors = imgTuple[2]
        img_features = kmeans.predict(descriptors)
        img_feature_histogram = [0] * num_features
        for img_feature in img_features:
            img_feature_histogram[img_feature] += 1
        img_feature_histogram_normalized = [float(i)/sum(img_feature_histogram) for i in img_feature_histogram]
        imgFTuples.append((imgTuple[0], imgTuple[1], img_feature_histogram_normalized))
    return imgFTuples

#główne ciało skryptu
clearCacheIfRequested()
imgTuples = getImageTuples() # pobierz listę trójek (ścieżka do obrazka, klasa obrazka, lista deskryptorów SIFT obrazka)
kmeans = getKMeansOfDescriptors(imgTuples) # oblicz grupowanie k-średnich dla deskryptorów, będą to cechy przypisywane do obrazków
imgFTuples = getImageFeatureTuples(imgTuples, kmeans) # pobierz listę trójek (ścieżka do obrazka, klasa obrazka, histogram cech obrazka)

model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(num_features)))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# TODO należy dodać podział na zbiór trenujący i testujący (proporcje odpowiednio 0.8 do 0.2)
'''
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('first_try.h5')  # always save your weights after training or during training
'''