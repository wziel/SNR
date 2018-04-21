from __future__ import print_function 
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
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


root_dir = 'C:/Users/Wojciech/Desktop/SNR/' # ścieżka do katalogu głównego gdzie opisane są boundingbox
root_img_dir = root_dir + 'SET_B/' # ścieżka do katalogu głównej z folderami ptaków
root_multiply_temp_dir = root_dir + 'SET_B/temp/' # scieżka do katalogu, w którym generowane są tymczasowo przekształcenia
cachedDataFileName = root_dir + "cachedData.p" # plik w którym serializowane będą dane imageTuples
perceptronWeightsFileName = root_dir + "perceptron_weights.h5" # plik w którym serializowane są wagi perceptronu po procesie uczenia
bounding_boxes_file_name = root_dir + "bounding_boxes.txt"
num_classes = 50 # liczba klas (ptaków) do rozpoznawania
num_kmeans_descriptors = num_classes * 60 * 10 # Liczba losowych deskrptorów branych pod uwagę podczas obliczania cech k-średnich, obecnie średnio 10 na obrazek, musi być dobrana tak aby kmeans wykonywał się wystarczajaco szybko - docelowo mozna zwiększyć
num_features = num_classes * 10 # Liczba grup (cech) równa num_classes * 10 to podobno dobra praktyka
percentOfTraingSet = 0.8 # procent obrazków trafiających do zbioru trenującego, zbiór testowy będzie zawierał resztę
batch_size = 50
epochs = 100
ignoreCache = False # czy należy ignorować cache wartości i przeprowadzić wszystkie obliczenia na nowo
        

## Pobranie danych wejściowych i obliczenie deskryptorów SIFT dla każdego obrazka i zwrócenie listy tuple (lista deskryptorów tego obrazka, klasa obrazka)
def getSiftData():
    trainSet = []
    testSet = []
    boundMap = createboundMap()
    class_no = 0 # wyświetlanie postepu w konsoli
    random.seed(111) # losowść za każdym razem taka sama
    # weź wszystkie podkatalogi które mają pliki jpg, (w sumie jest 50 podkatalogów (rodzajów ptaków), 60 zdjęć każdy (60 obrazków danego ptaka))
    for dirPath, _, fileNames in os.walk(root_img_dir):
        if(fileNames):
            #print("Calculating SIFT " + str(class_no) + "/50...") # wyświetl postęp w konsoli
            print("Calculating SIFT {0}/50...\r".format(str(class_no)))
            dirImgs = []
            for fileName in fileNames:
                filePath = dirPath + "/" + fileName
                img = boundImage(cv2.imread(filePath), boundMap[fileName[0:32]])
                dirImgs.append(img)
            random.shuffle(dirImgs)
            splitPoint = int(percentOfTraingSet * len(dirImgs)) # punkt podziału tego folderu na zbiór testowy i treningowy
            for baseTrainImg in dirImgs[:splitPoint:]:
                for trainImg in multiplyImage(baseTrainImg):
                    trainSet.append((getSiftDescriptors(trainImg), class_no))
            for testImg in dirImgs[splitPoint::]:
                testSet.append((getSiftDescriptors(testImg), class_no))
            class_no = class_no + 1
    random.shuffle(trainSet)
    random.shuffle(testSet)
    return (trainSet, testSet)
   
#tworzy słownik gdzie kluczem jest nazwa zdjęcia a wartością lista x,y,xh,yh
def createboundMap():
    boundMap={}
    with open(bounding_boxes_file_name, "r") as boundaries:
        for line in boundaries:
            cordStr=line.split(" ")
            coordinates=list(map(int,cordStr[1:5]))
            key=cordStr[0].replace("-","")
            boundMap[key]=coordinates
    return boundMap

#wycina zdjecie
def boundImage(imgOrig, boundArea):
    x, y,xh,yh=boundArea
    imgCopy=imgOrig[y: y+yh, x:x+xh]
    return imgCopy

def multiplyImage(img):
    # TODO zrobić to w sposób bardziej geenryczny i losowy, obrazów może być też więcej, np. 10 na jeden
    # TODO czy w przypadku SIFT (SCALE INVARIANT feature transform) skalowanie i rotacje mają sens?
    imgs = []
    imgs.append(img)
    datagen = ImageDataGenerator(
        rotation_range=20,
        height_shift_range=0.2,
        width_shift_range=0.2,       
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    x = img_to_array(img_pil)
    x = x.reshape((1,) + x.shape)

    multiplyIndex = 0
    for batch in datagen.flow(x,save_to_dir=root_multiply_temp_dir, save_prefix='bird', save_format='jpeg'):
        multiplyIndex += 1
        if multiplyIndex >= 10:
            break  

    for fileName in os.listdir(root_multiply_temp_dir):
        filePath = root_multiply_temp_dir + fileName
        imgs.append(cv2.imread(filePath))
        os.remove(filePath)
    return imgs

## Obliczenie deskryptorów SIFT dla każdego obrazka ze zbioru treningowego lub testowego i zwrócenie tuple (lista deskryptorów tego obrazka, klasa obrazka)
def getSiftDescriptors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ##convert image to gray scale
    sift = cv2.xfeatures2d.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

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

#  Obliczenie cech dla każdego obrazka i zwrócenie tuple (ścieżka do pliku obrazka, klasa obrazka, histogram cech obrazka)
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
    trainSet, testSet = getSiftData() # pobierz listę par (lista deskryptorów SIFT obrazka, klasa obrazka)
    kmeans = getKMeans(trainSet) # oblicz grupowanie k-średnich dla deskryptorów zbioru treningowego, będą to cechy przypisywane do obrazków
    trainFSet = getFeatureData(trainSet, kmeans) # pobierz listę par (histogram cech obrazka, klasa obrazka)
    testFSet = getFeatureData(testSet, kmeans) # pobierz listę par  (histogram cech obrazka, klasa obrazka)
    x_train, y_train = getXYData(trainFSet) # Pobranie list danych wejściowych i oczekiwanych wyjść sieci
    x_test, y_test = getXYData(testFSet) # Pobranie list danych wejściowych i oczekiwanych wyjść sieci
    pickle.dump((x_train, y_train, x_test, y_test), open(cachedDataFileName, "wb"))

model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(num_features,)))
model.add(Dense(90, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save_weights(perceptronWeightsFileName)
model.load_weights(perceptronWeightsFileName)