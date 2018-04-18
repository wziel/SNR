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

root_img_dir = 'C:/Informatyka/SNR/SET_B/' # ścieżka do katalogu głównej z folderami ptaków
root_= 'C:/Informatyka/SNR/' #ścieżka do katalogu głównego gdzie opisane są boundingbox
cachedImagesTuplesFileName = root_img_dir + "cachedImagesTuples.p" # plik w którym serializowane będą dane imageTuples
cachedKMeansFileName = root_img_dir + "cachedKMeans.p" # plik w którym serializowane będą dane imageTuples
perceptronWeightsFileName = root_img_dir + "perceptron_weights.h5" # plik w którym serializowane są wagi perceptronu po procesie uczenia
num_classes = 50 # liczba klas (ptaków) do rozpoznawania
num_kmeans_descriptors = num_classes * 60 * 10 # Liczba losowych deskrptorów branych pod uwagę podczas obliczania cech k-średnich, obecnie średnio 10 na obrazek, musi być dobrana tak aby kmeans wykonywał się wystarczajaco szybko - docelowo mozna zwiększyć
num_features = num_classes * 10 # Liczba grup (cech) równa num_classes * 10 to podobno dobra praktyka
percentOfTraingSet = 0.8 # procent obrazków trafiających do zbioru trenującego, zbiór testowy będzie zawierał resztę
batch_size = 50
epochs = 1000
ignoreCache = False # czy należy ignorować cache wartości i przeprowadzić wszystkie obliczenia na nowo

# zmienne dotyczace modyfikacji obrazow wejsciowych
rotateDegree = 8
gaussianSize = 7
percentScale = 5

## Obliczenie deskryptorów SIFT dla każdego obrazka ze zbioru treningowego lub testowego i zwrócenie tuple (klasa obrazka, lista deskryptorów tego obrazka)
def SIFT(imgTuples, isTraining):
    imgSIFTTuples = []
    tuplesLength = len(imgTuples)
    it = 0
    for img, class_no in imgTuples:
        mili = getCurrentMiliTime() - startMili
        startMili = getCurrentMiliTime()
        imgs = []
        imgs.append(img)
        # dla zbioru treningowego wykonanie przekształceń oryginalnego zdjęcia
        if(isTraining):
            print("SIFT processing bird class " + str(it) + "/" + str(tuplesLength) + " from training set " + str(mili) + "ms ...") # wyświetl postęp w konsoli
            (height, width) = img.shape[:2]
            center = (width/2, height/2)
            imgs.append(cv2.warpAffine(img, cv2.getRotationMatrix2D(center, rotateDegree, 1.0), (width, height)))
            imgs.append(cv2.warpAffine(img, cv2.getRotationMatrix2D(center, -rotateDegree, 1.0), (width, height)))
            imgs.append(cv2.GaussianBlur(img,(gaussianSize, gaussianSize), 0))
            imgs.append(cv2.resize(img, None, fx=1, fy=(1.00 + percentScale/100), interpolation = cv2.INTER_CUBIC))
            imgs.append(cv2.resize(img, None, fx=(1.00 + percentScale/100), fy=1, interpolation = cv2.INTER_CUBIC))
        else:
            print("SIFT processing bird class " + str(it) + "/" + str(tuplesLength) + " from test set " + str(mili) + "ms ...") # wyświetl postęp w konsoli
        #obliczenie deskryptorów SIFT - dla zbioru testowego tylko jedno zdjecie / dla zbioru treningowego zdjecie oryginalne i przekształcone    
        for image in imgs:
            descriptors = getSIFTDescriptors(image)
            imgSIFTTuples.append((class_no, descriptors))
        it = it + 1
    # losowość zbioru
    random.seed(111)
    random.shuffle(imgSIFTTuples)
    return imgSIFTTuples


## Pobranie danych wejściowych i podzielenie na zbiór testowy i treningowy
def getTreningAndTestData():
    # jeśli istnienie cache danych to zwróć go bez ponownego obliczania wartości
    if(os.path.isfile(cachedImagesTuplesFileName) and not ignoreCache):
        print("Using precalculated SIFT values") # wyświetl postęp w konsoli
        return pickle.load( open( cachedImagesTuplesFileName, "rb" ) )
     # oblicz od początku wszystkie wartości SIFT
    imgTuples = []
    boundMap = createboundMap()
    class_no = 0 # wyświetlanie postepu w konsoli
    # weź wszystkie podkatalogi które mają pliki jpg, (w sumie jest 50 podkatalogów (rodzajów ptaków), 60 zdjęć każdy (60 obrazków danego ptaka))
    for dirPath, _, fileNames in os.walk(root_img_dir):
        if(fileNames):
            for fileName in fileNames:
                filePath = dirPath + "/" + fileName
                img = boundPicture(cv2.imread(filePath),boundMap[fileName[0:32]])
                imgTuple = (img, class_no)
                imgTuples.append(imgTuple)
            class_no = class_no + 1
    random.seed(111) # losowść za każdym razem taka sama
    random.shuffle(imgTuples)
     # podziel zdjecia na zbior testowy i treningowy
    splitPoint = int(percentOfTraingSet * len(imgTuples))
    data = (imgTuples[:splitPoint:], imgTuples[splitPoint::])
    return data

## Pobranie danych wejściowych i obliczenie deskryptorów SIFT dla każdego obrazka i zwrócenie tuple (klasa obrazka, lista deskryptorów tego obrazka)
def getSIFTData():
   (imgTuplesTraining, imgTuplesTest) = getTreningAndTestData()
   startMili = getCurrentMiliTime()
   data = (SIFT(imgTuplesTraining, true), SIFT(imgTuplesTest, false))
   print("SIFT done. Caching result.") # wyświetl postęp w konsoli
   pickle.dump(data, open(cachedImagesTuplesFileName, "wb"))
   return data
   

# Obliczenie deskryptorów SIFT dla obrazka o zadanej ścieżce na dysku
def getSIFTDescriptors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ##convert image to gray scale
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

#tworzy słownik gdzie kluczem jest nazwa zdjęcia a wartością lista x,y,xh,yh
def createboundMap():
    boundMap={}
    startMili=getCurrentMiliTime()
    print("Processing boundaries for images")
    with open(root_+"bounding_boxes.txt","r") as boundaries:
        for line in boundaries:
            cordStr=line.split(" ")
            coordinates=list(map(int,cordStr[1:5]))
            key=cordStr[0].replace("-","")
            boundMap[key]=coordinates
    print ("Dictionary for boundaries has been created in: "+str(getCurrentMiliTime()-startMili)+"ms")
    return boundMap

#wycina zdjecie
def boundPicture(imgOrig, boundArea):
    x, y,xh,yh=boundArea
    imgCopy=imgOrig[y: y+yh, x:x+xh]
    return imgCopy

def getCurrentMiliTime():
    return int(round(time.time() * 1000))

# Obliczenie klasyfikatora k-średnich dla deskryptorów SIFT, innymi słowy stworzenie funkcji przypisującej deskryptory do grup (cech)
def getSIFTKMeans(imgTuples):
    if(os.path.isfile(cachedKMeansFileName) and not ignoreCache):
        print("Using precalculated KMeans") # wyświetl postęp w konsoli
        return pickle.load( open( cachedKMeansFileName, "rb" ) )
    print("KMeans started...") # wyświetl postęp w konsoli
    flatDescriptors = [desc for imgTuple in imgTuples for desc in imgTuple[1]] # weź wszystkie deskryptory wszystkich obrazów
    random.seed(111) # losowść za każdym razem taka sama
    random.shuffle(flatDescriptors)
    flatDescriptorsLimited = flatDescriptors[:num_kmeans_descriptors] ## ogranicz liczbę deskryptorów do obliczania k-średnich, tak aby algorytm wykonywał się wystarczająco szybko
    kmeans = KMeans(n_clusters=num_features) # Liczba k grup na które będą podzielone deskryptory
    kmeans.fit(flatDescriptorsLimited)
    print("KMeans Done. Caching result.") # wyświetl postęp w konsoli
    pickle.dump(kmeans, open(cachedKMeansFileName, "wb"))
    return kmeans

#  Obliczenie cech dla każdego obrazka i zwrócenie tuple (ścieżka do pliku obrazka, klasa obrazka, histogram cech obrazka)
def getFeatureData(imgTuples, kmeans):
    imgFTuples = [] # wynik funkcji
    for imgTuple in imgTuples:
        descriptors = imgTuple[1]
        img_features = kmeans.predict(descriptors)
        img_feature_histogram = [0] * num_features
        for img_feature in img_features:
            img_feature_histogram[img_feature] += 1
        img_feature_histogram_normalized = [float(i)/sum(img_feature_histogram) for i in img_feature_histogram]
        imgFTuples.append((imgTuple[0], img_feature_histogram_normalized))
    return imgFTuples

# Pobranie list danych wejściowych i oczekiwanych wyjść sieci
def getXYData(fSet):
    xSet = [img[1] for img in fSet]
    ySet = keras.utils.to_categorical([img[0] for img in fSet], num_classes)
    return (np.asarray(xSet).astype('float32'), np.asarray(ySet).astype('float32'))

#główne ciało skryptu
trainSet, testSet = getSIFTData() # pobierz listy trójek (ścieżka do obrazka, klasa obrazka, lista deskryptorów SIFT obrazka)
kmeans = getSIFTKMeans(trainSet) # oblicz grupowanie k-średnich dla deskryptorów, będą to cechy przypisywane do obrazków
trainFSet = getFeatureData(trainSet, kmeans) # pobierz listę trójek (ścieżka do obrazka, klasa obrazka, histogram cech obrazka)
testFSet = getFeatureData(testSet, kmeans) # pobierz listę trójek (ścieżka do obrazka, klasa obrazka, histogram cech obrazka)
x_train, y_train = getXYData(trainFSet) # Pobranie list danych wejściowych i oczekiwanych wyjść sieci
x_test, y_test = getXYData(testFSet) # Pobranie list danych wejściowych i oczekiwanych wyjść sieci

model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(num_features,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
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

# TODO czy należy zwiększyć sztucznie liczbę obrazków? Dodawanie losowych rotacji, skalowania itd - jest takie narzędzie w pythonie. Obecnie mało obrazków, tylko 60 na 1 klasę.
# TODO czy obrazki trzeba jakoś wstępnie obrobić? obecnie są na nich gałęzie itp. nie powinny być wykadrowane na ptaka? Bouding Boxes.
# TODO num_features = num_classes * 10 czy taka wartość jest okej? 500 cech, przy 50 klasach ptaków
# TODO batch_size = 20 jak batch size wpływa na uczenie? jaki powinien być rozmiar?