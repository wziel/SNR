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

root_dir = os.path.join('C:\\', 'Users', 'Wojciech', 'Desktop', 'SNR') # ścieżka do katalogu głównego gdzie opisane są boundingbox. Należy zmienić w zależności od środowiska uruchomienia.
root_img_dir = os.path.join(root_dir, 'SET_B') # ścieżka do katalogu głównej z folderami ptaków
train_img_dir = os.path.join(root_dir, 'train') # ścieżka do katalogu głównej z folderami ptaków
test_img_dir = os.path.join(root_dir, 'test') # ścieżka do katalogu głównej z folderami ptaków
bounding_boxes_file_name = os.path.join(root_dir, "bounding_boxes.txt")
percentOfTraingSet = 0.8 # procent obrazków trafiających do zbioru trenującego, zbiór testowy będzie zawierał resztę
## Pobranie danych wejściowych i obliczenie deskryptorów SIFT dla każdego obrazka i zwrócenie listy tuple (lista deskryptorów tego obrazka, klasa obrazka)

def splitAndBoundImgs():
    boundMap = createboundMap()
    random.seed(111) # losowść za każdym razem taka sama
    for dirName in [name for name in os.listdir(root_img_dir) if os.path.isdir(os.path.join(root_img_dir, name))]:
        print("Processing folder " + dirName)
        dirPath = os.path.join(root_img_dir, dirName)
        fileNames = [name for name in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, name))]
        random.shuffle(fileNames)
        splitPoint = int(percentOfTraingSet * len(fileNames)) # punkt podziału tego folderu na zbiór testowy i treningowy
        for fileName in fileNames[:splitPoint:]:
            filePath = os.path.join(dirPath, fileName)
            img = boundImage(cv2.imread(filePath), boundMap[fileName[0:32]])
            newFilePath = os.path.join(train_img_dir, dirName, fileName)
            ensure_dir(newFilePath)
            cv2.imwrite(newFilePath, img)
        for fileName in fileNames[splitPoint::]:
            filePath = os.path.join(dirPath, fileName)
            img = boundImage(cv2.imread(filePath), boundMap[fileName[0:32]])
            newFilePath = os.path.join(test_img_dir, dirName, fileName)
            ensure_dir(newFilePath)
            cv2.imwrite(newFilePath, img)
    
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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)