import os
from sklearn import svm
from numpy import genfromtxt
import numpy as np
import argparse
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.preprocessing.image import img_to_array
from PIL import Image
import sys

root_dir = os.path.join('C:\\', 'Informatyka', 'SNR', 'conv')
model_dir=os.path.join(root_dir, 'conv_final.best.160-0.4517.hdf5')
train_img_dir = os.path.join(root_dir, 'train')
output_dir= orig_stdout = os.path.join(root_dir, 'out.txt')



model = keras.models.load_model(model_dir)

#print(model.summary())

layer_name = 'dense_2'

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
f = open(output_dir, 'w')
orig_stdout = sys.stdout
sys.stdout = f
for root, subdirs, files in os.walk(train_img_dir):
    cls_id = 0
    for d in subdirs:
        cls = d
        num_files_in_dir = 0
        list_of_files = []
        for root, subdirs, files in os.walk(os.path.join(train_img_dir, d)):
            num_files_in_dir = len(files)
            list_of_files = files

        for f in list_of_files:

            file_path = os.path.join(train_img_dir, d, f)
            img = Image.open(file_path)
            if (int(model.input.shape[1]),int(model.input.shape[2])) != img.size:
                img = img.resize((int(model.input.shape[1]),int(model.input.shape[2])))
            rgb = Image.new("RGB", img.size, (255, 255, 255))
            rgb.paste(img)

            arr = img_to_array(rgb)
            intermediate_output = intermediate_layer_model.predict(np.array([arr]))
            intermediate_output = intermediate_output.flatten()
            intermediate_output = intermediate_output.tolist()
            csv_line = ",".join([str(i) for i in intermediate_output]) + "," + str(cls_id)
            print(csv_line)

        cls_id += 1
sys.stdout = orig_stdout
f.close()
