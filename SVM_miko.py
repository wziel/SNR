from sklearn import svm
from numpy import genfromtxt
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn import preprocessing

root_dir = os.path.join('C:\\', 'Informatyka', 'SNR', 'conv')
model_dir=os.path.join(root_dir, 'conv_final.best.160-0.4517.hdf5')
train_img_dir = os.path.join(root_dir, 'images')
output_dir= os.path.join(root_dir, 'out.txt')

def load_data(csv_data_file_name, train_ratio=0.8):
    array = genfromtxt(csv_data_file_name, delimiter=',')
    array_scaled_values = preprocessing.scale(array[:, :-1])
    array = np.column_stack((array_scaled_values, array[:, -1]))
    class_labels = np.unique(array[:,-1].astype(np.int32))
    print("Data loading done")
    print("Loaded {} samples splitted into {} classes".format(array.shape[0], class_labels.shape[0]))
    print("Splitting each class by {ratio} ratio".format(ratio=train_ratio))

    x_train = np.empty((0, array.shape[1] - 1))
    y_train = np.empty((0,1))
    x_test = np.empty((0, array.shape[1] - 1))
    y_test = np.empty((0,1))

    for c in np.nditer(class_labels):
        row_filter = array[:,-1].astype(np.int32) == c
        class_samples = array[row_filter]
        np.random.shuffle(class_samples)

        training_set_size = int(train_ratio * class_samples.shape[0])

        if float(training_set_size) != train_ratio * class_samples.shape[0]:
            training_set_size += 1

        x_train_class = class_samples[:training_set_size, :-1]
        y_train_class = class_samples[:training_set_size, -1:].astype(dtype=np.int32)

        x_test_class = class_samples[training_set_size:, :-1]
        y_test_class = class_samples[training_set_size:, -1:].astype(dtype=np.int32)

        x_train = np.append(x_train, x_train_class, axis=0)
        y_train = np.append(y_train, y_train_class, axis=0)

        x_test = np.append(x_test, x_test_class, axis=0)
        y_test = np.append(y_test, y_test_class, axis=0)

    return x_train, y_train.flatten(), x_test, y_test.flatten()


train_x, train_y, test_x, test_y = load_data(output_dir, 0.8)

#this is SVM one-vs-one classifier
for deg in range(1,6):
    print(deg)
    clf = svm.SVC(kernel='poly', degree=deg)
    clf.fit(train_x, train_y)
    print("Train: {0:.4f}".format(clf.score(train_x, train_y)))
    print("Test: {0:.4f}".format(clf.score(test_x, test_y)))
