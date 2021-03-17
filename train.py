import numpy as np
import os
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from fall_model import baseline_model
from fall_model import lstm_model

select_data = []
vector = []

BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14}


def select(x):
    for s in x:
        # select_data.append((s[0], s[1], s[2], s[5], s[8], s[9],  s[11], s[12]))
        select_data.append((s[0], s[1], s[8], s[9], s[11], s[12]))
    return np.array(select_data)


def merge(x1, x2):
    x3 = np.vstack((x1, x2))
    print(x3)


def label_to_vector(y):
    for v in y:
        if v == -1:
            vector.append((1, 0))
        if v == 0:
            vector.append((1, 1))
        if v == 1:
            vector.append((0, 1))
    return vector


def load_data():
    npy_path = "./output/npy"
    filenames = os.listdir(npy_path)
    filenames.sort()

    x = np.load(os.path.join(npy_path, filenames[0]))

    for i in range(1, len(filenames)):
        file = os.path.splitext(filenames[i])
        filename, file_type = file
        if file_type == ".npy":
            one_data = np.load(os.path.join(npy_path, filenames[i]), allow_pickle=True)
            x = np.vstack((x, one_data))

    print(x.shape)
    label_data = pd.read_csv("./dataset/UR/urfall-cam0-falls.csv")
    y = label_data["label"]
    # y = y[84:189]
    y = to_categorical(y, num_classes=3)

    # y = np.array(label_to_vector(y))
    print(y.shape)
    se = select(x)
    # x = x.reshape((-1, x.shape[1] * x.shape[2]))
    x = se.reshape((-1, 1, se.shape[1] * se.shape[2]))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2021)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    # baseline_model(x_train, y_train, x_test, y_test)
    lstm_model(x_train, y_train, x_test, y_test)
