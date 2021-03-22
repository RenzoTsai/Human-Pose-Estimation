import numpy as np
import os
import pandas as pd
from keras.engine.saving import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from fall_model import baseline_model
from fall_model import lstm_model

select_data = []
vector = []

BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14}

n_frames = 50

label_count = 0


def extract_frames(frames, labels):
    # n_frames = 10 * int(len(frames) / 10)
    n_frames = len(frames)
    idx = np.round(np.linspace(0, len(frames)-1, n_frames)).astype(int)
    new_frames = frames[idx]
    global label_count
    new_labels = labels[idx + label_count]
    label_count += len(frames)
    return new_frames, new_labels


seq_len = 10


def create_dataset(frames, labels):
    x = []
    y = []
    for i in range(len(frames) - seq_len + 1):
        one_set_x = frames[i: i + seq_len, ]
        one_y = labels[i + seq_len - 1]
        x.append(one_set_x)
        y.append(one_y)
    x = np.array(x)
    y = np.array(y)
    return x, y


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

    label_data = pd.read_csv("./dataset/UR/urfall-cam0-falls.csv")
    labels = label_data["label"]
    y = to_categorical(labels, num_classes=3)
    labels = y
    one_data = np.load(os.path.join(npy_path, filenames[0]))
    new_x, new_y = extract_frames(one_data, labels)
    print(one_data.shape)
    # x = new_x
    # y = new_y
    x, y = create_dataset(new_x, new_y)

    for i in range(1, len(filenames)):
        file = os.path.splitext(filenames[i])
        filename, file_type = file
        if file_type == ".npy":
            one_data = np.load(os.path.join(npy_path, filenames[i]), allow_pickle=True)
            print(one_data.shape)
            new_x, new_y = extract_frames(one_data, labels)
            new_x, new_y = create_dataset(new_x, new_y)

            x = np.vstack((x, new_x))
            y = np.vstack((y, new_y))

    print(x.shape)

    # y = np.array(label_to_vector(y))
    print(y.shape)
    # exit(0)
    # se = select(x)
    # x = se.reshape((-1, se.shape[1] * se.shape[2]))
    # x = se.reshape((-1, 1, se.shape[1] * se.shape[2]))
    x = x.reshape((-1, x.shape[1], x.shape[2] * x.shape[3]))

    one_test = x[x.shape[0]-2]
    print(y[x.shape[0]-2])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2021)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test , one_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, test = load_data()
    # baseline_model(x_train, y_train, x_test, y_test)
    lstm_model(x_train, y_train, x_test, y_test)

    model = load_model('fall_model.h5')
    test = test.reshape(1, test.shape[0], test.shape[1])

    pred = model.predict(test)
    print(test, pred)
