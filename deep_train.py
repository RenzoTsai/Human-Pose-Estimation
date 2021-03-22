import numpy as np
import os
import pandas as pd
from keras.engine.saving import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from fall_model import deep_3D_cnn_model
from fall_model import deep_cnn_lstm_model

select_data = []
vector = []

n_frames = 80

label_count = 0


def extract_frames(frames, labels):
    # n_frames = 10 * int(len(frames) / 10)
    # n_frames = len(frames)
    # n_frames = 50
    # idx = np.round(np.linspace(0, len(frames)-1, n_frames)).astype(int)
    idx = np.array([i for i in range(0, len(frames))])
    new_frames = frames[idx]
    print(idx)
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


def load_3d_data():
    npy_path = "./output/deep_npy"
    filenames = os.listdir(npy_path)
    filenames.sort()

    label_data = pd.read_csv("./dataset/UR/urfall-cam0-falls.csv")
    labels = label_data["label"]
    y = to_categorical(labels, num_classes=3)
    labels = y
    one_data = np.load(os.path.join(npy_path, filenames[0]))
    new_x, new_y = extract_frames(one_data, labels)
    print(one_data.shape)
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

    x = x.reshape((-1, x.shape[1], x.shape[2], x.shape[3], 1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2021)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def load_2d_data():
    npy_path = "./output/deep_npy"
    filenames = os.listdir(npy_path)
    filenames.sort()

    label_data = pd.read_csv("./dataset/UR/urfall-cam0-falls.csv")
    labels = label_data["label"]
    y = to_categorical(labels, num_classes=3)
    labels = y
    one_data = np.load(os.path.join(npy_path, filenames[0]))
    new_x, new_y = extract_frames(one_data, labels)
    #print(one_data.shape)
    x, y = create_dataset(new_x, new_y)

    for i in range(1, len(filenames)):
        file = os.path.splitext(filenames[i])
        filename, file_type = file
        if file_type == ".npy":
            one_data = np.load(os.path.join(npy_path, filenames[i]), allow_pickle=True)
            #print(one_data.shape)
            new_x, new_y = extract_frames(one_data, labels)
            new_x, new_y = create_dataset(new_x, new_y)

            x = np.vstack((x, new_x))
            y = np.vstack((y, new_y))

    print(x.shape)
    x = x.reshape((-1, x.shape[2], x.shape[3], 1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2021)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_3d_data()
    # deep_3D_cnn_model(x_train, y_train, x_test, y_test)
    deep_cnn_lstm_model(x_train, y_train, x_test, y_test)

