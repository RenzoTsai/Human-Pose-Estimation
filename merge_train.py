import numpy as np
import os
import pandas as pd
from keras.engine.saving import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

import fall_model

select_data = []
vector = []

n_frames = 80

label_count = 0


def extract_frames(frames, points, labels):
    idx = np.array([i for i in range(0, len(frames))])
    new_frames = frames[idx]
    new_points = points[idx]
    global label_count

    new_labels = labels[idx + label_count]
    label_count += len(frames)
    return new_frames, new_points, new_labels


seq_len = 10


def create_dataset(frames, points, labels):
    x_frames = []
    x_points = []
    y = []
    for i in range(len(frames) - seq_len + 1):
        one_set_x_frames = frames[i: i + seq_len, ]
        one_set_x_points = points[i: i + seq_len, ]
        one_y = labels[i + seq_len - 1]
        x_frames.append(one_set_x_frames)
        x_points.append(one_set_x_points)
        y.append(one_y)
    x_frames = np.array(x_frames)
    x_points = np.array(x_points)
    y = np.array(y)
    return x_frames, x_points, y


def load_data():
    img_npy_path = "./output/deep_npy"
    if os.path.exists(os.path.join(img_npy_path, ".DS_Store")):
        os.remove(os.path.join(img_npy_path, ".DS_Store"))
    img_filenames = os.listdir(img_npy_path)
    img_filenames.sort()

    point_npy_path = "./output/processed_npy"
    if os.path.exists(os.path.join(point_npy_path, ".DS_Store")):
        os.remove(os.path.join(point_npy_path, ".DS_Store"))
    point_filenames = os.listdir(point_npy_path)
    point_filenames.sort()

    label_data = pd.read_csv("./dataset/UR/urfall-cam0-falls.csv")
    labels = label_data["label"]
    y = to_categorical(labels, num_classes=3)
    labels = y

    print(os.path.join(img_npy_path, img_filenames[0]))
    one_img_data = np.load(os.path.join(img_npy_path, img_filenames[0]), allow_pickle=True)

    one_point_data = np.load(os.path.join(point_npy_path, point_filenames[0]), allow_pickle=True)
    new_x_img, new_x_point, new_y = extract_frames(one_img_data, one_point_data, labels)

    x_img, x_point, y = create_dataset(new_x_img, new_x_point, new_y)

    for i in range(1, len(img_filenames)):
        file = os.path.splitext(img_filenames[i])
        filename, file_type = file
        if file_type == ".npy":
            one_img_data = np.load(os.path.join(img_npy_path, img_filenames[i]), allow_pickle=True)
            one_point_data = np.load(os.path.join(point_npy_path, point_filenames[i]), allow_pickle=True)
            print(one_img_data.shape, one_point_data.shape, y.shape)
            new_x_img, new_x_point, new_y = extract_frames(one_img_data, one_point_data, labels)
            new_x_img, new_x_point, new_y = create_dataset(new_x_img, new_x_point, new_y)
            x_img = np.vstack((x_img, new_x_img))

            x_point = np.vstack((x_point, new_x_point))
            y = np.vstack((y, new_y))

    x_img = x_img.reshape((-1, x_img.shape[1], x_img.shape[2], x_img.shape[3], 1))
    x_point = x_point.reshape((-1, x_point.shape[1], x_point.shape[2] * x_point.shape[3]))

    x_idx = np.array(range(0, len(x_point)))
    x_idx_train, x_idx_test, y_train, y_test = train_test_split(x_idx, y, test_size=0.2, random_state=2021)


    x_img_train = x_img[x_idx_train]
    x_point_train = x_point[x_idx_train]
    x_img_test = x_img[x_idx_test]
    x_point_test = x_point[x_idx_test]

    print(x_img_train.shape, x_point_train.shape, y_train.shape, x_img_test.shape, x_point_test.shape, y_test.shape)
    return x_img_train, x_point_train, y_train, x_img_test, x_point_test, y_test


if __name__ == '__main__':
    x_img_train, x_point_train, y_train, x_img_test, x_point_test, y_test = load_data()
    fall_model.combined_model(x_img_train, x_point_train, y_train, x_img_test, x_point_test, y_test)
