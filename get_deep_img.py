import os
import cv2
import numpy as np
import tensorflow as tf

DEEP_PATH = "./dataset/UR/deep"


def get_deep_data(folder_path):
    filenames = os.listdir(folder_path)
    filenames.sort()
    data = []
    for file in filenames:
        if not file == ".DS_Store":
            img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
            data.append(img)
    return np.array(data)


if __name__ == '__main__':
    if not os.path.exists("output/deep_npy"):
        os.mkdir("output/deep_npy")
    folders = os.listdir(DEEP_PATH)
    folders.sort()
    for folder in folders:
        if not folder == ".DS_Store":
            print(os.path.join(DEEP_PATH, folder))
            deep_data = get_deep_data(os.path.join(DEEP_PATH, folder))
            print(deep_data.shape)
            new_filename = folder + "_new"
            new_path = os.path.join("output/deep_npy", new_filename)
            np.save(new_path, deep_data)
