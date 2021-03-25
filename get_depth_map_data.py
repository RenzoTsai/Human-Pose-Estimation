import os
import cv2
import numpy as np
import tensorflow as tf

DEEP_PATH = "./dataset/UR/depth"


def get_depth_data(folder_path):
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
    if not os.path.exists("output/depth_npy"):
        os.mkdir("output/depth_npy")
    folders = os.listdir(DEEP_PATH)
    folders.sort()
    for folder in folders:
        if not folder == ".DS_Store":
            print(os.path.join(DEEP_PATH, folder))
            depth_data = get_depth_data(os.path.join(DEEP_PATH, folder))
            print(depth_data.shape)
            new_filename = folder + "_new"
            new_path = os.path.join("output/depth_npy", new_filename)
            np.save(new_path, depth_data)
