import os

import numpy as np

n_frames = 30
def load_data(data_path):
    return np.load(data_path)


def replace_invalid_points(old_points, new_points):
    for i in range(len(new_points)):
        if all(new_points[i] == [-1, -1]):
            print(new_points)
            new_points[i] = old_points[i]
            print(new_points)
            print("------------")


def pre_process(img_data):
    for i in range(len(img_data) - 1):
        replace_invalid_points(img_data[i], img_data[i + 1])


def check_points(data):
    for i in range(1, len(data)):
        for j in range(len(data[i])):
            if all(data[i][j] == [-1, -1]):
                print(i)
                print("error")


if __name__ == '__main__':
    if not os.path.exists("output/processed_npy"):
        os.mkdir("output/processed_npy")

    filenames = os.listdir("output/npy")
    filenames.sort()
    for file_path in filenames:
        file = os.path.splitext(file_path)
        filename, file_type = file
        if file_type == ".npy":
            print(file)
            data = load_data("output/npy/" + file_path)
            pre_process(data)
            check_points(data)
            new_filename = filename + "_new"
            new_path = os.path.join("output/processed_npy", new_filename)
            np.save(new_path, data)
