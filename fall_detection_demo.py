import time

import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.engine.saving import load_model
import send_msg

from send_msg import User

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

global proto_file, weights_file, POSE_PAIRS

new_position = 0
old_position = -1
norm_length = 0
cos = 0.5 ** 0.5

# 0 for standing, 5 for falling
status = 0

MODE = "COCO"

if MODE is "COCO":
    proto_file = "./models/pose/mpi/pose_deploy_linevec.prototxt"
    weights_file = "./models/pose/coco/pose_iter_440000.caffemodel"

    # Body Parts attr（omit background: 18)
    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

elif MODE is "MPI":
    proto_file = "./models/pose/mpi/pose_deploy_linevec.prototxt"
    weights_file = "./models/pose/mpi/pose_iter_160000.caffemodel"

    # Body Parts attr （omit background: 15)
    BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14}

    POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                  ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                  ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

output_path = "./output/fall_output_mytest"

data_set = []

fps = 30


# Should change some settings if needed
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg', default="./dataset/my_record/mypic2",
                        help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--test', default="./dataset/my_record/mypic2")
    parser.add_argument('--thr', default=0.11, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    parser.add_argument('--mode', default="fall", help='Choose the modes.')
    return parser.parse_args()


# Load trained network
def load_network():
    network = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    return network


# Get the square of certain person from detected pose points
def get_square(points_array, origin_frame):
    frame_width, frame_height = origin_frame.shape[0], origin_frame.shape[1]
    x1, x2 = int(max(points_array[:, 0]) * frame_width / 100), int(min(points_array[:, 0]) * frame_width / 100)
    y1, y2 = int(max(points_array[:, 1]) * frame_height / 100), int(min(points_array[:, 1]) * frame_height / 100)
    if x2 < 0:
        sort_x = sorted(points_array[:, 0])
        for i in sort_x:
            if i >= 0:
                x2 = int(i * frame_width / 100)
                break

    if y2 < 0:
        sort_y = sorted(points_array[:, 1])
        for i in sort_y:
            if i >= 0:
                y2 = int(i * frame_height / 100)
                break

    if x2 < 0 or y2 < 0 or x1 == x2 or y1 == y2:
        return (x1, y1), (x1, y2), (x2, y1), (x2, y2), 0
    else:
        return (x1, y1), (x1, y2), (x2, y1), (x2, y2), 1


# Get the width height ratio
def get_width_height_ratio(p1, p2, p3):
    return (p1[0] - p3[0]) / (p1[1] - p2[1])


# Process video frame
def process_video_frame(args):
    mode = input("Enter 1 to change medianFrame, or enter 2 to use the previous one:\n")
    if not os.path.exists("./output/medianFrame.png") or mode == '1':
        get_median_frame(args)
    global fps
    sub = cv2.createBackgroundSubtractorMOG2(history=2 * fps, varThreshold=2, detectShadows=True)

    initFrame = cv2.imread("./output/medianFrame.png")
    for i in range(10):
        sub.apply(initFrame)

    frame_count = 0
    model_based_detected_status = 0
    detected_frames = []
    prev_points_array = []

    trained_model = load_model('./models/my_trained_model/fall_model.h5')

    cap = cv2.VideoCapture('dataset/fall.mov')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_w = int(cap.get(3))
    frame_h = int(cap.get(4))
    global fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_w, frame_h))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            dots = get_mask_dots_sub(frame, sub)

            rect = cv2.minAreaRect(dots)
            box = np.int0(cv2.boxPoints(rect))

            O_point, crop_frame, box = get_crop_frame(box, frame)

            time_start = time.time()
            points_array, valid = apply_openpose(args, crop_frame, frame, O_point)
            time_end = time.time()
            print('cost: ', time_end - time_start)

            label = model_based_predict(detected_frames, frame_count, points_array, trained_model, valid, prev_points_array)

            if label == 0:
                model_based_detected_status += 1
            elif label == 1:
                model_based_detected_status += 4
            elif label == 2 and model_based_detected_status > 0:
                model_based_detected_status -= 1

            if model_based_detected_status >= 1.5 * fps:
                suspected_img_path = "./output/suspected_img.png"
                cv2.imwrite(suspected_img_path, frame)
                send_msg.send_email(user, suspected_img_path)
                exit(0)

            rule_based_predict(box, frame)
            out.write(frame)
        else:
            break

    cap.release()
    out.release()


def replace_invalid_points(old_points, new_points):
    for i in range(len(new_points)):
        if (new_points[i] == [-1, -1]).any():
            new_points[i] = old_points[i]


# Process image frame
def process_img_frame(args, user):
    mode = input("Enter 1 to change medianFrame, or enter 2 to use the previous one:\n")
    if not os.path.exists("./output/medianFrame.png") or mode == '1':
        get_median_frame(args)

    sub = cv2.createBackgroundSubtractorMOG2(history=2 * fps, varThreshold=2, detectShadows=True)

    initFrame = cv2.imread("./output/medianFrame.png")
    for i in range(10):
        sub.apply(initFrame)

    filenames = os.listdir(args.test)
    filenames.sort()
    test_path = args.test
    dirs = test_path.split("/")
    out_path = os.path.join(output_path, dirs[-1])
    frame_count = 0
    model_based_detected_status = 0
    detected_frames = []
    prev_points_array = []

    trained_model = load_model('./models/my_trained_model/fall_model.h5')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for file in filenames:
        imgPath = os.path.join(args.test, file)
        print(imgPath)
        frame = cv2.imread(imgPath)

        dots = get_mask_dots_sub(frame, sub)

        rect = cv2.minAreaRect(dots)
        box = np.int0(cv2.boxPoints(rect))

        O_point, crop_frame, box = get_crop_frame(box, frame)

        time_start = time.time()
        points_array, valid = apply_openpose(args, crop_frame, frame, O_point)
        time_end = time.time()
        print('cost: ', time_end - time_start)

        label = model_based_predict(detected_frames, frame_count, points_array, trained_model, valid, prev_points_array)

        if label == 0:
            model_based_detected_status += 1
        elif label == 1:
            model_based_detected_status += 4
        elif label == 2 and model_based_detected_status > 0:
            model_based_detected_status -= 1

        if model_based_detected_status >= 1.5 * fps:
            suspected_img_path = "./output/suspected_img.png"
            cv2.imwrite(suspected_img_path, frame)
            send_msg.send_email(user, suspected_img_path)
            exit(0)
        rule_based_predict(box, frame)

        cv2.imwrite(os.path.join(out_path, file), frame)

    data_set_np = np.array(data_set)
    return data_set_np


def rule_based_predict(box, frame):
    out_data = fall_detect()
    global old_position
    old_position = new_position
    draw_position(out_data, frame)
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)


def model_based_predict(detected_frames, frame_count, points_array, trained_model, valid, prev_points_array):
    if frame_count == 0:
        prev_points_array = points_array
        frame_count += 1

    replace_invalid_points(prev_points_array, points_array)
    prev_points_array = points_array

    if valid != 0:
        detected_frames.append(points_array)
    else:
        print("No Person Points Detected.")

    if len(detected_frames) > 10:
        detected_frames.pop(0)
        detected_frames_array = np.array(detected_frames)
        detected_frames_array = detected_frames_array.reshape(1, 10, detected_frames_array.shape[1] *
                                                              detected_frames_array.shape[2])
        pred = trained_model.predict(detected_frames_array)
        label = np.argmax(pred)
        print(pred, label)
        return label
    return -1


def get_range(frames, medianFrame):
    old_frame = frames[0]
    divs = []
    for i in range(1, len(frames)):
        dframe = cv2.absdiff(frames[i], old_frame)
        old_frame = frames[i]
        divs.append(dframe)
    divMedianFrame = np.median(divs, axis=0).astype(dtype=np.uint8)
    maxFrame = medianFrame + divMedianFrame * 9
    minFrame = medianFrame - divMedianFrame * 9
    return maxFrame, minFrame


def get_mask_dots(frame, max_frame, min_frame):
    (b, g, r) = cv2.split(frame)
    (b_max, g_max, r_max) = cv2.split(max_frame)
    (b_min, g_min, r_min) = cv2.split(min_frame)
    b_mask = cv2.inRange(b, b_min, b_max)
    g_mask = cv2.inRange(g, g_min, g_max)
    r_mask = cv2.inRange(r, r_min, r_max)
    mask = cv2.merge([b_mask, g_mask, r_mask])
    mask = 255 - mask
    thr, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (8, 8))
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)
    kernel_size = int(frame.shape[0] / 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    mask = cv2.erode(mask, None, iterations=2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    mask = cv2.erode(mask, None, iterations=5)
    mask = cv2.dilate(mask, None, iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    plt.imshow(frame)
    plt.imshow(mask, alpha=0.6)
    plt.show()
    _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        dots = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    else:
        dots = (0, 0)
    return dots


global old_dots


# Process and show the mask of foreground
def get_mask_dots_sub(frame, sub):
    mask = get_sub_mask(frame, sub)
    plt.imshow(frame)
    plt.imshow(mask, alpha=0.6)
    plt.show()
    _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    global old_dots
    if len(cnts) > 0:
        dots = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        old_dots = dots
    else:
        dots = old_dots
    return dots


# Get the mask of foreground
def get_sub_mask(frame, sub):
    mask = sub.apply(frame)
    # thr, mask = cv2.threshold(fgmask.copy(), 128, 255, cv2.THRESH_BINARY)
    blurred = cv2.blur(mask.copy(), (8, 8))

    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

    kernel_size = int(frame.shape[0] / 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, None, iterations=5)
    mask = cv2.dilate(mask, None, iterations=6)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


# Get the crop frame
def get_crop_frame(box, frame):
    xs = [i[0] for i in box]
    ys = [i[1] for i in box]
    x1 = abs(min(xs))
    x2 = abs(max(xs))
    y1 = abs(min(ys))
    y2 = abs(max(ys))
    height = y2 - y1
    width = x2 - x1
    area_ratio = height * width / (frame.shape[0] * frame.shape[1])
    # print(area_ratio)
    if area_ratio < 1 / 30:
        x1, y1, height, width = 0, 0, frame.shape[0], frame.shape[1]
        rect = ((0, frame.shape[1]),
                (0, 0),
                (frame.shape[0], 0),
                (frame.shape[0], frame.shape[1]))

        box = np.array(rect)
    crop_frame = frame[y1:y1 + height, x1:x1 + width]

    O_point = [x1, y1]
    return O_point, crop_frame, box


# Get the median frame of background
def get_median_frame(args):
    filenames = os.listdir(args.bg)
    filenames.sort()
    frames = []
    for file in filenames:
        imgPath = os.path.join(args.bg, file)
        frame = cv2.imread(imgPath)
        print(imgPath)
        frames.append(frame)
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    cv2.imwrite("./output/medianFrame.png", medianFrame)
    return medianFrame, frames


# Detect whether the angle is out of the max angle(45°)
def out_of_max_angle():
    return 1 if (cos > 0.5 ** 0.5 or cos < -0.5 ** 0.5) else 0


# Label the position
def draw_position(out_data, frame):
    global status
    fall_thr = 3.0
    up_thr = 2.8
    try:
        if out_data > up_thr:
            if status > 0:
                status -= 2 if out_data > 6 and status > 1 else 1
            if status < 2:
                cv2.putText(frame, "Up " + "{}".format(out_data) + " " + "{}".format(status), (5, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif out_data < -fall_thr and 3 > status >= 0:
            status += 1
            cv2.putText(frame, "Fall(P) " + "{}".format(out_data) + " " + "{}".format(status), (5, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif out_data < -fall_thr and status < 5:
            status += 2 if out_data < -6 and status < 4 else 1
            cv2.putText(frame, "Fall " + "{}".format(out_data) + " " + "{}".format(status), (5, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif status > 4:
            cv2.putText(frame, "Fall " + "{}".format(out_data) + " " + "{}".format(status), (5, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif out_of_max_angle() and status > 0:
            if status < 5:
                status += 1
                cv2.putText(frame, "Fall(P) " + "{}".format(out_data) + " " + "{}".format(status), (5, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            else:
                cv2.putText(frame, "Fall " + "{}".format(out_data) + " " + "{}".format(status), (5, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        elif status > 0:
            cv2.putText(frame, "Fall(P) " + "{}".format(out_data) + " " + "{}".format(status), (5, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        else:
            cv2.putText(frame, "Stand " + "{}".format(out_data) + " " + "{}".format(status), (5, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    except:
        cv2.putText(frame, "Stand " + "{}".format(out_data) + " " + "{}".format(status), (5, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv2.putText(frame, "COS: " + "{}".format(cos), (5, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.putText(frame, "normal_len: " + "{}".format(norm_length), (5, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))


count = 0


def fall_detect():
    global count
    if old_position == -1:
        move_distance = 0
    else:
        move_distance = new_position - old_position

    move_per_sec = move_distance * (fps - count)

    if move_distance == 0:
        count += 1
    else:
        count = 0

    try:
        out = move_per_sec / norm_length
        return -out
    except:
        pass


def apply_openpose(args, frame, origin_frame, O_point):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    in_width = int((args.height / frame_height) * frame_width)
    in_height = args.height
    time1 = time.time()
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)
    out = net.forward()
    time2 = time.time()
    print("op cost: ", time2 - time1)
    out = out[:, :len(BODY_PARTS), :, :]
    H = out.shape[2]
    W = out.shape[3]
    # Empty list to store the detected points
    points = []
    points_normal = []
    for i in range(len(BODY_PARTS)):
        # Confidence map of body parts.
        prob_map = out[0, i, :, :]

        # Find global maxima of the prob_map.
        minVal, prob, minLoc, point = cv2.minMaxLoc(prob_map)

        # Scale the point to fit on the original image
        x = (frame_width * point[0]) / W + O_point[0]
        y = (frame_height * point[1]) / H + O_point[1]

        x_normal = x / origin_frame.shape[0] * 100
        y_normal = y / origin_frame.shape[1] * 100

        if prob > args.thr:
            cv2.circle(origin_frame, (int(x), int(y)), 10, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(origin_frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # Add the point to the list if prob is greater than the threshold
            points.append((int(x), int(y)))
            points_normal.append((int(x_normal), int(y_normal)))
        else:
            points.append((-1, -1))
            points_normal.append((-1, -1))

    for pair in POSE_PAIRS:
        partA = BODY_PARTS[pair[0]]
        partB = BODY_PARTS[pair[1]]
        if points[partA] and points[partB] and points[partA] != (-1, -1) and points[partB] != (-1, -1):
            cv2.line(origin_frame, points[partA], points[partB], (0, 255, 0), 2)

    points_array = np.array(points_normal)
    # data_set.append(points_array)
    p1, p2, p3, p4, valid = get_square(points_array, origin_frame)
    if valid == 0:
        return points_array, valid
    if points[8] != (-1, -1) and points[1] != (-1, -1):
        global new_position, norm_length, cos
        new_position = points_normal[1][1]
        a = np.array(points_normal[1])
        b = np.array(points_normal[8])
        norm_length = np.linalg.norm(a - b)
        c = a - b
        d = np.array([1, 0])
        cos = c.dot(d) / (np.linalg.norm(c) * np.linalg.norm(d))

    cv2.line(origin_frame, p1, p2, (255, 0, 0), 2)
    cv2.line(origin_frame, p1, p3, (255, 0, 0), 2)
    cv2.line(origin_frame, p2, p4, (255, 0, 0), 2)
    cv2.line(origin_frame, p3, p4, (255, 0, 0), 2)

    return points_array, valid


if __name__ == '__main__':
    args = parse()
    net = load_network()
    output = "./output"
    if not os.path.exists(output):
        os.mkdir(output)
    user = send_msg.load_user_account()

    data_to_save = process_img_frame(args, user)
    # process_video_frame(args)
    path = args.test
    dirs = path.split("/")
    np_path = os.path.join(output, dirs[-1])
    np.save(np_path, data_to_save)
    cv2.destroyAllWindows()
