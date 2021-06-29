
# Usage : python lstm_featureExtractor.py -vp path-to-video-files/ -sp path-to-shape-predictor-68-face-landmarks-dat -o output-file
# Use : pip install opencv-python==3.4.2.16

import cv2
import dlib
import imageio
from tqdm import tqdm
import numpy as np
import os

from numpy import savez_compressed
from copy import deepcopy

predictor_path = './shape_predictor_68_face_landmarks.dat'
video_folder_path = 'video dataset path'
video_ouput_path = 'save path/'


def frame_concat(frame_ori):

    global delta_1, delta_2
    frame_len_ori = len(frame_ori[:, 0, 0]) # 29
    for i in range(frame_len_ori):

        if i == 0:
            delta_1 = np.expand_dims((frame_ori[i + 1, :, :] - frame_ori[i, :, :]), axis=0)
        elif i < frame_len_ori-2:
            delta_1 = np.concatenate((delta_1, np.expand_dims((frame_ori[i + 1, :, :] - frame_ori[i, :, :]), axis=0)),
                                     axis=0)

    frame_len_1 = len(delta_1[:, 0, 0])  # 29
    for i in range(frame_len_1):
        if i == 0:
            delta_2 = np.expand_dims((delta_1[i + 1, :, :] - delta_1[i, :, :]), axis=0)
        elif i < frame_len_1 - 2:
            delta_2 = np.concatenate((delta_2, np.expand_dims((delta_1[i + 1, :, :] - delta_1[i, :, :]), axis=0)),
                                     axis=0)

    delta_1 = np.concatenate((delta_1, np.zeros((2, 68, 2))), axis=0)
    delta_2 = np.concatenate((delta_2, np.zeros((4, 68, 2))), axis=0)
    delta_out = np.concatenate((frame_ori, delta_1, delta_2), axis=2)

    return delta_out


def window_slinding(input_3d, size):
    input_4d = np.zeros(size)
    input_3d = np.transpose(input_3d, (1, 2, 0))

    for ii in range(25):
        input_4d[ii, :, :, :] = input_3d[:, :, ii:ii + 5]

    return input_4d


def min_max(A):
    return (A - A.min()) / (A.max() - A.min())


def face_roi_npz_saved():
    detector = dlib.get_frontal_face_detector()

    for roots, dirnames, filenames in tqdm(list(os.walk(video_folder_path))):
        for filename in filenames:

            if filename.split('.')[-1] == 'mp4':
                file_path = os.path.join(roots, filename)

                point_seq = []
                break_singnal = False

                save_path = os.path.join(video_ouput_path,
                                         file_path.split('/')[-1].split('\\')[0],
                                         file_path.split('/')[-1].split('\\')[1])

                save_path_name = os.path.join(video_ouput_path,
                                              file_path.split('/')[-1].split('\\')[0],
                                              file_path.split('/')[-1].split('\\')[1],
                                              file_path.split('/')[-1].split('\\')[-1] + '.npz')

                if os.path.isfile(save_path_name):
                    continue

                vid = imageio.get_reader(file_path, 'ffmpeg')
                cnt = 0
                for frm_cnt in range(0, vid.count_frames()):

                    try:
                        img = vid.get_data(frm_cnt)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    except:
                        print('FRAME EXCEPTION!!')
                        break_singnal = True
                        break

                    faces = detector(img)  # rectangles[[(46, 30) (201, 184)]]

                    if len(faces) != 1:
                        print('FACE DETECTION FAILED!!')
                        break_singnal = True
                        break

                    for k, f in enumerate(faces):
                        # shape = predictor(img, f)
                        crop = img[f.top(): f.bottom(), f.left():f.right()]

                    try:
                        crop = cv2.resize(crop, (120, 120))
                    except Exception as e:
                        print('err', e)
                        print('name', file_path)
                        continue

                    # cv2.imwrite(video_ouput_path + '{0:04}.jpg'.format(cnt), crop)
                    crop_minmax = min_max(crop)
                    point_seq.append(crop_minmax)
                    frames = np.array(point_seq)
                    cnt += 1

                if break_singnal:
                    pass
                else:
                    if len(frames) != 29:
                        continue
                    frames_slidng = window_slinding(input_3d=frames, size=(25, 120, 120, 5))

                    try:
                        savez_compressed(save_path_name, frames_slidng)
                    except:
                        os.makedirs(save_path, exist_ok=True)
                        savez_compressed(save_path_name, frames_slidng)

def lip_roi_npz_saved():
    global crop
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    marks = np.zeros((2, 20))
    width_crop_max = 0
    height_crop_max = 0

    for roots, dirnames, filenames in tqdm(list(os.walk(video_folder_path))):
        for filename in filenames:

            if filename.split('.')[-1] == 'mp4':
                file_path = os.path.join(roots, filename)
                print(file_path)

                point_seq = []
                break_singnal = False

                save_path = os.path.join(video_ouput_path,
                                         file_path.split('/')[-1].split('\\')[0],
                                         file_path.split('/')[-1].split('\\')[1])

                save_path_name = os.path.join(video_ouput_path,
                                              file_path.split('/')[-1].split('\\')[0],
                                              file_path.split('/')[-1].split('\\')[1],
                                              file_path.split('/')[-1].split('\\')[-1] + '.npz')

                if os.path.isfile(save_path_name):
                    continue

                vid = imageio.get_reader(file_path, 'ffmpeg')
                cnt = 0
                for frm_cnt in range(0, vid.count_frames()):

                    try:
                        img = vid.get_data(frm_cnt)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    except:
                        print('FRAME EXCEPTION!!')
                        break_singnal = True
                        break

                    faces = detector(img)  # rectangles[[(46, 30) (201, 184)]]

                    if len(faces) != 1:
                        print('FACE DETECTION FAILED!!')
                        break_singnal = True
                        break

                    for k, f in enumerate(faces):
                        shape = predictor(img, f)

                        co = 0
                        # Specific for the mouth.
                        for ii in range(48, 68):
                            X = shape.part(ii)
                            A = (X.x, X.y)
                            marks[0, co] = X.x
                            marks[1, co] = X.y
                            co += 1

                        # Get the extreme points(top-left & bottom-right)
                        X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]),
                                                            int(np.amin(marks, axis=1)[1]),
                                                            int(np.amax(marks, axis=1)[0]),
                                                            int(np.amax(marks, axis=1)[1])]

                        # Find the center of the mouth.
                        X_center = (X_left + X_right) / 2.0
                        Y_center = (Y_left + Y_right) / 2.0

                        # Make a boarder for cropping.
                        border = 30
                        X_left_new = X_left - border
                        Y_left_new = Y_left - border
                        X_right_new = X_right + border
                        Y_right_new = Y_right + border

                        # Width and height for cropping(before and after considering the border).
                        width_new = X_right_new - X_left_new
                        height_new = Y_right_new - Y_left_new
                        width_current = X_right - X_left
                        height_current = Y_right - Y_left

                        # Determine the cropping rectangle dimensions(the main purpose is to have a fixed area).
                        if width_crop_max == 0 and height_crop_max == 0:
                            width_crop_max = width_new
                            height_crop_max = height_new
                        else:
                            width_crop_max += 1.5 * np.maximum(width_current - width_crop_max, 0)
                            height_crop_max += 1.5 * np.maximum(height_current - height_crop_max, 0)

                        X_left_crop = int(X_center - width_crop_max / 2.0)
                        X_right_crop = int(X_center + width_crop_max / 2.0)
                        Y_left_crop = int(Y_center - height_crop_max / 2.0)
                        Y_right_crop = int(Y_center + height_crop_max / 2.0)
                        crop = img[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop]

                    try:
                        crop = cv2.resize(crop, (120, 120))
                    except Exception as e:
                        print('err', e)
                        print('name', file_path)
                        continue

                    # cv2.imwrite(video_ouput_path + '{0:04}.jpg'.format(cnt), crop)
                    crop_minmax = min_max(crop)
                    point_seq.append(crop_minmax)
                    frames = np.array(point_seq)
                    cnt += 1

                if break_singnal:
                    pass
                else:
                    if len(frames) != 29:
                        continue
                    frames_slidng = window_slinding(input_3d=frames, size=(25, 120, 120, 5))

                    try:
                        savez_compressed(save_path_name, frames_slidng)
                    except:
                        os.makedirs(save_path, exist_ok=True)
                        savez_compressed(save_path_name, frames_slidng)


def landmark_npz_saved():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for roots, dirnames, filenames in os.walk(video_folder_path):
        for filename in reversed(filenames):

            if filename.split('.')[-1] == 'mp4' :
                file_path = os.path.join(roots, filename)
                print('\n', file_path)

                point_seq = []
                break_singnal = False

                save_path = os.path.join(video_ouput_path,
                                         file_path.split('/')[-1].split('\\')[0],
                                         file_path.split('/')[-1].split('\\')[1])

                save_path_name = os.path.join(video_ouput_path,
                                              file_path.split('/')[-1].split('\\')[0],
                                              file_path.split('/')[-1].split('\\')[1],
                                              file_path.split('/')[-1].split('\\')[-1] + '.npz')

                print(save_path_name)

                if os.path.isfile(save_path_name):
                    continue

                vid = imageio.get_reader(file_path, 'ffmpeg')
                for frm_cnt in tqdm(range(0, vid.count_frames())):
                    points = np.zeros((68, 2), dtype=np.float32)

                    try:
                        img = vid.get_data(frm_cnt)
                    except:
                        print('FRAME EXCEPTION!!')
                        break_singnal = True
                        break

                    dets = detector(img, 1)  # rectangles[[(46, 30) (201, 184)]]
                    # print(dets)

                    if len(dets) != 1:
                        print('FACE DETECTION FAILED!!')
                        break_singnal = True
                        break

                    for k, d in enumerate(dets):
                        # print('d', d) # rectangles[[(46, 30) (201, 184)]]
                        shape = predictor(img, d)

                        for i in range(68):
                            points[i, 0] = shape.part(i).x
                            points[i, 1] = shape.part(i).y

                    point_seq.append(deepcopy(points))
                    frames = np.array(point_seq)

                if break_singnal:
                    pass
                else:
                    print(frames.shape)
                    if len(frames) != 29:
                        continue
                    frames_delta = frame_concat(frames)
                    frames_slidng = window_slinding(input_3d=frames_delta, size=(25, 68, 6, 5))
                    print('frames_slidng', frames_slidng.shape)

                    try:
                        savez_compressed(save_path_name, frames_slidng)
                    except:
                        os.makedirs(save_path, exist_ok=True)
                        savez_compressed(save_path_name, frames_slidng)
                    # 'E:/LRW/LANDMARK_EXTRACT/LRW_video-26-feature/'

if __name__ == "__main__":
    print('start')
    landmark_npz_saved()
