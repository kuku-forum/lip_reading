import numpy as np
import cv2
import dlib
import time

import layer
from prettytable import PrettyTable


# label_word = ['delta' ,'oscar','india','alpha','hotel','lima','echo','mike','papa','november','quebec',
#               'whiskey', 'kilo','romeo','sierra','yankee','bravo','victor','juliet','charlie','golf',
#               'uniform','xray', 'foxtrot','tango','zulu']
#
# alphabet = ['D', 'O', 'I', 'A', 'H', 'L', 'E', 'M', 'P', 'N', 'Q', 'W', 'K',
#             'R', 'S', 'Y', 'B', 'V', 'J', 'C', 'G', 'U', 'X', 'F', 'T', 'Z']


label_word = ['alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel','india','juliet','kilo',
              'lima', 'mike','november','oscar','papa','quebec','romeo','sierra','tango','uniform',
              'victor', 'whiskey', 'xray','yankee','zulu']

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

weight_path = "weight path .hdf5" ## REAL DATASET

select_model = layer.Select_model()
predict_model = layer.Select_model().model_LT5()
predict_compile = select_model.predict_compile(predict_model)
predict_compile.load_weights(weight_path)

FPS = 30
FRAME_ROWS = 120
FRAME_COLS = 120

predictor_path = '.\\shape_predictor_68_face_landmarks.dat '

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def process_video(frame, pre_count, pre_detection):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Required parameters for mouth extraction.
    width_crop_max = 0
    height_crop_max = 0

    # Detection of the frames
    if pre_count % 10 == 0:
        start_dectector = time.time()
        detections = detector(frame, 1)
        # print('detector time', time.time() - start_dectector)
    else:
        detections = pre_detection

    # 20 mark for mouth
    marks = np.zeros((2, 20))
    mouth_gray = np.zeros((120, 120))

    # All unnormalized face features.
    if len(detections) > 0:
        for k, d in enumerate(detections):

            # Shape of the face.
            shape = predictor(frame, d)
            co = 0
            # Specific for the mouth.
            for ii in range(48, 68):
                X = shape.part(ii)
                # A = (X.x, X.y)
                marks[0, co] = X.x
                marks[1, co] = X.y
                co += 1

            # Get the extreme points(top-left & bottom-right)
            X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]), int(np.amin(marks, axis=1)[1]),
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

            # # # Uncomment if the lip area is desired to be rectangular # # # #
            # Find the cropping points(top-left and bottom-right).
            X_left_crop = int(X_center - width_crop_max / 2.0)
            X_right_crop = int(X_center + width_crop_max / 2.0)
            Y_left_crop = int(Y_center - height_crop_max / 2.0)
            Y_right_crop = int(Y_center + height_crop_max / 2.0)

            if X_left_crop >= 0 and Y_left_crop >= 0 and X_right_crop < w and Y_right_crop < h:
                mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :]

                # Save the mouth area.
                mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                mouth_gray = cv2.resize(mouth_gray, (FRAME_COLS, FRAME_ROWS))

            else:
                cv2.putText(frame, 'The full mouth is not detectable. ', (30, 30), font, 1, (0, 255, 255), 2)

            cv2.rectangle(frame, (X_left_crop, Y_left_crop), (X_right_crop, Y_right_crop), (0, 255, 0), 2)

    return frame, mouth_gray, detections


def pred_table(pred):
    pred = np.squeeze(pred, 0)
    pred_sort = pred.argsort()[-3:][::-1]

    table = PrettyTable()
    table_rank = PrettyTable()

    table.field_names = label_word
    table.add_row(pred)

    table_rank.field_names = ['Top1', 'Top2', 'Top3']
    table_rank.add_row((label_word[pred_sort[0]], label_word[pred_sort[1]], label_word[pred_sort[2]]))

    print(table)
    print(table_rank)


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("video_fps", video_fps)
    assert FPS == video_fps, "video FPS is not 25 Hz"
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('width :', w)
    print('height :', h)

    count = 0
    dp_triger = False
    pre_count = 0
    pre_detection = 0

    tri_switch = False
    lip_reading = False
    put_predict_word = False
    operation_key = False

    prevTime = 0  # 이전 시간을 저장할 변수

    test_img_4D = np.zeros([25, 120, 120, 5])
    test_img_3D = np.zeros([120, 120, 29])

    while True:
        strat_while = time.time()
        ret, frame_main = cap.read()

        if ret == 0:
            print("webcam is not finded")
            break

        frame = cv2.cvtColor(frame_main, cv2.COLOR_BGR2RGB)
        prevTime1 = 0
        # strat_preprocessing = time.time()
        canvas, mouth_frame, pre_detection = process_video(frame, pre_count, pre_detection)
        # print('preprocessing time', time.time()-strat_preprocessing)
        canvas = cv2.flip(canvas, 1)
        pre_count += 1

        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / (sec)

        put_fps = "FPS : %0.1f" % fps
        cv2.putText(canvas, put_fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if tri_switch and count < 29:

            if np.sum(mouth_frame) == 0:
                print('pass')
                pass
            else:
                mouth_img = mouth_frame / 255.
                test_img_3D[:, :, count] = mouth_img
                count += 1
                put_count = "count : %0.1f" % count
                # if count % 5 == 0:
                    # print(count)
                cv2.putText(canvas, put_count, (420, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 255, 20), 2)

        if count == 29:
            for ii in range(25):
                test_img_4D[ii, :, :, :] = test_img_3D[:, :, ii:ii + 5]

            print('done')
            count = 0
            tri_switch = False

        if lip_reading:
            process_data = np.expand_dims(test_img_4D, axis=0)
            start_lipreading_time = time.time()
            y_predict = predict_compile.predict(process_data)
            # print('lip_reading time', time.time() - start_lipreading_time)

            y_predict = np.round(y_predict * 100, 2)

            result = label_word[np.argmax(y_predict)]

            if dp_triger:
                result_convert = result_convert + alphabet[np.argmax(y_predict)]
            else:
                result_convert = alphabet[np.argmax(y_predict)]
                dp_triger = True

            pred_table(y_predict)

            put_predict_word = True
            lip_reading = False

        if put_predict_word:
            cv2.putText(canvas, result, (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (135, 0, 255), 2)

        if operation_key:
            cv2.putText(canvas, result_convert, (280, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (135, 0, 255), 2)

        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        cv2.imshow('window', canvas)

        input_key = cv2.waitKey(1)

        if input_key == ord('q'):

            print('exit')
            break

        elif input_key == ord('z'):
            print('preprocessing')
            tri_switch = True
            put_predict_word = False
            test_img_4D = np.zeros([25, 120, 120, 5])
            test_img_3D = np.zeros([120, 120, 29])
            pre_count = 0

        elif input_key == ord('x'):
            print('lip reading')
            lip_reading = True
            operation_key = True

        elif input_key == ord('c'):
            print('reset \n')
            dp_triger = False
            operation_key = False
            put_predict_word = False
            result_convert = []
        # print('while time', time.time() - strat_while)

    cap.release()
    cv2.destroyAllWindows()
