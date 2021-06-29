import datetime
import layer
import numpy as np
import os
import random
import tensorflow as tf

from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras.utils.np_utils import to_categorical
from keras.backend.tensorflow_backend import set_session


def set_gpu_option(which_gpu):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = which_gpu
    set_session(tf.Session(config=config))
    return


class AUX(object):
    def __init__(self, select_root_path):
        self.ROOT_PATH = select_root_path
        self.OUTPUT_DIR = os.path.realpath(os.path.join(self.ROOT_PATH, 'data', 'res'))
        self.LOG_DIR = os.path.realpath(os.path.join(self.ROOT_PATH, 'data', 'res_logs'))

    def is_dir(self, path: str):
        return isinstance(path, str) and os.path.exists(path) and os.path.isdir(path)

    def make_dir_if_not_exists(self, path: str):
        if not self.is_dir(path): os.makedirs(path)

    def create_callbacks(self, run_name: str):
        run_log_dir = os.path.join(self.LOG_DIR, run_name)
        self.make_dir_if_not_exists(run_log_dir)

        # Tensorboard
        tensorboard = TensorBoard(log_dir=run_log_dir)

        # Training logger
        csv_log = os.path.join(run_log_dir, 'training.csv')
        csv_logger = CSVLogger(csv_log, separator=',', append=True)

        # Model checkpoint saver
        checkpoint_dir = os.path.join(self.OUTPUT_DIR, run_name)
        self.make_dir_if_not_exists(checkpoint_dir)

        checkpoint_template = os.path.join(checkpoint_dir, "MAI_{epoch:03d}_{val_loss:.2f}_{val_acc:.2f}.hdf5")
        checkpoint = ModelCheckpoint(checkpoint_template, monitor='val_loss', save_weights_only=True, mode='auto',
                                     save_best_only=False, period=1, verbose=1)

        return [checkpoint, csv_logger, tensorboard]


class Data_input(object):

    def __init__(self, video_ouput_path):
        self.train_file_path = []
        self.val_file_path = []
        self.test_file_path = []

        self.label_dir = os.listdir(video_ouput_path)
        print(self.label_dir)
        self.temp_one_hot = []
        for i in range(len(self.label_dir)):
            self.temp_one_hot.append(i)

        self.data_label = dict(zip(self.label_dir, to_categorical(self.temp_one_hot)))

        for roots, dirnames, filenames in os.walk(video_ouput_path):
            for filename in filenames:
                file_path = os.path.join(roots, filename)

                selec_word = file_path.split('\\')[-3]
                flag = file_path.split('\\')[-2]

                if not selec_word in self.label_dir:
                    continue
                if flag == 'train':
                    self.train_file_path.append(file_path)
                elif flag == 'val':
                    self.val_file_path.append(file_path)
                elif flag == 'test':
                    self.test_file_path.append(file_path)

    ## select를 통해 train, val, test를 선정
    def generate_dataset(self, select, batch_size=1, input_size=(1, 25, 68, 9, 5)):

        global data_path, i_train
        while 1:
            concat_train_input = np.zeros(input_size)
            concat_train_label = np.zeros((1, len(self.label_dir)))

            if select == 'Train':
                data_path = self.train_file_path
                random.shuffle(data_path)
            elif select == 'Val':
                data_path = self.val_file_path
            elif select == 'Test':
                data_path = self.test_file_path
                data_path.sort()

            try:
                for i, i_train in enumerate(data_path):

                    train_npy = np.load(i_train)['arr_0']
                    train_npy = np.expand_dims(train_npy, axis=0)
                    concat_train_input = np.concatenate([concat_train_input, train_npy])

                    train_label = np.expand_dims(self.data_label[i_train.split('\\')[-3]], axis=0)
                    concat_train_label = np.concatenate([concat_train_label, train_label])

                    if (i + 1) % batch_size == 0:
                        concat_train_input = np.delete(concat_train_input, [0], axis=0)
                        concat_train_label = np.delete(concat_train_label, [0], axis=0)

                        yield concat_train_input, concat_train_label
                        concat_train_input = np.zeros(input_size)
                        concat_train_label = np.zeros((1, len(self.label_dir)))
            except Exception as inst:
                print('value Error : {}'.format(i_train))
                print(inst)
                continue


def main_train(select_root_path, video_ouput_path, data_input_size, model_input_size, batch_size):
    pre = Data_input(video_ouput_path)
    aux = AUX(select_root_path)
    select_model = layer.Select_model()
    train_model = layer.Select_model().model_ROI(class_num=len(pre.label_dir), input_size=model_input_size)  # lm practice
    train_compile = select_model.train_compile(train_model)

    name = datetime.datetime.now().strftime(video_ouput_path.split('\\')[-2].split('_')[1] + '-%Y-%m-%d-%H-%M')
    callbacks = aux.create_callbacks(name)

    history = train_compile.fit_generator(generator=pre.generate_dataset(select='Train', batch_size=batch_size,
                                                                         input_size=data_input_size),
                                          validation_data=pre.generate_dataset(select='Val', batch_size=batch_size,
                                                                               input_size=data_input_size),
                                          steps_per_epoch=len(pre.train_file_path) / batch_size,
                                          epochs=500,
                                          validation_steps=len(pre.val_file_path) / batch_size,
                                          verbose=1,
                                          callbacks=callbacks,
                                          shuffle=True,
                                          max_queue_size=10,
                                          workers=16)

    return history


def main_predict(weight_path, video_ouput_path, data_input_size, model_input_size):
    pre = Data_input(video_ouput_path)
    select_model = layer.Select_model()
    train_model = layer.Select_model().model_ROI(class_num=len(pre.label_dir), input_size=model_input_size)  # lm practice
    predict_compile = select_model.predict_compile(train_model)
    predict_compile.load_weights(weight_path)

    test_file_list = pre.test_file_path
    val_file_list = pre.val_file_path

    print("-- Predict --")
    print("-- test len :", len(test_file_list))
    print("-- val len :", len(val_file_list))
    batch_size = 5

    scores = predict_compile.evaluate_generator(pre.generate_dataset(select='Val',
                                                                     batch_size=batch_size,
                                                                     input_size=data_input_size),
                                                steps=len(val_file_list)/batch_size)

    print("%s: %.2f%%" % (predict_compile.metrics_names[1], scores[1] * 100))  # top1


if __name__ == "__main__":
    set_gpu_option("0")

    ## select_root_path : root 경로이며, 해당 경로를 통해 Weight 파일, log 파일 저장 위치 생성
    ## ex) select_root_path\\data\\res\\
    select_root_path = 'root path\\'

    ## video_ouput_path : 학습 비디오 경로
    video_ouput_path = 'video path\\'

    ## S001,s002 : validation data
    weight_path = 'weight path'
    # weight_path = 'E:\\LRW\\data\\res\\0_PRE_WEIGHT\\MAI_ROI_FACE_029_0.03_0.71.hdf5'

    # main_train(select_root_path=select_root_path,
    #            video_ouput_path=video_ouput_path,
    #            data_input_size=(1, 25, 120, 120, 5),
    #            model_input_size=(25, 120, 120, 5),
    #            batch_size=5)



    main_predict(weight_path=weight_path,
                 video_ouput_path=video_ouput_path,
                 data_input_size=(1, 25, 120, 120, 5),
                 model_input_size=(25, 120, 120, 5))
