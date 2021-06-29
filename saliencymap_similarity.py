import layer
import numpy as np
import os
import random
import keras.backend as K
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import operator

from keras.utils.np_utils import to_categorical
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
from itertools import combinations


def set_gpu_option(which_gpu):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = which_gpu
    set_session(tf.Session(config=config))
    return


set_gpu_option("0")


def cos_sim(A, B):
    A = A.reshape(-1)
    B = B.reshape(-1)

    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def euclidean_distance(pt1, pt2):
    pt1 = pt1.reshape(-1)
    pt2 = pt2.reshape(-1)
    distance = 0

    for i in range(len(pt1)):
        distance += (pt1[i] - pt2[i]) ** 2

    return distance ** 0.5


def z_score(A, delete_range):
    A = np.delete(A, delete_range, axis=1)
    A = (A - np.mean(A)) / np.std(A)
    return A


def saliencymap_func(model, x, index):
    global b
    grad = K.gradients(model.output[:, index], model.input)[0]
    iter = K.function([model.input], [grad])
    heatmap = iter([x])[0]
    heatmap = np.abs(heatmap)
    heatmap_reshape = processing_saliencymap(heatmap)

    return heatmap_reshape


def processing_saliencymap(saliency):
    rtn = np.zeros((29, len(saliency[0, :, 0, 0]), len(saliency[0, 0, :, 0])))
    for i in range(29):
        if i < 5:
            for j in range(i + 1):
                rtn[i] += saliency[j, :, :, i - j]
            rtn[i] /= i + 1
        elif i >= 24:
            for j in range(29 - i):
                rtn[i] += saliency[24 - j, :, :, 4 + j - (28 - i)]
            rtn[i] /= (28 - i) + 1
        else:
            for j in range(5):
                rtn[i] += saliency[i + j - 4, :, :, 4 - j]
            rtn[i] /= 5
    return rtn


class Data_input(object):

    def __init__(self, video_ouput_path):
        self.train_file_path = []
        self.val_file_path = []
        self.test_file_path = []

        self.label_dir = os.listdir(video_ouput_path)
        self.temp_one_hot = []
        for i in range(len(self.label_dir)):
            self.temp_one_hot.append(i)

        self.data_label = dict(zip(self.label_dir, to_categorical(self.temp_one_hot)))

        for roots, dirnames, filenames in os.walk(video_ouput_path):
            for filename in filenames:
                file_path = os.path.join(roots, filename)
                flag = file_path.split('\\')[-2]
                # print('flag {}'.format(flag))
                if flag == 'train':
                    self.train_file_path.append(file_path)
                elif flag == 'val':
                    self.val_file_path.append(file_path)
                elif flag == 'test':
                    self.test_file_path.append(file_path)

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
                # random.shuffle(data_path)
            elif select == 'Test':
                data_path = self.test_file_path

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
            except:
                print('value Error : {}'.format(i_train))
                continue


def save_heatmap(save_root, tmp_np, word_name, npz_path):
    heatmap_save = np.delete(tmp_np, 0, axis=0)
    np.savez_compressed(save_root + npz_path.split('\\')[-3] + '\\NPZ\\{}'.format(word_name),
                        heatmap_save)
    # tmp_np = np.zeros((1, 29, 38))
    print('save')

def main_heatmap_list_npz_saved(arg_count, weight_path, npz_path, dataset_path, save_root):
    np_feature_path = np.load(npz_path)['arr_0']
    word_list = os.listdir(dataset_path)
    word_list.sort()
    print('word_list', word_list) # alpha, bravo ...

    model_input_size = (25, 68, 6, 5)

    pre = Data_input(dataset_path)
    # print('#1', len(pre.label_dir))
    select_model = layer.Select_model()
    train_model = layer.Select_model().model_landmark(class_num=len(pre.label_dir),
                                            input_size=model_input_size)  # lm practice
    predict_compile = select_model.predict_compile(train_model)
    predict_compile.load_weights(weight_path)

    tmp_np = np.zeros((1, 29, 68))
    count = arg_count

    for i in tqdm(np_feature_path):
        GT_word = i.split('\\')[-3]
        if GT_word in word_list[count]:
            if i.split('\\')[-2] == 'train':
                continue

            file = np.load(i)['arr_0']
            X = np.expand_dims(file, axis=0)

            output = predict_compile.predict(X)
            pred_index = np.argmax(output[0])
            # print('pred', pred_index)

            if GT_word == word_list[pred_index]:
                # if GT_word != word_list[count]:

                print('GT_word', word_list[pred_index])
                heatmap = saliencymap_func(predict_compile, file, pred_index)
                heatmap = np.mean(heatmap, axis=2) * 10e5
                heatmap_expand = np.expand_dims(heatmap, axis=0)
                # print(heatmap_expand.shape)

                tmp_np = np.concatenate((tmp_np, heatmap_expand))
            else:
                continue

    save_heatmap(save_root, tmp_np, word_list[count], npz_path)


def feature_list_npz_saved(input_feature, saved_path_name):
    feature_path = []

    for roots, dirs, files in os.walk(input_feature):
        for file in files:
            if file.split('.')[-1] == 'npz':
                file_path = os.path.join(roots, file)
                feature_path.append(file_path)
            else:
                continue

    np_feature_path = np.asarray(feature_path)

    tmp_list_f = []
    for i in range(len(np_feature_path)):
        tmp_f = np_feature_path[i].split('\\')
        tmp2_f = os.path.join(tmp_f[-3], tmp_f[-2], tmp_f[-1])
        tmp3_f = tmp2_f.split('.')[-3] + '.' + tmp2_f.split('.')[-2]
        tmp_list_f.append(tmp3_f)

    np.savez_compressed(saved_path_name, np_feature_path)


def saliecymap_npz_to_mean(NPZ_path, save_path):
    NPZ_list = os.listdir(NPZ_path)

    for file in NPZ_list:
        file_path = os.path.join(NPZ_path, file)
        file_value = np.load(file_path)['arr_0']
        file_mean = np.mean(file_value, axis=0)
        np.savez_compressed(os.path.join(save_path, file.split('.')[0]) + '_mean', file_mean)


def cosim_euclidean_similarity(root_path):
    folder_path = root_path + 'NPZ_mean\\'
    word_list = os.listdir(folder_path)
    com_word_list = list(combinations(word_list, 2))
    delete_range = (0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47)

    cos_word1 = []
    euc_word1 = []
    dic_cos = {}
    dic_euc = {}

    for i in tqdm(range(len(com_word_list))):
        word1_path = folder_path + com_word_list[i][0]
        word2_path = folder_path + com_word_list[i][1]

        word1_value = np.load(word1_path)['arr_0']  # ex) access
        word2_value = np.load(word2_path)['arr_0']  # ex) young

        heatmap_1 = z_score(word1_value, delete_range)
        heatmap_2 = z_score(word2_value, delete_range)

        cos_word1.append(cos_sim(heatmap_1, heatmap_2))
        euc_word1.append(euclidean_distance(heatmap_1, heatmap_2))

        f = open(root_path + 'write_2.csv', 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([com_word_list[i], cos_word1[i], euc_word1[i]])

        dic_cos[str(com_word_list[i])] = cos_word1[i]
        dic_euc[str(com_word_list[i])] = euc_word1[i]

    f.close()

    dic_cos_sort = sorted(dic_cos.items(), key=operator.itemgetter(1))
    dic_euc_sort = sorted(dic_euc.items(), key=operator.itemgetter(1), reverse=True)

    f = open(root_path + 'dic_cos_sort.csv', 'a', newline='')
    wr = csv.writer(f)
    # for key, value in dic_cos.items():
    for tmp in dic_cos_sort:
        key = tmp[0]
        value = tmp[1]
        wr.writerow([key, value])
    f.close()

    f = open(root_path + 'dic_euc_sort.csv', 'a', newline='')
    wr = csv.writer(f)
    # for key, value in dic_cos.items():
    for tmp in dic_euc_sort:
        key = tmp[0]
        value = tmp[1]
        wr.writerow([key, value])
    f.close()


if __name__ == "__main__":

    landmark_featrue_folder = '.\\landmark featrue folder\\'
    landmark_feature_list_path = '.\\make landmark feature path list\\'
    feature_list_npz_saved(input_feature=landmark_featrue_folder, saved_path_name=landmark_feature_list_path)

    weight_path = 'weight(.h5) file'
    dataset_path = 'original dataset path\\'
    saliencymap_saved = 'heatmap save root\\'
    for i in range(26):
        main_heatmap_list_npz_saved(arg_count=i, weight_path=weight_path, npz_path=landmark_feature_list_path, dataset_path=dataset_path, save_root=saliencymap_saved)

    saliencymap_mean_saved = 'saliencymap npz to saliencymap saved mean \\'
    saliecymap_npz_to_mean(saliencymap_saved, saliencymap_mean_saved)

    cosim_euclidean_similarity(saliencymap_mean_saved)

