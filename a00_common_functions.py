# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import pickle
import os
import glob
import time
import cv2
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
from multiprocessing import Process, Manager
import random
from a02_zoo import get_input_shape, preprocess_input_overall
# import tifffile

random.seed(2016)
np.random.seed(2016)


MAX_IMAGES_FOR_INFERENCE = 12000
INPUT_PATH = '../input/'


def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def get_indexes():
    return ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy',
            'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road',
            'selective_logging', 'slash_burn', 'water']


def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def random_augment_image(image):
    box_size = 224
    start0 = random.randint(0, image.shape[0] - box_size)
    start1 = random.randint(0, image.shape[1] - box_size)
    image = image[start0:start0 + box_size, start1:start1 + box_size]

    # all possible mirroring and flips
    # (in total there are only 8 possible configurations)
    mirror = random.randint(0, 1)
    if mirror == 1:
        # flipud
        image = image[::-1, :, :]
    angle = random.randint(0, 3)
    if angle != 0:
        image = np.rot90(image, k=angle)

    # image = cv2.resize(image, (299, 299), cv2.INTER_LANCZOS4)
    return image


def get_augmented_image_list_single_part(proc_num, files, augment_number_per_image, input_shape, return_dict):
    image_list = []

    # print('Start process: {}. Images {}'.format(proc_num, len(files)*augment_number_per_image))
    for ti in range(len(files)):
        image = cv2.imread(files[ti])
        for j in range(augment_number_per_image):
            im1 = random_augment_image(image.copy())
            # im1 = cv2.resize(im1, input_shape, cv2.INTER_LINEAR)
            image_list.append(im1)
    return_dict[proc_num] = np.array(image_list)
    # print('Finished process {}'.format(proc_num))


def append_mirrors(im, image_list):
    image_list.append(im)
    for i in range(3):
        im = np.rot90(im, k=1)
        image_list.append(im)
    im = np.rot90(im, k=1)
    im = im[::-1, :, :]
    image_list.append(im)
    for i in range(3):
        im = np.rot90(im, k=1)
        image_list.append(im)
    return image_list


def append_random_mirror(im, image_list):
    rnd = random.randint(0, 7)
    if rnd < 4:
        if rnd == 0:
            image_list.append(im)
            return image_list
        for i in range(3):
            im = np.rot90(im, k=1)
            if rnd == i+1:
                image_list.append(im)
                return image_list
    else:
        im = im[::-1, :, :]
        if rnd == 4:
            image_list.append(im)
            return image_list
        for i in range(3):
            im = np.rot90(im, k=1)
            if rnd == i + 5:
                image_list.append(im)
                return image_list
    return image_list


def get_augmented_image_list_single_part_using_special_cases_8(proc_num, files, augment_number_per_image, input_shape, return_dict):
    image_list = []
    box_size = 224
    # print('Start process: {}. Images {}'.format(proc_num, len(files)*augment_number_per_image))
    for ti in range(len(files)):
        image = cv2.imread(files[ti])
        im = cv2.resize(image, (224, 224), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im, image_list)

    return_dict[proc_num] = np.array(image_list)
    # print('Finished process {}'.format(proc_num))


def get_augmented_image_list(files, augment_number_per_image, input_shape):
    # Split on parts
    threads = 6
    files_split = []
    step = len(files) // threads

    files_split.append(files[:step])
    for i in range(1, threads-1):
        files_split.append(files[i*step:(i+1)*step])
    files_split.append(files[(threads-1)*step:])

    manager = Manager()
    return_dict = manager.dict()
    p = dict()
    for i in range(threads):
        p[i] = Process(target=get_augmented_image_list_single_part_using_special_cases_8, args=(i, files_split[i], augment_number_per_image, input_shape, return_dict))
        p[i].start()
    for i in range(threads):
        p[i].join()
    # print('Return dictionary: ', len(return_dict), return_dict.keys())

    concat_list = []
    for i in range(threads):
        concat_list.append(return_dict[i])
    image_list = np.concatenate(concat_list)
    return np.array(image_list)


def get_raw_predictions_for_images(model, cnn_type, files_to_process, augment_number_per_image):
    predictions_list = []
    batch_len = MAX_IMAGES_FOR_INFERENCE // augment_number_per_image
    current_position = 0
    print('Predict for {} images...'.format(len(files_to_process) * augment_number_per_image))
    while current_position < len(files_to_process):
        if current_position + batch_len < len(files_to_process):
            part_files = files_to_process[current_position:current_position + batch_len]
        else:
            part_files = files_to_process[current_position:]
        image_list = get_augmented_image_list(part_files, augment_number_per_image, get_input_shape(cnn_type))
        print('Test shape: ', str(image_list.shape))
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_input_overall(cnn_type, image_list)
        predictions_list.append(model.predict(image_list, batch_size=32, verbose=2))
        current_position += batch_len
    predictions = np.concatenate(predictions_list)
    if len(predictions) != len(files_to_process) * augment_number_per_image:
        print('Some error here on augmentation!')
        exit()

    # Averaging predictions
    preds = []
    total = 0
    for i in range(len(files_to_process)):
        part = []
        for j in range(augment_number_per_image):
            part.append(predictions[total])
            total += 1
        part = np.mean(np.array(part), axis=0)
        preds.append(part)
    if len(preds) !=  len(files_to_process):
        print('Some error here on augmentation (averaging)!')
        exit()

    return np.array(preds)


def get_augmented_image_list_single_part_using_special_cases_16(proc_num, files, augment_number_per_image, input_shape, return_dict):
    image_list = []
    box_size = 224
    delta = (256 - 224) // 2
    # print('Start process: {}. Images {}'.format(proc_num, len(files)*augment_number_per_image))
    for ti in range(len(files)):
        image = cv2.imread(files[ti])
        im = cv2.resize(image, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im, image_list)
        im_part = image[delta:delta+box_size, delta:delta+box_size, :].copy()
        image_list = append_mirrors(im_part, image_list)

    return_dict[proc_num] = np.array(image_list)
    # print('Finished process {}'.format(proc_num))


def get_augmented_image_list_single_part_using_special_cases_24(proc_num, files, augment_number_per_image, input_shape, return_dict):
    image_list = []
    box_size = 224
    delta1 = (256 - 224) // 2
    delta2 = (256 - 224) // 4
    # print('Start process: {}. Images {}'.format(proc_num, len(files)*augment_number_per_image))
    for ti in range(len(files)):
        image = cv2.imread(files[ti])

        im = cv2.resize(image, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im, image_list)

        im_part = image[delta1:delta1+box_size, delta1:delta1+box_size, :].copy()
        if im_part.shape[0] != box_size:
            im_part = cv2.resize(im_part, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im_part, image_list)

        im_part = image[delta2:image.shape[0] - delta2, delta2:image.shape[1] - delta2, :].copy()
        im_part = cv2.resize(im_part, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im_part, image_list)

    return_dict[proc_num] = np.array(image_list)
    # print('Finished process {}'.format(proc_num))


def get_augmented_image_list_single_part_using_special_cases_32(proc_num, files, augment_number_per_image, input_shape, return_dict):
    image_list = []
    box_size = 224
    out_size = input_shape[0]
    delta1 = (256 - 224) // 2
    delta2 = (256 - 224) // 4
    # print('Start process: {}. Images {}'.format(proc_num, len(files)*augment_number_per_image))
    for ti in range(len(files)):
        image = cv2.imread(files[ti])

        im = cv2.resize(image, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im, image_list)

        im_part = image[delta1:delta1+box_size, delta1:delta1+box_size, :].copy()
        if im_part.shape[0] != out_size:
            im_part = cv2.resize(im_part, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im_part, image_list)

        im_part = image[delta2:image.shape[0] - delta2, delta2:image.shape[1] - delta2, :].copy()
        im_part = cv2.resize(im_part, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im_part, image_list)

        # 4 corners (2 sizes + random rotation)
        im_part = image[0:0 + box_size, 0:0 + box_size, :].copy()
        if im_part.shape[0] != out_size:
            im_part = cv2.resize(im_part, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_random_mirror(im_part, image_list)
        im_part = image[image.shape[0]-box_size:image.shape[0], 0:0 + box_size, :].copy()
        if im_part.shape[0] != out_size:
            im_part = cv2.resize(im_part, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_random_mirror(im_part, image_list)
        im_part = image[0:0 + box_size, image.shape[1] - box_size:image.shape[1], :].copy()
        if im_part.shape[0] != out_size:
            im_part = cv2.resize(im_part, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_random_mirror(im_part, image_list)
        im_part = image[image.shape[0]-box_size:image.shape[0], image.shape[1] - box_size:image.shape[1], :].copy()
        if im_part.shape[0] != out_size:
            im_part = cv2.resize(im_part, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_random_mirror(im_part, image_list)

        box_size_2 = random.randint(box_size+1, image.shape[0] - 1)
        im_part = image[0:0 + box_size_2, 0:0 + box_size_2, :].copy()
        im_part = cv2.resize(im_part, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_random_mirror(im_part, image_list)
        box_size_2 = random.randint(box_size + 1, image.shape[0] - 1)
        im_part = image[image.shape[0]-box_size_2:image.shape[0], 0:0 + box_size_2, :].copy()
        im_part = cv2.resize(im_part, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_random_mirror(im_part, image_list)
        box_size_2 = random.randint(box_size + 1, image.shape[0] - 1)
        im_part = image[0:0 + box_size_2, image.shape[1] - box_size_2:image.shape[1], :].copy()
        im_part = cv2.resize(im_part, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_random_mirror(im_part, image_list)
        box_size_2 = random.randint(box_size + 1, image.shape[0] - 1)
        im_part = image[image.shape[0]-box_size_2:image.shape[0], image.shape[1] - box_size_2:image.shape[1], :].copy()
        im_part = cv2.resize(im_part, (out_size, out_size), cv2.INTER_LANCZOS4)
        image_list = append_random_mirror(im_part, image_list)


    return_dict[proc_num] = np.array(image_list)
    # print('Finished process {}'.format(proc_num))


def get_augmented_image_list_single_part_using_special_cases_88(proc_num, files, augment_number_per_image, input_shape, return_dict):
    image_list = []
    box_size = input_shape[0]
    delta1 = (256 - 224) // 2
    delta2 = (256 - 224) // 4
    # print('Start process: {}. Images {}'.format(proc_num, len(files)*augment_number_per_image))
    for ti in range(len(files)):
        image = cv2.imread(files[ti])

        im = cv2.resize(image, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im, image_list)

        im_part = image[delta1:delta1+box_size, delta1:delta1+box_size, :].copy()
        if im_part.shape[0] != box_size:
            im_part = cv2.resize(im_part, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im_part, image_list)

        im_part = image[delta2:image.shape[0] - delta2, delta2:image.shape[1] - delta2, :].copy()
        im_part = cv2.resize(im_part, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im_part, image_list)

        # 4 corners (2 sizes + random rotation)
        im_part = image[0:0 + box_size, 0:0 + box_size, :].copy()
        image_list = append_mirrors(im_part, image_list)
        im_part = image[image.shape[0]-box_size:image.shape[0], 0:0 + box_size, :].copy()
        image_list = append_mirrors(im_part, image_list)
        im_part = image[0:0 + box_size, image.shape[1] - box_size:image.shape[1], :].copy()
        image_list = append_mirrors(im_part, image_list)
        im_part = image[image.shape[0]-box_size:image.shape[0], image.shape[1] - box_size:image.shape[1], :].copy()
        image_list = append_mirrors(im_part, image_list)

        box_size_2 = random.randint(box_size+1, image.shape[0] - 1)
        im_part = image[0:0 + box_size_2, 0:0 + box_size_2, :].copy()
        im_part = cv2.resize(im_part, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im_part, image_list)
        box_size_2 = random.randint(box_size + 1, image.shape[0] - 1)
        im_part = image[image.shape[0]-box_size_2:image.shape[0], 0:0 + box_size_2, :].copy()
        im_part = cv2.resize(im_part, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im_part, image_list)
        box_size_2 = random.randint(box_size + 1, image.shape[0] - 1)
        im_part = image[0:0 + box_size_2, image.shape[1] - box_size_2:image.shape[1], :].copy()
        im_part = cv2.resize(im_part, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im_part, image_list)
        box_size_2 = random.randint(box_size + 1, image.shape[0] - 1)
        im_part = image[image.shape[0]-box_size_2:image.shape[0], image.shape[1] - box_size_2:image.shape[1], :].copy()
        im_part = cv2.resize(im_part, (box_size, box_size), cv2.INTER_LANCZOS4)
        image_list = append_mirrors(im_part, image_list)


    return_dict[proc_num] = np.array(image_list)
    # print('Finished process {}'.format(proc_num))


def get_augmented_image_list_v2(files, augment_number_per_image, input_shape):
    # Split on parts
    threads = 6
    files_split = []
    step = len(files) // threads

    files_split.append(files[:step])
    for i in range(1, threads-1):
        files_split.append(files[i*step:(i+1)*step])
    files_split.append(files[(threads-1)*step:])

    manager = Manager()
    return_dict = manager.dict()
    p = dict()
    for i in range(threads):
        p[i] = Process(target=get_augmented_image_list_single_part_using_special_cases_24, args=(i, files_split[i], augment_number_per_image, input_shape, return_dict))
        p[i].start()
    for i in range(threads):
        p[i].join()
    # print('Return dictionary: ', len(return_dict), return_dict.keys())

    concat_list = []
    for i in range(threads):
        concat_list.append(return_dict[i])
    image_list = np.concatenate(concat_list)
    return np.array(image_list)


def get_augmented_image_list_v3(files, augment_number_per_image, input_shape):
    # Split on parts
    threads = 10
    files_split = []
    step = len(files) // threads

    files_split.append(files[:step])
    for i in range(1, threads-1):
        files_split.append(files[i*step:(i+1)*step])
    files_split.append(files[(threads-1)*step:])

    manager = Manager()
    return_dict = manager.dict()
    p = dict()
    for i in range(threads):
        p[i] = Process(target=get_augmented_image_list_single_part_using_special_cases_32, args=(i, files_split[i], augment_number_per_image, input_shape, return_dict))
        p[i].start()
    for i in range(threads):
        p[i].join()
    # print('Return dictionary: ', len(return_dict), return_dict.keys())

    concat_list = []
    for i in range(threads):
        concat_list.append(return_dict[i])
    image_list = np.concatenate(concat_list)
    return np.array(image_list)


def get_augmented_image_list_v4(files, augment_number_per_image, input_shape):
    # Split on parts
    threads = 10
    files_split = []
    step = len(files) // threads

    files_split.append(files[:step])
    for i in range(1, threads-1):
        files_split.append(files[i*step:(i+1)*step])
    files_split.append(files[(threads-1)*step:])

    manager = Manager()
    return_dict = manager.dict()
    p = dict()
    for i in range(threads):
        p[i] = Process(target=get_augmented_image_list_single_part_using_special_cases_88, args=(i, files_split[i], augment_number_per_image, input_shape, return_dict))
        p[i].start()
    for i in range(threads):
        p[i].join()
    # print('Return dictionary: ', len(return_dict), return_dict.keys())

    concat_list = []
    for i in range(threads):
        concat_list.append(return_dict[i])
    image_list = np.concatenate(concat_list)
    return np.array(image_list)


def get_raw_predictions_for_images_v2(model, cnn_type, files_to_process):
    augment_number_per_image = 24
    print('Predict for {} images...'.format(len(files_to_process) * augment_number_per_image))
    predictions_list = []
    batch_len = MAX_IMAGES_FOR_INFERENCE // augment_number_per_image
    current_position = 0
    while current_position < len(files_to_process):
        if current_position + batch_len < len(files_to_process):
            part_files = files_to_process[current_position:current_position + batch_len]
        else:
            part_files = files_to_process[current_position:]
        image_list = get_augmented_image_list_v2(part_files, augment_number_per_image, get_input_shape(cnn_type))
        print('Test shape: ', str(image_list.shape))
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_input_overall(cnn_type, image_list)
        predictions_list.append(model.predict(image_list, batch_size=32, verbose=2))
        current_position += batch_len
    predictions = np.concatenate(predictions_list)
    if len(predictions) != len(files_to_process) * augment_number_per_image:
        print('Some error here on augmentation!')
        exit()

    # Averaging predictions
    preds = []
    total = 0
    for i in range(len(files_to_process)):
        part = []
        for j in range(augment_number_per_image):
            part.append(predictions[total])
            total += 1
        part = np.mean(np.array(part), axis=0)
        preds.append(part)
    if len(preds) !=  len(files_to_process):
        print('Some error here on augmentation (averaging)!')
        exit()

    return np.array(preds)


def get_raw_predictions_for_images_v3(model, cnn_type, files_to_process):
    from keras import backend as K
    tr_back = False
    if K.image_dim_ordering() == 'tf':
        print('Backward translate')
        tr_back = True

    augment_number_per_image = 32
    print('Predict for {} images...'.format(len(files_to_process) * augment_number_per_image))
    predictions_list = []
    batch_len = MAX_IMAGES_FOR_INFERENCE // augment_number_per_image
    current_position = 0
    while current_position < len(files_to_process):
        if current_position + batch_len < len(files_to_process):
            part_files = files_to_process[current_position:current_position + batch_len]
        else:
            part_files = files_to_process[current_position:]
        image_list = get_augmented_image_list_v3(part_files, augment_number_per_image, get_input_shape(cnn_type))
        print('Test shape: ', str(image_list.shape))
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_input_overall(cnn_type, image_list)
        if K.image_dim_ordering() == 'tf':
            image_list = image_list.transpose((0, 2, 3, 1))
        predictions_list.append(model.predict(image_list, batch_size=32, verbose=2))
        current_position += batch_len
    predictions = np.concatenate(predictions_list)
    if len(predictions) != len(files_to_process) * augment_number_per_image:
        print('Some error here on augmentation!')
        exit()

    # Averaging predictions
    preds = []
    total = 0
    for i in range(len(files_to_process)):
        part = []
        for j in range(augment_number_per_image):
            part.append(predictions[total])
            total += 1
        part = np.mean(np.array(part), axis=0)
        preds.append(part)
    if len(preds) !=  len(files_to_process):
        print('Some error here on augmentation (averaging)!')
        exit()

    return np.array(preds)


def get_raw_predictions_for_images_v4(model, cnn_type, files_to_process):
    augment_number_per_image = 88
    print('Predict for {} images...'.format(len(files_to_process) * augment_number_per_image))
    predictions_list = []
    batch_len = MAX_IMAGES_FOR_INFERENCE // augment_number_per_image
    current_position = 0
    while current_position < len(files_to_process):
        if current_position + batch_len < len(files_to_process):
            part_files = files_to_process[current_position:current_position + batch_len]
        else:
            part_files = files_to_process[current_position:]
        image_list = get_augmented_image_list_v4(part_files, augment_number_per_image, get_input_shape(cnn_type))
        print('Test shape: ', str(image_list.shape))
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_input_overall(cnn_type, image_list)
        predictions_list.append(model.predict(image_list, batch_size=32, verbose=2))
        current_position += batch_len
    predictions = np.concatenate(predictions_list)
    if len(predictions) != len(files_to_process) * augment_number_per_image:
        print('Some error here on augmentation!')
        exit()

    # Averaging predictions
    preds = []
    total = 0
    for i in range(len(files_to_process)):
        part = []
        for j in range(augment_number_per_image):
            part.append(predictions[total])
            total += 1
        part = np.mean(np.array(part), axis=0)
        preds.append(part)
    if len(preds) !=  len(files_to_process):
        print('Some error here on augmentation (averaging)!')
        exit()

    return np.array(preds)


def get_train_label_matrix():
    tbl = pd.read_csv(INPUT_PATH + "train_v2.csv")
    labels = tbl['tags'].apply(lambda x: x.split(' '))
    counts = defaultdict(int)
    for l in labels:
        for l2 in l:
            counts[l2] += 1
    indexes = get_indexes()
    for i in range(len(indexes)):
        tbl['label_{}'.format(i)] = 0

    files = []
    for id in tbl['image_name'].values:
        files.append("../input/train-jpg/" + id + '.jpg')
    files = np.array(files)

    lbl = np.zeros((len(labels), len(indexes)))
    for j in range(len(labels)):
        l = labels[j]
        for i in range(len(indexes)):
            if indexes[i] in l:
                lbl[j][i] = 1

    return lbl


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=6))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))

