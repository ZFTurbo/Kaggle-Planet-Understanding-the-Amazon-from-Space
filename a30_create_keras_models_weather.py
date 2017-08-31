# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import glob
import cv2
import pandas as pd
from sklearn.model_selection import KFold
from a00_common_functions import get_train_label_matrix, get_indexes

GPU_TO_USE = 0
USE_THEANO = 1

# Uncomment if you need to calculate specific fold
# FOLD_TO_CALC = [5]

if USE_THEANO:
    os.environ["KERAS_BACKEND"] = "theano"
    os.environ["THEANO_FLAGS"] = "device=gpu{},lib.cnmem=0.81".format(GPU_TO_USE, GPU_TO_USE)
else:
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(GPU_TO_USE)


import datetime
import random
from a02_zoo import *

random.seed(2016)
np.random.seed(2016)

PATIENCE = 50
NB_EPOCH = 1000
RESTORE_FROM_LAST_CHECKPOINT = 0
UPDATE_BEST_MODEL = 0
RECREATE_MODELS = 0


INPUT_PATH = "../input/"
MODELS_PATH = '../models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
OUTPUT_PATH = "../subm/"
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
CODE_COPY_FOLDER = "../models/code/"
if not os.path.isdir(CODE_COPY_FOLDER):
    os.mkdir(CODE_COPY_FOLDER)
HISTORY_FOLDER_PATH = "../models/history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)


FULL_IMAGE_ARRAY = dict()
def prepread_images():
    files = glob.glob(INPUT_PATH + "train-jpg/*.jpg")
    total = 0
    for f in files:
        FULL_IMAGE_ARRAY[os.path.basename(f)] = cv2.imread(f)
        total += 1
        if total % 5000 == 0:
            print('Read {} files from {}...'.format(total, len(files)))


def random_intensity_change(img, max_change):
    img = img.astype(np.int16)
    for j in range(3):
        delta = random.randint(-max_change, max_change)
        img[:, :, j] += delta
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)


def batch_generator_train(cnn_type, files, labels, augment=False):
    import keras.backend as K
    global FULL_IMAGE_ARRAY

    dim_ordering = K.image_dim_ordering()
    in_shape = get_input_shape(cnn_type)
    batch_size = get_batch_size(cnn_type)
    if len(FULL_IMAGE_ARRAY) == 0:
        prepread_images()

    while True:
        index = random.sample(range(len(files)), batch_size)
        batch_files = files[index]
        batch_labels = labels[index]

        image_list = []
        mask_list = []
        for i in range(len(batch_files)):
            # image = cv2.imread(batch_files[i])
            image = FULL_IMAGE_ARRAY[os.path.basename(batch_files[i])]

            if cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V4' or cnn_type == 'XCEPTION':
                random_border = 20
                start0 = random.randint(0, random_border)
                start1 = random.randint(0, random_border)
                end0 = random.randint(0, random_border)
                end1 = random.randint(0, random_border)
                image = image[start0:image.shape[0] - end0, start1:image.shape[1] - end1]
                image = cv2.resize(image, (299, 299), cv2.INTER_LANCZOS4)
            else:
                box_size = random.randint(200, 256)
                start0 = random.randint(0, image.shape[0] - box_size)
                start1 = random.randint(0, image.shape[1] - box_size)
                image = image[start0:start0 + box_size, start1:start1 + box_size]
                image = cv2.resize(image, in_shape, cv2.INTER_LANCZOS4)

            if augment:
                # all possible mirroring and flips
                # (in total there are only 8 possible configurations)
                mirror = random.randint(0, 1)
                if mirror == 1:
                    # flipud
                    image = image[::-1, :, :]
                angle = random.randint(0, 3)
                if angle != 0:
                    image = np.rot90(image, k=angle)
                image = random_intensity_change(image, 3)

            mask = batch_labels[i]
            image_list.append(image.astype(np.float32))
            mask_list.append(mask)
        image_list = np.array(image_list)
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_input_overall(cnn_type, image_list)
        if dim_ordering == 'tf':
            image_list = image_list.transpose((0, 2, 3, 1))
        mask_list = np.array(mask_list)
        yield image_list, mask_list


def train_single_model(num_fold, cnn_type, train_files, valid_files, train_labels, valid_labels):
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    print('Creating and compiling model [{}]...'.format(cnn_type))
    model = get_pretrained_model(cnn_type, 4, final_layer_activation='softmax')

    final_model_path = MODELS_PATH + '{}_fold_{}_weather.h5'.format(cnn_type, num_fold)
    cache_model_path = MODELS_PATH + '{}_temp_fold_{}_weather.h5'.format(cnn_type, num_fold)
    if os.path.isfile(final_model_path) and RECREATE_MODELS == 0:
        print('Model {} already exists. Skip it'.format(final_model_path))
        return 0.0
    if os.path.isfile(cache_model_path) and RESTORE_FROM_LAST_CHECKPOINT:
        print('Load model from last point: ', cache_model_path)
        model.load_weights(cache_model_path)
    elif os.path.isfile(final_model_path) and UPDATE_BEST_MODEL:
        print('Load model from best point: ', final_model_path)
        model.load_weights(final_model_path)
    else:
        print('Start training from begining')

    print('Fitting model...')
    batch_size = get_batch_size(cnn_type)
    print('Batch size: {}'.format(batch_size))
    print('Learning rate: {}'.format(get_learning_rate(cnn_type)))
    samples_train_per_epoch = batch_size * (1 + len(train_files) // (10 * batch_size))
    samples_valid_per_epoch = samples_train_per_epoch
    print('Samples train: {}, Samples valid: {}'.format(samples_train_per_epoch, samples_valid_per_epoch))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(generator=batch_generator_train(cnn_type, train_files, train_labels, True),
                  nb_epoch=NB_EPOCH,
                  samples_per_epoch=samples_train_per_epoch,
                  validation_data=batch_generator_train(cnn_type, valid_files, valid_labels, True),
                  nb_val_samples=samples_valid_per_epoch,
                  verbose=2, max_q_size=100,
                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}_weather.csv'.format(cnn_type, num_fold, min_loss, get_learning_rate(cnn_type), now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    return min_loss


def run_cross_validation_create_models(nfolds, cnn_type):
    global FOLD_TO_CALC
    tbl = pd.read_csv(INPUT_PATH + "train_v2.csv")
    lbl = get_train_label_matrix()
    indexes = get_indexes()

    # Only select weather indexes
    # ['clear', 'partly_cloudy', 'haze', 'cloudy']
    choose = [5, 6, 10, 11]
    print('Choose weather indexes: {}'.format(np.array(indexes)[choose]))
    lbl = lbl[:, choose]

    files = []
    for id in tbl['image_name'].values:
        files.append("../input/train-jpg/" + id + '.jpg')
    files = np.array(files)

    print('Label shape:', lbl.shape)
    print('Max labels in row:', max(lbl.sum(axis=1)))

    kf = KFold(n_splits=nfolds, shuffle=True, random_state=get_random_state(cnn_type))
    num_fold = 0
    sum_score = 0
    for train_ids, valid_ids in kf.split(range(len(files))):
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(train_ids))
        print('Split valid: ', len(valid_ids))
        train_files = files[train_ids]
        valid_files = files[valid_ids]
        train_labels = lbl[train_ids]
        valid_labels = lbl[valid_ids]

        if 'FOLD_TO_CALC' in globals():
            if num_fold not in FOLD_TO_CALC:
                continue

        score = train_single_model(num_fold, cnn_type, train_files, valid_files, train_labels, valid_labels)
        sum_score += score

    print('Avg loss: {}'.format(sum_score/nfolds))
    return sum_score/nfolds


if __name__ == '__main__':
    num_folds = 5
    score1 = ''
    for cnn in ['DENSENET_121']:
        score1 = run_cross_validation_create_models(num_folds, cnn)


'''
Training history:
Fold 1: 0.185169663315
Fold 2: 0.1752433345
Fold 3: 0.175999773528
Fold 4: 0.177833901902
Fold 5: 0.168742229386
'''
