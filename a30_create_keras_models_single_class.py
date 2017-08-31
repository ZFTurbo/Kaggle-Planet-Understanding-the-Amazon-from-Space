# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import glob
import cv2
import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict

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

    files_no_class = files[labels == 0].copy()
    files_have_class = files[labels == 1].copy()

    while True:
        b_no = batch_size // 2
        b_have = batch_size - b_no
        index1 = random.sample(range(len(files_no_class)), b_no)
        index2 = random.sample(range(len(files_have_class)), b_have)
        batch_files = np.concatenate((files_no_class[index1], files_have_class[index2]))
        batch_labels = [0] * b_no + [1] * b_have

        image_list = []
        mask_list = []
        for i in range(len(batch_files)):
            # image = cv2.imread(batch_files[i])
            image = FULL_IMAGE_ARRAY[os.path.basename(batch_files[i])]

            if cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V4' or cnn_type == 'XCEPTION':
                random_border = 30
                start0 = random.randint(0, random_border)
                start1 = random.randint(0, random_border)
                end0 = random.randint(0, random_border)
                end1 = random.randint(0, random_border)
                image = image[start0:image.shape[0] - end0, start1:image.shape[1] - end1]
                image = cv2.resize(image, (299, 299), cv2.INTER_LANCZOS4)
            else:
                box_size = random.randint(180, 256)
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
                image = random_intensity_change(image, 5)

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


def train_single_model(num_fold, cnn_type, class_id, train_files, valid_files, train_labels, valid_labels):
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    tl1 = train_labels[:, class_id]
    tl2 = valid_labels[:, class_id]
    print('Class present in {} train cases'.format(len(tl1[tl1 == 1])))
    print('Class present in {} valid cases'.format(len(tl2[tl2 == 1])))

    if class_id == 1:
        lr = 0.000003
    elif class_id == 2:
        lr = 0.000003
    elif class_id == 3:
        lr = 0.000003
    elif class_id == 0:
        lr = 0.00003
    elif class_id == 5:
        lr = 0.00003
    elif class_id == 8:
        lr = 0.00003
    elif class_id == 9:
        lr = 0.000003
    elif class_id == 13:
        lr = 0.00003
    elif class_id == 11:
        lr = 0.00003
    elif class_id == 12:
        lr = 0.00003
    elif class_id == 14:
        lr = 0.000003
    elif class_id == 15:
        lr = 0.000003
    elif class_id == 16:
        lr = 0.00003
    else:
        lr = 0.000003

    print('Creating and compiling model [{}]...'.format(cnn_type))
    model = get_pretrained_model(cnn_type, 1, learning_rate=lr, final_layer_activation='sigmoid')

    final_model_path = MODELS_PATH + '{}_fold_{}_single_class_{}.h5'.format(cnn_type, num_fold, class_id)
    cache_model_path = MODELS_PATH + '{}_temp_fold_{}_single_class_{}.h5'.format(cnn_type, num_fold, class_id)
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
    print('Learning rate: {}'.format(lr))
    samples_train_per_epoch = batch_size * (1 + len(train_files) // (10 * batch_size))
    samples_valid_per_epoch = samples_train_per_epoch
    print('Samples train: {}, Samples valid: {}'.format(samples_train_per_epoch, samples_valid_per_epoch))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(generator=batch_generator_train(cnn_type, train_files, train_labels[:, class_id], True),
                  nb_epoch=NB_EPOCH,
                  samples_per_epoch=samples_train_per_epoch,
                  validation_data=batch_generator_train(cnn_type, valid_files, valid_labels[:, class_id], True),
                  nb_val_samples=samples_valid_per_epoch,
                  verbose=2, max_q_size=300,
                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_sc_{}_{}_{:.4f}_lr_{}_{}.csv'.format(cnn_type, class_id, num_fold, min_loss, get_learning_rate(cnn_type), now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    return min_loss


def run_cross_validation_create_models_single_class(nfolds, cnn_type, class_id):
    global FOLD_TO_CALC
    tbl = pd.read_csv(INPUT_PATH + "train_v2.csv")
    labels = tbl['tags'].apply(lambda x: x.split(' '))
    counts = defaultdict(int)
    for l in labels:
        for l2 in l:
            counts[l2] += 1
    indexes = sorted(list(counts.keys()))
    for i in range(len(indexes)):
        tbl['label_{}'.format(i)] = 0

    files = []
    for id in tbl['image_name'].values:
        files.append(INPUT_PATH + "train-jpg/" + id + '.jpg')
    files = np.array(files)

    lbl = np.zeros((len(labels), len(indexes)))
    for j in range(len(labels)):
        l = labels[j]
        for i in range(len(indexes)):
            if indexes[i] in l:
                lbl[j][i] = 1

    # print(len(tbl))
    # print(len(labels))
    print(lbl.shape)

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

        score = train_single_model(num_fold, cnn_type, class_id, train_files, valid_files, train_labels, valid_labels)
        sum_score += score

    print('Avg loss: {}'.format(sum_score/nfolds))
    return sum_score/nfolds


if __name__ == '__main__':
    num_folds = 5
    score1 = ''
    for class_id in range(0, 17):
        print('Class: {}'.format(class_id))
        for cnn in ['DENSENET_121']:
            score1 = run_cross_validation_create_models_single_class(num_folds, cnn, class_id)


'''
Training history:

Class 0 (lr = 0.00003):
Minimum loss for given fold:  0.253413404281
Minimum loss for given fold:  0.248380952066
Minimum loss for given fold:  0.257372181771
Minimum loss for given fold:  0.26305075751
Minimum loss for given fold:  0.249311803058
Avg loss: 0.2543058197370466

Class 1 (lr = 0.000003):
Minimum loss for given fold:  0.108140268237
Minimum loss for given fold:  0.0683494081152
Minimum loss for given fold:  0.127105548575
Minimum loss for given fold:  0.0354281800977
Minimum loss for given fold:  0.0873878211542
Avg loss: 0.08528224523581658

Class 2 (lr = 0.00003):
Minimum loss for given fold:  0.3910083831
Minimum loss for given fold:  0.385183117686

Class 2 (lr = 0.000003):
Minimum loss for given fold:  0.388385605113
Minimum loss for given fold:  0.379616589366
Minimum loss for given fold:  0.341731107299
Minimum loss for given fold:  0.340404011907
Minimum loss for given fold:  0.331169278349
Avg loss: 0.3562613184069409

Class 3 (lr = 0.000003):
Minimum loss for given fold:  0.330049385029
Minimum loss for given fold:  0.342264363059
Minimum loss for given fold:  0.191731132314
Minimum loss for given fold:  0.288646133939
Minimum loss for given fold:  0.391212643481
Avg loss: 0.3087807315643187

Class 4 (lr = 0.000003):
Minimum loss for given fold:  0.413155568419
Minimum loss for given fold:  0.346210029666
Minimum loss for given fold:  0.389502455216
Minimum loss for given fold:  0.365005755866
Minimum loss for given fold:  0.422119773005
Avg loss: 0.3856122146004144

Class 5 (lr = 0.00003):
Minimum loss for given fold:  0.14555569979
Minimum loss for given fold:  0.166418199968
Minimum loss for given fold:  0.153322775457
Minimum loss for given fold:  0.145887596717
Minimum loss for given fold:  0.144087674372

Class 6 (lr = 0.000003):
Minimum loss for given fold:  0.0762309606049
Minimum loss for given fold:  0.0769969969614
Minimum loss for given fold:  0.0780014535268
Minimum loss for given fold:  0.0684322627767
Minimum loss for given fold:  0.0662103267042
Avg loss: 0.07317440011481076

Class 7 (lr = 0.000003):
Minimum loss for given fold:  0.168240847641
Minimum loss for given fold:  0.13196111221
Minimum loss for given fold:  0.202992705743
Minimum loss for given fold:  0.218001743572
Minimum loss for given fold:  0.111677368884
Avg loss: 0.16657475560995533

Class 8 (lr = 0.00003)
Minimum loss for given fold:  0.366216352094
Minimum loss for given fold:  0.372726423689
Minimum loss for given fold:  0.378883124962
Minimum loss for given fold:  0.358972315343
Minimum loss for given fold:  0.366272835774
Avg loss: 0.3686142103723538

Class 9 ()
Minimum loss for given fold:  0.249355089172
Minimum loss for given fold:  0.244136921579
Minimum loss for given fold:  0.235164790115
Minimum loss for given fold:  0.254924793532
Minimum loss for given fold:  0.240773150398
Avg loss: 0.24487094895937195

Class 10:
Minimum loss for given fold:  0.211632032154
Minimum loss for given fold:  0.206836078017
Minimum loss for given fold:  0.21180134523
Minimum loss for given fold:  0.193908381799
Minimum loss for given fold:  0.200499110361
Avg loss: 0.20493538951225304

Class 11:
Minimum loss for given fold:  0.133019096482
Minimum loss for given fold:  0.139944022527
Minimum loss for given fold:  0.121319166928
Minimum loss for given fold:  0.127049939878
Minimum loss for given fold:  0.140708353002
Avg loss: 0.13240811576359662

Class 12 ():
Minimum loss for given fold:  0.153227462225
Minimum loss for given fold:  0.153025816772
Minimum loss for given fold:  0.147686096728
Minimum loss for given fold:  0.147378727081
Minimum loss for given fold:  0.133062615749
Avg loss: 0.14687614371097518

Class 13:
Minimum loss for given fold:  0.228606079165
Minimum loss for given fold:  0.230709340768
Minimum loss for given fold:  0.220355313731
Minimum loss for given fold:  0.226218112549
Minimum loss for given fold:  0.23019926849
Avg loss: 0.227217622940647

Class 14:
Minimum loss for given fold:  0.306654619605
Minimum loss for given fold:  0.355313857304
Minimum loss for given fold:  0.288375787437
Minimum loss for given fold:  0.249127054771
Minimum loss for given fold:  0.290934943767
Avg loss: 0.2980812525767603

Class 15:
Minimum loss for given fold:  0.373362306275
Minimum loss for given fold:  0.396898513867
Minimum loss for given fold:  0.311251565584
Minimum loss for given fold:  0.275662233915
Minimum loss for given fold:  0.300677734324
Avg loss: 0.3315704707930117

Class 16:
Minimum loss for given fold:  0.288113219204
Minimum loss for given fold:  0.270322405262
Minimum loss for given fold:  0.276544707977
Minimum loss for given fold:  0.27940388125
Minimum loss for given fold:  0.289342449357
Avg loss: 0.2807453326099081
'''