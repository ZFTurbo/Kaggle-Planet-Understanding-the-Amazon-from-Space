# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *

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

import random
from a02_zoo import *

random.seed(2016)
np.random.seed(2016)

RESTORE_FROM_LAST_CHECKPOINT = 0

INPUT_PATH = "../input/"
MODELS_PATH = '../models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
OUTPUT_PATH = "../subm/"
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
FEATURES_PATH = "../features/"
if not os.path.isdir(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)
CODE_COPY_FOLDER = "../models/code/"
if not os.path.isdir(CODE_COPY_FOLDER):
    os.mkdir(CODE_COPY_FOLDER)
HISTORY_FOLDER_PATH = "../models/history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)
CACHE_PATH = "../cache/"
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)


def get_validation_score_weather(nfolds, cnn_type):
    global FOLD_TO_CALC
    from keras.models import load_model
    from keras import backend as K

    if K.backend() == 'tensorflow':
        print('Update dim ordering to "tf"')
        K.set_image_dim_ordering('tf')

    restore_from_cache = 0

    choose = [5, 6, 10, 11]

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
        files.append("../input/train-jpg/" + id + '.jpg')
    files = np.array(files)

    lbl = np.zeros((len(labels), len(indexes)))
    for j in range(len(labels)):
        l = labels[j]
        for i in range(len(indexes)):
            if indexes[i] in l:
                lbl[j][i] = 1

    # print(lbl)
    print(lbl.shape)
    stat = []

    kf = KFold(n_splits=nfolds, shuffle=True, random_state=get_random_state(cnn_type))
    num_fold = 0
    result = np.zeros((len(labels), len(choose)))
    for train_ids, valid_ids in kf.split(range(len(files))):
        num_fold += 1

        if 'FOLD_TO_CALC' in globals():
            if num_fold not in FOLD_TO_CALC:
                continue

        start_time = time.time()
        cache_file = CACHE_PATH + '{}_valid_fold_{}_weather'.format(cnn_type, num_fold)
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(train_ids))
        print('Split valid: ', len(valid_ids))
        valid_files = files[valid_ids]
        valid_labels = lbl[valid_ids][:, choose]
        if not (os.path.isfile(cache_file) and restore_from_cache):
            final_model_path = MODELS_PATH + '{}_fold_{}_weather.h5'.format(cnn_type, num_fold)
            print('Loading model {}...'.format(final_model_path))
            if cnn_type == 'RESNET101' or cnn_type == 'RESNET152' or 'DENSENET' in cnn_type:
                model = get_pretrained_model(cnn_type, len(choose), final_layer_activation='softmax')
                model.load_weights(final_model_path)
                weights = model.layers[-1].get_weights()
                print(weights[0].shape)
                print(weights[1].shape)
            else:
                c = dict()
                c['f2beta_loss'] = f2beta_loss
                model = load_model(final_model_path, custom_objects=c)
            preds = get_raw_predictions_for_images_v3(model, cnn_type, valid_files)
            save_in_file(preds, cache_file)
        else:
            preds = load_from_file(cache_file)

        for i in range(len(valid_ids)):
            result[valid_ids[i], :] = preds[i]
        print(preds.shape)
        print(valid_labels.shape)

        best_score = -1
        best_thr = -1
        for thr1 in range(1, 100):
            p = preds.copy()
            thr = thr1 / 100
            p[p > thr] = 1
            p[p <= thr] = 0
            score = f2_score(valid_labels, p)
            print('THR: {} SCORE: {}'.format(thr, score))
            if score > best_score:
                best_score = score
                best_thr = thr

        stat.append((best_score, best_thr))
        print('Best score: {} THR: {}'.format(best_score, best_thr))
        print('Fold time: {} seconds'.format(time.time() - start_time))
        # if num_fold == 1:
        #    exit()

    best_score = -1
    best_thr = -1
    for thr1 in range(1, 100):
        p = result.copy()
        thr = thr1 / 100
        p[p > thr] = 1
        p[p <= thr] = 0
        score = f2_score(lbl[:, choose], p)
        print('THR: {} SCORE: {}'.format(thr, score))
        if score > best_score:
            best_score = score
            best_thr = thr
    print('Best overall score: {} THR: {}'.format(best_score, best_thr))
    for i in range(len(stat)):
        print('Best score fold {}: {} THR: {}'.format(i+1, stat[i][0], stat[i][1]))

    # Save validation file
    out = open(FEATURES_PATH + "valid_{}_score_{}_thr_{}_weather.csv".format(cnn_type, best_score, best_thr), "w")
    # out = open(FEATURES_PATH + "valid_{}_weather.csv".format(cnn_type), "w")
    out.write("image_name")
    for i in choose:
        out.write("," + indexes[i])
    out.write("\n")
    ids = tbl['image_name'].values
    for i in range(len(result)):
        out.write(ids[i])
        for j in range(len(choose)):
            out.write("," + str(result[i][j]))
        out.write("\n")
    out.close()

    return best_score, best_thr


def process_test_weather(nfolds, cnn_type, score, thr):
    global FOLD_TO_CALC
    from keras.models import load_model
    from keras import backend as K

    if K.backend() == 'tensorflow':
        print('Update dim ordering to "tf"')
        K.set_image_dim_ordering('tf')

    restore_from_cache = 0
    choose = [5, 6, 10, 11]

    tbl = pd.read_csv(INPUT_PATH + "sample_submission_v2.csv")
    indexes = get_indexes()
    ids = tbl['image_name'].values

    files = []
    for id in ids:
        files.append(INPUT_PATH + "test-jpg/" + id + '.jpg')
    files = np.array(files)

    preds = []
    for num_fold in range(1, nfolds+1):

        if 'FOLD_TO_CALC' in globals():
            if num_fold not in FOLD_TO_CALC:
                continue

        cache_file = CACHE_PATH + '{}_test_fold_{}_weather'.format(cnn_type, num_fold)
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        if os.path.isfile(cache_file) and restore_from_cache:
            print('Restore from cache...')
            p = load_from_file(cache_file)
        else:
            final_model_path = MODELS_PATH + '{}_fold_{}_weather.h5'.format(cnn_type, num_fold)
            print('Loading model {}...'.format(final_model_path))
            if cnn_type == 'RESNET101' or cnn_type == 'RESNET152' or 'DENSENET' in cnn_type:
                model = get_pretrained_model(cnn_type, len(choose), final_layer_activation='softmax')
                model.load_weights(final_model_path)
            else:
                c = dict()
                c['f2beta_loss'] = f2beta_loss
                model = load_model(final_model_path, custom_objects=c)
            p = get_raw_predictions_for_images_v3(model, cnn_type, files)
            save_in_file(p, cache_file)
        preds.append(p)
    preds = np.array(preds)
    print(preds.shape)
    preds = np.mean(preds, axis=0)

    # Save raw file
    out = open(FEATURES_PATH + "test_{}_score_{}_thr_{}_weather.csv".format(cnn_type, score, thr), "w")
    out.write("image_name")
    for i in choose:
        out.write("," + indexes[i])
    out.write("\n")
    ids = tbl['image_name'].values
    for i in range(len(preds)):
        out.write(ids[i])
        for j in range(len(choose)):
            out.write("," + str(preds[i][j]))
        out.write("\n")
    out.close()

    # Create submission
    out = open(OUTPUT_PATH + "subm_{}_score_{}_thr_{}_weather.csv".format(cnn_type, score, thr), "w")
    out.write("image_name,tags\n")
    for i in range(len(files)):
        out.write(ids[i] + ',')
        for j in range(len(choose)):
            if preds[i][j] > thr:
                out.write(indexes[j] + ' ')
        out.write("\n")
    out.close()


if __name__ == '__main__':
    num_folds = 5
    for cnn in ['DENSENET_121']:
        best_score, best_thr = get_validation_score_weather(num_folds, cnn)
        process_test_weather(num_folds, cnn, best_score, best_thr)


'''
Validation result:

Best overall score: 0.9541445064401411 THR: 0.18
Best score fold 1: 0.9531132364012799 THR: 0.16
Best score fold 2: 0.9549042443064183 THR: 0.21
Best score fold 3: 0.9564864483342744 THR: 0.22
Best score fold 4: 0.953799201251647 THR: 0.18
Best score fold 5: 0.9557163445983705 THR: 0.23
'''