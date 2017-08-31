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
CLASSES_NUMBER = 17

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


def get_validation_score(nfolds, cnn_type):
    from keras.models import load_model
    from keras import backend as K

    if K.backend() == 'tensorflow':
        print('Update dim ordering to "tf"')
        K.set_image_dim_ordering('tf')

    restore_from_cache = 0

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

    # print(lbl)
    print(lbl.shape)
    stat = []

    kf = KFold(n_splits=nfolds, shuffle=True, random_state=get_random_state(cnn_type))
    num_fold = 0
    result = np.zeros((len(labels), len(indexes)))
    for train_ids, valid_ids in kf.split(range(len(files))):
        num_fold += 1
        start_time = time.time()
        cache_file = CACHE_PATH + '{}_valid_fold_{}.pklz'.format(cnn_type, num_fold)
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(train_ids))
        print('Split valid: ', len(valid_ids))
        valid_files = files[valid_ids]
        valid_labels = lbl[valid_ids]
        if not (os.path.isfile(cache_file) and restore_from_cache):
            final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
            print('Loading model {}...'.format(final_model_path))
            if cnn_type == 'RESNET101' or cnn_type == 'RESNET152' or 'DENSENET' in cnn_type:
                model = get_pretrained_model(cnn_type, CLASSES_NUMBER)
                model.load_weights(final_model_path)
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
        score = f2_score(lbl, p)
        print('THR: {} SCORE: {}'.format(thr, score))
        if score > best_score:
            best_score = score
            best_thr = thr
    print('Best overall score: {} THR: {}'.format(best_score, best_thr))
    for i in range(len(stat)):
        print('Best score fold {}: {} THR: {}'.format(i+1, stat[i][0], stat[i][1]))

    # Save validation file
    out = open(FEATURES_PATH + "valid_{}_score_{}_thr_{}.csv".format(cnn_type, best_score, best_thr), "w")
    # out = open(FEATURES_PATH + "valid_{}.csv".format(cnn_type), "w")
    out.write("image_name")
    for i in range(len(indexes)):
        out.write("," + indexes[i])
    out.write("\n")
    ids = tbl['image_name'].values
    for i in range(len(result)):
        out.write(ids[i])
        for j in range(len(indexes)):
            out.write("," + str(result[i][j]))
        out.write("\n")
    out.close()

    return best_score, best_thr


def process_test(nfolds, cnn_type, score, thr):
    global FOLD_TO_CALC
    from keras.models import load_model
    from keras import backend as K

    if K.backend() == 'tensorflow':
        print('Update dim ordering to "tf"')
        K.set_image_dim_ordering('tf')

    restore_from_cache = 0

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

        cache_file = CACHE_PATH + '{}_test_fold_{}'.format(cnn_type, num_fold)
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        if os.path.isfile(cache_file) and restore_from_cache:
            print('Restore from cache...')
            p = load_from_file(cache_file)
        else:
            final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
            print('Loading model {}...'.format(final_model_path))
            if cnn_type == 'RESNET101' or cnn_type == 'RESNET152' or 'DENSENET' in cnn_type:
                model = get_pretrained_model(cnn_type, CLASSES_NUMBER)
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

    # Save raw feature file
    out = open(FEATURES_PATH + "test_{}_score_{}_thr_{}.csv".format(cnn_type, score, thr), "w")
    out.write("image_name")
    for i in range(len(indexes)):
        out.write("," + indexes[i])
    out.write("\n")
    ids = tbl['image_name'].values
    for i in range(len(preds)):
        out.write(ids[i])
        for j in range(len(indexes)):
            out.write("," + str(preds[i][j]))
        out.write("\n")
    out.close()

    # Create submission
    out = open(OUTPUT_PATH + "subm_{}_score_{}_thr_{}.csv".format(cnn_type, score, thr), "w")
    out.write("image_name,tags\n")
    for i in range(len(files)):
        out.write(ids[i] + ',')
        for j in range(len(indexes)):
            if preds[i][j] > thr:
                out.write(indexes[j] + ' ')
        out.write("\n")
    out.close()


if __name__ == '__main__':
    num_folds = 5
    score1 = ''
    for cnn in ['INCEPTION_V3_DENSE_LAYERS', 'INCEPTION_V4', 'DENSENET_121', 'DENSENET_169', 'DENSENET_161',
                'RESNET50_DENSE_LAYERS', 'RESNET101', 'VGG16', 'VGG19', 'RESNET152', 'XCEPTION']:
        best_score, best_thr = get_validation_score(num_folds, cnn)
        process_test(num_folds, cnn, best_score, best_thr)


'''
Validation history:

INCEPTION_V3_DENSE_LAYERS
Best overall score: 0.9248598060971008 THR: 0.2
Best score fold 1: 0.9237842186349807 THR: 0.2
Best score fold 2: 0.9243091765877415 THR: 0.21
Best score fold 3: 0.9258338706397004 THR: 0.22
Best score fold 4: 0.9253362287668513 THR: 0.2
Best score fold 5: 0.9254969455343668 THR: 0.18

DENSENET_121
Best overall score: 0.9282839676880005 THR: 0.2
Best score fold 1: 0.9259732549616995 THR: 0.18
Best score fold 2: 0.927808702322814 THR: 0.18
Best score fold 3: 0.9302367605882403 THR: 0.21
Best score fold 4: 0.9289837762131035 THR: 0.17
Best score fold 5: 0.9292442752504116 THR: 0.2

DENSENET_169
Best overall score: 0.9255917518310303 THR: 0.2
Best score fold 1: 0.9263497963226633 THR: 0.22
Best score fold 2: 0.9229784022084283 THR: 0.14
Best score fold 3: 0.9273674053370875 THR: 0.24
Best score fold 4: 0.9265559805626908 THR: 0.2
Best score fold 5: 0.9262725947617672 THR: 0.22

DENSENET_161
Best overall score: 0.9269659592740706 THR: 0.21
Best score fold 1: 0.9264609624386172 THR: 0.2
Best score fold 2: 0.9248898149127779 THR: 0.22
Best score fold 3: 0.9289309539548167 THR: 0.21
Best score fold 4: 0.9275478928335408 THR: 0.21
Best score fold 5: 0.9282206246774681 THR: 0.17

INCEPTION_v4
Best overall score: 0.9262189395255209 THR: 0.17
Best score fold 1: 0.922717236909142 THR: 0.18
Best score fold 2: 0.9251909182647715 THR: 0.17
Best score fold 3: 0.9268627144887531 THR: 0.21
Best score fold 4: 0.9280003580757566 THR: 0.17
Best score fold 5: 0.9286064966963335 THR: 0.17

RESNET50_DENSE_LAYERS
Best overall score: 0.9263739674493671 THR: 0.18
Best score fold 1: 0.9239484491111165 THR: 0.17
Best score fold 2: 0.9235899447121498 THR: 0.18
Best score fold 3: 0.9296567172921516 THR: 0.19
Best score fold 4: 0.9282728191281 THR: 0.17
Best score fold 5: 0.9271708073379697 THR: 0.19

RESNET101
Best overall score: 0.9253183157874696 THR: 0.2
Best score fold 1: 0.924428816482149 THR: 0.25
Best score fold 2: 0.9239647724658601 THR: 0.21
Best score fold 3: 0.9280631565793689 THR: 0.24
Best score fold 4: 0.926793153998684 THR: 0.19
Best score fold 5: 0.9247433106035584 THR: 0.16

XCEPTION
Best overall score: 0.9259769103573405 THR: 0.18
Best score fold 1: 0.9247594110616024 THR: 0.18
Best score fold 2: 0.925430670838188 THR: 0.2
Best score fold 3: 0.9297797702591243 THR: 0.22
Best score fold 4: 0.925844795320704 THR: 0.16
Best score fold 5: 0.9250094054431726 THR: 0.18

RESNET152
Best overall score: 0.9275937814558509 THR: 0.18
Best score fold 1: 0.9268510953182838 THR: 0.18
Best score fold 2: 0.9264972708020228 THR: 0.2
Best score fold 3: 0.9283127362941068 THR: 0.16
Best score fold 4: 0.9300323600597092 THR: 0.18
Best score fold 5: 0.9276937670196796 THR: 0.2

VGG16
Best overall score: 0.927268456620429 THR: 0.19
Best score fold 1: 0.925309640350592 THR: 0.19
Best score fold 2: 0.9256756438246051 THR: 0.21
Best score fold 3: 0.9301456100138225 THR: 0.19
Best score fold 4: 0.9265122943092845 THR: 0.2
Best score fold 5: 0.9291713665462222 THR: 0.19

'''