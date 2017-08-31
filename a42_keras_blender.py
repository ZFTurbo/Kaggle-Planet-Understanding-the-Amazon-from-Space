# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import platform
# os.environ["THEANO_FLAGS"] = "device=cpu"
gpu_use = 0
if platform.processor() == 'Intel64 Family 6 Model 63 Stepping 2, GenuineIntel':
    os.environ["THEANO_FLAGS"] = "device=gpu{},lib.cnmem=0.81,,base_compiledir='C:\\\\Users\\\\user\\\\AppData\\\\Local\\\\Theano{}'".format(gpu_use, gpu_use)

import glob
import time
import cv2
import pandas as pd
import datetime
import shutil
from a00_common_functions import *
from a00_common_functions_scores import *
from sklearn.metrics import accuracy_score
from scipy.stats.mstats import gmean
from sklearn.model_selection import KFold
from a42_gbm_blender import get_optimal_score_very_fast, get_optimal_score_very_fast_for_full_array

INPUT_PATH = "../input/"
OUTPUT_PATH = "../subm/"
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
MODELS_PATH = "../models/"
CACHE_PATH = "../cache/"
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
FEATURES_PATH = "../features/"
HISTORY_FOLDER_PATH = "../models/history2/"


def Blender_CNN(num_of_res, num_of_inp, params):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten, Activation
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU

    level1_size = 300
    if 'level1_size' in params:
        level1_size = params['level1_size']
    level2_size = 300
    if 'level2_size' in params:
        level2_size = params['level2_size']
    level3_size = 250
    if 'level3_size' in params:
        level3_size = params['level3_size']

    dropout_val_1 = 0.1
    if 'dropout_val_1' in params:
        dropout_val_1 = params['dropout_val_1']
    dropout_val_2 = 0.1
    if 'dropout_val_2' in params:
        dropout_val_2 = params['dropout_val_2']
    dropout_val_3 = 0.1
    if 'dropout_val_3' in params:
        dropout_val_3 = params['dropout_val_3']

    activation_1 = 'prelu'
    if 'activation_1' in params:
        activation_1 = params['activation_1']
    activation_2 = 'prelu'
    if 'activation_2' in params:
        activation_2 = params['activation_2']
    activation_3 = 'prelu'
    if 'activation_3' in params:
        activation_3 = params['activation_3']

    use_3rd_level = 1
    if 'use_3rd_level' in params:
        use_3rd_level = params['use_3rd_level']

    model = Sequential()
    model.add(Dense(level1_size, input_shape=(num_of_inp,)))
    if activation_1 == 'prelu':
        model.add(PReLU())
    elif activation_1 == 'relu':
        model.add(Activation('relu'))
    else:
        model.add(Activation('elu'))
    model.add(Dropout(dropout_val_1))
    model.add(Dense(level2_size))
    if activation_2 == 'prelu':
        model.add(PReLU())
    elif activation_2 == 'relu':
        model.add(Activation('relu'))
    else:
        model.add(Activation('elu'))
    model.add(Dropout(dropout_val_2))
    if use_3rd_level == 1:
        model.add(Dense(level3_size))
        if activation_3 == 'prelu':
            model.add(PReLU())
        elif activation_3 == 'relu':
            model.add(Activation('relu'))
        else:
            model.add(Activation('elu'))
        model.add(Dropout(dropout_val_3))
    model.add(Dense(num_of_res, activation='sigmoid'))
    return model


def batch_generator_train(X, y, batch_size):
    rng = range(X.shape[0])

    while True:
        index = random.sample(rng, batch_size)
        input = X[index, :]
        output = y[index, :]

        yield input, output


def random_keras_step(random_state, iter, lbl, indexes, train_files, train_s, test_s, X_test_transform):
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.optimizers import Adam, SGD, Adadelta

    start_time = time.time()
    rs = random_state + iter

    num_folds = random.randint(4, 10)
    batch_size = random.randint(200, 1000)

    patience = random.randint(50, 150)
    learning_rate = random.uniform(0.00001, 0.001)
    # roptim = random.choice(['Adam', 'SGD', 'Adadelta'])
    roptim = 'Adam'
    if 0:
        roptim = 'Adam'
        learning_rate = 0.001
        batch_size = 210

    # metric = random.choice(['binary_crossentropy', 'f2beta_loss'])
    metric = random.choice(['binary_crossentropy'])
    cnn_param = dict()
    cnn_param['dropout_val_1'] = random.uniform(0.05, 0.5)
    cnn_param['dropout_val_2'] = random.uniform(0.1, 0.5)
    cnn_param['dropout_val_3'] = random.uniform(0.1, 0.5)
    cnn_param['level1_size'] = random.randint(400, 700)
    cnn_param['level2_size'] = random.randint(350, 600)
    cnn_param['level3_size'] = random.randint(200, 500)
    cnn_param['activation_1'] = random.choice(['prelu', 'relu', 'elu'])
    cnn_param['activation_2'] = random.choice(['prelu', 'relu', 'elu'])
    cnn_param['activation_3'] = random.choice(['prelu', 'relu', 'elu'])
    # cnn_param['use_3rd_level'] = random.choice([0, 1])
    # cnn_param['use_3rd_level'] = random.choice([0, 1])
    # cnn_param['activation'] = 'relu'

    if 0:
        cnn_param['level1_size'] = 1000
        cnn_param['level2_size'] = 1000
        cnn_param['level3_size'] = 1000
        cnn_param['use_3rd_level'] = 0
        cnn_param['dropout_val_1'] = 0
        cnn_param['dropout_val_2'] = 0.5
        cnn_param['dropout_val_3'] = 0.5

    log_str = 'Keras iter {}. FOLDS: {} LR: {}, PATIENCE: {}, OPTIM: {}, METRIC: {}, BATCH: {}'.format(
        iter,
        num_folds,
        learning_rate,
        patience,
        roptim,
        metric,
        batch_size)
    print(log_str)
    print('CNN params: {}'.format(cnn_param))

    validation_arr = np.zeros(lbl.shape)
    validation_arr[:, :] = -1

    test_preds = np.zeros((num_folds, len(test_s[0]), len(indexes)))
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=rs)
    num_fold = 0
    for train_index, test_index in kf.split(list(range(len(train_s[0])))):
        num_fold += 1
        # print('Start fold {} from {}'.format(num_fold, num_folds))
        feat = get_features_for_table(train_s[0])
        X_train = train_s[0][feat].as_matrix()[train_index]
        X_valid = train_s[0][feat].as_matrix()[test_index]
        for i in range(1, len(train_files)):
            feat = get_features_for_table(train_s[i])
            X_train = np.concatenate((X_train, train_s[i][feat].as_matrix()[train_index]), axis=1)
            X_valid = np.concatenate((X_valid, train_s[i][feat].as_matrix()[test_index]), axis=1)
        y_train = lbl[train_index]
        y_valid = lbl[test_index]
        # print('Shape train: ', X_train.shape)
        # print('Shape valid: ', X_valid.shape)

        final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format("blender", num_fold)
        cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format("blender", num_fold)

        model = Blender_CNN(len(indexes), X_train.shape[1], cnn_param)
        if roptim == 'Adam':
            optim = Adam(lr=learning_rate)
        elif roptim == 'SGD':
            optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        elif roptim == 'Adadelta':
            optim = Adadelta()
        else:
            print('Unknown optimizer {}!'.format(roptim))
            exit()

        # optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=optim, loss=f2beta_loss, metrics=['binary_crossentropy', 'accuracy'])
        # model.compile(optimizer=optim, loss=f2beta_loss, metrics=['binary_crossentropy'])
        if metric == 'binary_crossentropy':
            model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f2beta_loss])
        else:
            model.compile(optimizer=optim, loss=f2beta_loss, metrics=['binary_crossentropy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
            ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        ]

        samples_per_epoch = batch_size * (1 + X_train.shape[0] // batch_size)
        nb_val_samples = batch_size * (1 + X_valid.shape[0] // batch_size)
        history = model.fit_generator(generator=batch_generator_train(X_train, y_train, batch_size),
                                      nb_epoch=10000,
                                      samples_per_epoch=samples_per_epoch,
                                      validation_data=batch_generator_train(X_valid, y_valid, batch_size),
                                      nb_val_samples=nb_val_samples,
                                      verbose=0, max_q_size=100,
                                      callbacks=callbacks)

        min_loss = min(history.history['val_loss'])
        print('Loss fold {}: {}'.format(num_fold, min_loss))
        model.load_weights(cache_model_path)
        model.save(final_model_path)
        now = datetime.datetime.now()
        filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}.csv'.format("blender", num_fold, min_loss,
                                                                                    learning_rate,
                                                                                    now.strftime("%Y-%m-%d-%H-%M"))
        # pd.DataFrame(history.history).to_csv(filename, index=False)

        preds = model.predict(X_valid)
        validation_arr[test_index] = preds
        test_preds[num_fold - 1] = model.predict(X_test_transform)

    print("Time Keras: %s sec" % (round(time.time() - start_time, 0)))
    return validation_arr, test_preds, log_str, cnn_param


def get_features_for_table(t):
    features = list(set(t.columns.values) - {'image_name'})
    return features


def run_multiple_keras_blenders(train_files, test_files, required_iterations):

    restore_from_cache_file = []
    random_state = 1002
    indexes = get_indexes()
    lbl = get_train_label_matrix()
    train_s = [0]*len(train_files)
    test_s = [0]*len(test_files)
    log_list = []

    for i in range(len(train_files)):
        train_s[i] = pd.read_csv(train_files[i])
    for i in range(len(test_files)):
        test_s[i] = pd.read_csv(test_files[i])

    # Prepare test array
    feat = get_features_for_table(test_s[0])
    X_test_transform = test_s[0][feat].as_matrix()
    for i in range(1, len(test_files)):
        feat = get_features_for_table(test_s[i])
        X_test_transform = np.concatenate((X_test_transform, test_s[i][feat].as_matrix()), axis=1)
    print('Shape:', X_test_transform.shape)

    if len(restore_from_cache_file) == 0:
        validation_full_arr = []
        test_preds_full_arr = []
        scores_full_arr = []
        params_full_arr = []
        thr_full_arr = []
        for iter in range(required_iterations):
            validation_arr, test_preds, log_str, cnn_param = random_keras_step(random_state, iter, lbl, indexes, train_files, train_s, test_s, X_test_transform)
            validation_full_arr.append(validation_arr.copy())
            test_preds_full_arr.append(test_preds.mean(axis=0))
            params_full_arr.append(cnn_param)
            log_list.append(log_str)

            best_score, searcher = get_optimal_score_very_fast(indexes, lbl, validation_arr, 7, 0.001, 1)
            log_str = 'Best score {} for THR array: {}'.format(best_score, list(searcher))
            print(log_str)
            log_list.append(log_str)
            scores_full_arr.append(best_score)
            thr_full_arr.append(searcher)
            if iter > 0 and iter % 10 == 0:
                avg_score = np.array(scores_full_arr).mean()
                save_in_file((validation_full_arr, test_preds_full_arr, scores_full_arr, params_full_arr, thr_full_arr),
                             CACHE_PATH + "keras_cache_iter_{}_from_{}_score_{}.pklz".format(iter, required_iterations, avg_score))

        avg_score = np.array(scores_full_arr).mean()
        out_cache_file = CACHE_PATH + "keras_cache_iter_{}_score_{}.pklz".format(required_iterations, avg_score)
        print('Write final cache file: {}'.format(out_cache_file))
        save_in_file((validation_full_arr, test_preds_full_arr, scores_full_arr, params_full_arr, thr_full_arr), out_cache_file)
    else:
        only_1_best_subm = 0
        print('Restore from cache file: {}'.format(restore_from_cache_file))
        validation_full_arr, test_preds_full_arr, scores_full_arr, params_full_arr, thr_full_arr = load_from_file(restore_from_cache_file[0])

        scores_full_arr = np.array(scores_full_arr)
        validation_full_arr = np.array(validation_full_arr)
        test_preds_full_arr = np.array(test_preds_full_arr)
        params_full_arr = np.array(params_full_arr)
        thr_full_arr = np.array(thr_full_arr)

        # Experiment with top 1 score:
        if only_1_best_subm:
            mx = scores_full_arr.max()
            cond = scores_full_arr >= mx
            scores_full_arr = scores_full_arr[cond]
            validation_full_arr = validation_full_arr[cond]
            test_preds_full_arr = test_preds_full_arr[cond]
            params_full_arr = params_full_arr[cond]
            thr_full_arr = thr_full_arr[cond]

        for i in range(1, len(restore_from_cache_file)):
            validation_full_arr1, test_preds_full_arr1, scores_full_arr1, params_full_arr1, thr_full_arr1 = load_from_file(restore_from_cache_file[i])
            scores_full_arr1 = np.array(scores_full_arr1)
            validation_full_arr1 = np.array(validation_full_arr1)
            test_preds_full_arr1 = np.array(test_preds_full_arr1)
            params_full_arr1 = np.array(params_full_arr1)
            thr_full_arr1 = np.array(thr_full_arr1)

            # Experiment with top 1 score:
            if only_1_best_subm:
                mx = scores_full_arr1.max()
                cond = scores_full_arr1 >= mx
                scores_full_arr1 = scores_full_arr1[cond]
                validation_full_arr1 = validation_full_arr1[cond]
                test_preds_full_arr1 = test_preds_full_arr1[cond]
                params_full_arr1 = params_full_arr1[cond]
                thr_full_arr1 = thr_full_arr1[cond]

            validation_full_arr = np.concatenate((validation_full_arr, validation_full_arr1), axis=0)
            scores_full_arr = np.concatenate((scores_full_arr, scores_full_arr1), axis=0)
            test_preds_full_arr = np.concatenate((test_preds_full_arr, test_preds_full_arr1), axis=0)
            params_full_arr = np.concatenate((params_full_arr, params_full_arr1), axis=0)
            thr_full_arr = np.concatenate((thr_full_arr, thr_full_arr1), axis=0)

        print(validation_full_arr.shape)
        print(scores_full_arr.shape)
        print(test_preds_full_arr.shape)
        print(params_full_arr.shape)
        print(thr_full_arr.shape)


    scores_full_arr = np.array(scores_full_arr)
    validation_full_arr = np.array(validation_full_arr)
    test_preds_full_arr = np.array(test_preds_full_arr)
    condition = scores_full_arr > 0.931
    print('Left {} out of {} runs'.format(len(scores_full_arr[condition]), len(scores_full_arr)))
    validation_full_arr = validation_full_arr[condition]
    test_preds_full_arr = test_preds_full_arr[condition]

    if 1:
        validation_arr = np.mean(np.array(validation_full_arr), axis=0)
        test_preds = np.mean(np.array(test_preds_full_arr), axis=0)
    else:
        validation_arr = gmean(validation_full_arr, axis=0)
        test_preds = gmean(test_preds_full_arr, axis=0)

    if np.count_nonzero(validation_arr < 0) > 0:
        print('Some error here..')
        exit()

    # Check validation
    best_score, searcher = get_optimal_score_very_fast(indexes, lbl, validation_arr, 7, 0.00001, 3)
    # best_score, searcher = get_optimal_score_slow(indexes, lbl, validation_arr)
    log_str = 'Best score {} for THR array: {}'.format(best_score, list(searcher))
    print(log_str)
    log_list.append(log_str)

    preds = test_preds

    # Create submission
    tbl = pd.read_csv(INPUT_PATH + "sample_submission_v2.csv")
    ids = tbl['image_name'].values
    files = []
    for id in ids:
        files.append(INPUT_PATH + "test-jpg/" + id + '.jpg')
    files = np.array(files)

    sub_file = OUTPUT_PATH + "merger_keras_{}.csv".format(best_score)
    out = open(sub_file, "w")
    out.write("image_name,tags\n")
    for i in range(len(files)):
        out.write(ids[i] + ',')
        for j in range(len(indexes)):
            if preds[i][j] > searcher[j]:
                out.write(indexes[j] + ' ')
        out.write("\n")
    out.close()

    # Save raw test for analysis
    sub_file_raw = OUTPUT_PATH + "merger_keras_{}_raw_test.csv".format(best_score)
    out = open(sub_file_raw, "w")
    out.write("image_name")
    for i in indexes:
        out.write("," + str(i))
    out.write("\n")

    for i in range(len(files)):
        out.write(ids[i])
        for j in range(len(preds[i])):
            out.write(',' + str(preds[i][j]))
        out.write("\n")
    out.close()

    # Save raw validation for further analysis
    sub_file_raw = OUTPUT_PATH + "merger_keras_{}_raw_valid.csv".format(best_score)
    out = open(sub_file_raw, "w")
    out.write("image_name")
    for i in indexes:
        out.write("," + str(i))
    out.write("\n")

    ids = list(train_s[0]['image_name'].values)
    for i in range(validation_arr.shape[0]):
        out.write(ids[i])
        for j in range(len(validation_arr[i])):
            out.write(',' + str(validation_arr[i][j]))
        out.write("\n")
    out.close()


def get_feature_files():
    valid_files = []
    test_files = []

    print('List of feature files: ')
    files = glob.glob(FEATURES_PATH + "valid_*.csv")
    for f in files:
        valid_files.append(f)
        test_files.append(f.replace('valid_', 'test_'))
        print(valid_files[-1])
        print(test_files[-1])

    files = glob.glob(FEATURES_PATH + "feature_*_train.csv")
    for f in files:
        valid_files.append(f)
        test_files.append(f.replace('_train', '_test'))
        print(valid_files[-1])
        print(test_files[-1])

    return valid_files, test_files


if __name__ == '__main__':
    start_time = time.time()
    required_iterations = 100
    valid_files, test_files = get_feature_files()
    run_multiple_keras_blenders(valid_files, test_files, required_iterations)
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))
