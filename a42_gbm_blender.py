# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
# os.environ["THEANO_FLAGS"] = "device=cpu"
import glob
import time
import cv2
import pandas as pd
import datetime
from a00_common_functions import *
from a00_common_functions_scores import *
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from scipy.stats.mstats import gmean
import xgboost as xgb
import shutil

random.seed(2017)
np.random.seed(2017)

INPUT_PATH = "../input/"
OUTPUT_PATH = "../subm/"
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
CACHE_PATH = "../cache/"
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
FEATURES_PATH = "../features/"


def get_features_for_table(t):
    features = list(set(t.columns.values) - {'image_name'})
    return features


def xgboost_random_step(random_state, iter, lbl, indexes, train_files, train_s, test_s, matrix, X_test, additional_train, additional_test, add_features):
    start_time = time.time()
    rs = random_state + iter

    num_folds = random.randint(4, 10)
    eta = random.uniform(0.06, 0.45)
    max_depth = random.randint(2, 5)
    subsample = random.uniform(0.6, 0.99)
    colsample_bytree = random.uniform(0.6, 0.99)
    # eval_metric = random.choice(['auc', 'logloss'])
    eval_metric = 'logloss'
    # eta = 0.1
    # max_depth = 3
    # subsample = 0.95
    # colsample_bytree = 0.95
    log_str = 'XGBoost iter {}. FOLDS: {} METRIC: {} ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(iter,
                                                                                                           num_folds,
                                                                                                           eval_metric,
                                                                                                           eta,
                                                                                                           max_depth,
                                                                                                           subsample,
                                                                                                           colsample_bytree)
    print(log_str)
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": eval_metric,
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": rs,
        "nthread": 6,
        # 'gpu_id': 0,
        # 'updater': 'grow_gpu_hist',
    }
    num_boost_round = 1000
    early_stopping_rounds = 40

    validation_arr = np.zeros(lbl.shape)
    validation_arr[:, :] = -1

    test_preds = np.zeros((num_folds, len(test_s[0]), len(indexes)))
    # model_list = [[]]*len(indexes)
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

        # Additional
        if additional_train is not None:
            X_train = np.concatenate((X_train, additional_train[add_features].as_matrix()[train_index]), axis=1)
            X_valid = np.concatenate((X_valid, additional_train[add_features].as_matrix()[test_index]), axis=1)

        y_train = lbl[train_index]
        y_valid = lbl[test_index]
        # print('Len train: ', X_train.shape)
        # print('Len valid: ', X_valid.shape)

        random_skip = random.sample(list(range(len(train_files))), 1)
        for i in range(len(indexes)):
            dtrain = xgb.DMatrix(X_train, y_train[:, i])
            dvalid = xgb.DMatrix(X_valid, y_valid[:, i])

            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                            early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

            # print("Validating...")
            preds = gbm.predict(dvalid, ntree_limit=gbm.best_iteration + 1)
            test_preds[num_fold - 1, :, i] = gbm.predict(matrix, ntree_limit=gbm.best_iteration + 1)
            validation_arr[test_index, i] = preds
            # model_list[i].append((gbm, gbm.best_iteration))
    print("Time XGBoost: %s sec" % (round(time.time() - start_time, 0)))
    return validation_arr, test_preds, log_str, params


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


def get_additional_features(train_s):
    # Additional features
    sample_subm = pd.read_csv(INPUT_PATH + "sample_submission_v2.csv")
    additional_data = pd.read_csv(FEATURES_PATH + "neighbours_merged_cleaned_fix_train.csv")
    additional_train = pd.merge(train_s[0][['image_name']], additional_data, on='image_name', left_index=True)
    additional_data = pd.read_csv(FEATURES_PATH + "neighbours_merged_with_groups_fix_train.csv")
    additional_train = pd.merge(additional_train, additional_data, on='image_name', left_index=True)
    additional_train.reset_index(drop=True, inplace=True)
    additional_data = pd.read_csv(FEATURES_PATH + "neighbours_merged_cleaned_fix_test.csv")
    additional_test = pd.merge(sample_subm[['image_name']], additional_data, on='image_name', left_index=True)
    additional_data = pd.read_csv(FEATURES_PATH + "neighbours_merged_with_groups_fix_test.csv")
    additional_test = pd.merge(additional_test, additional_data, on='image_name', left_index=True)
    additional_test.reset_index(drop=True, inplace=True)

    return additional_train, additional_test


def multiple_gbm_runs(train_files, test_files, required_iterations):

    restore_from_cache_file = ''
    random_state = 2006
    random.seed(random_state)

    log_list = []
    indexes = get_indexes()
    lbl = get_train_label_matrix()
    train_s = [0]*len(train_files)
    test_s = [0]*len(test_files)

    for i in range(len(train_files)):
        train_s[i] = pd.read_csv(train_files[i])
        feat = get_features_for_table(train_s[i])
        if (np.isnan(np.sum(train_s[i][feat].as_matrix())) == True):
            print('Some problem with ', train_files[i], 'check it!')
            exit()
    for i in range(len(test_files)):
        test_s[i] = pd.read_csv(test_files[i])
        feat = get_features_for_table(test_s[i])
        if (np.isnan(np.sum(test_s[i][feat].as_matrix())) == True):
            print('Some problem with ', test_files[i], 'check it!')
            exit()

    feat = get_features_for_table(test_s[0])
    X_test = test_s[0][feat].as_matrix()
    for i in range(1, len(test_files)):
        feat = get_features_for_table(test_s[i])
        X_test = np.concatenate((X_test, test_s[i][feat].as_matrix()), axis=1)
    print('Test shape:', X_test.shape)

    additional_train, additional_test = get_additional_features(train_s)

    add_features = list(additional_test.columns.values)
    add_features.remove('image_name')
    at = additional_test[add_features].as_matrix()
    X_test = np.concatenate((X_test, at), axis=1)

    matrix_xgboost = xgb.DMatrix(X_test)

    if not os.path.isfile(restore_from_cache_file):
        validation_full_arr = []
        test_preds_full_arr = []
        scores_full_arr = []
        params_full_arr = []
        thr_full_arr = []
        for iter in range(required_iterations):
            validation_arr, test_preds, log_str, params = xgboost_random_step(random_state, iter, lbl, indexes, train_files, train_s, test_s, matrix_xgboost, X_test, additional_train, additional_test, add_features)
            validation_full_arr.append(validation_arr.copy())
            test_preds_full_arr.append(test_preds.mean(axis=0))
            log_list.append(log_str)

            best_score, searcher = get_optimal_score_very_fast(indexes, lbl, validation_arr, 7, 0.001, 1)
            log_str = 'Best score {} for THR array: {}'.format(best_score, list(searcher))
            print(log_str)
            log_list.append(log_str)
            scores_full_arr.append(best_score)
            thr_full_arr.append(searcher)
            params_full_arr.append(params)
            # exit()
            if iter > 0 and iter % 10 == 0:
                avg_score = np.array(scores_full_arr).mean()
                save_in_file((validation_full_arr, test_preds_full_arr, scores_full_arr, params_full_arr, thr_full_arr), CACHE_PATH + "gbm_cache_iter_{}_from_{}_score_{}.pklz".format(iter, required_iterations, avg_score))

        avg_score = np.array(scores_full_arr).mean()
        out_cache_file = CACHE_PATH + "gbm_cache_iter_{}_score_{}.pklz".format(required_iterations, avg_score)
        print('Write final cache file: {}'.format(out_cache_file))
        save_in_file((validation_full_arr, test_preds_full_arr, scores_full_arr, params_full_arr, thr_full_arr), out_cache_file)
    else:
        print('Restore from cache file: {}'.format(restore_from_cache_file))
        validation_full_arr, test_preds_full_arr, scores_full_arr, params_full_arr, thr_full_arr = load_from_file(restore_from_cache_file)

    scores_full_arr = np.array(scores_full_arr)
    validation_full_arr = np.array(validation_full_arr)
    test_preds_full_arr = np.array(test_preds_full_arr)
    condition = scores_full_arr > 0.931
    print('Left {} out of {} runs'.format(len(scores_full_arr[condition]), len(scores_full_arr)))
    validation_full_arr = validation_full_arr[condition]
    test_preds_full_arr = test_preds_full_arr[condition]

    # Experimental version
    if 0:
        best_score, searcher = get_optimal_score_very_fast_for_full_array(indexes, lbl, validation_full_arr, 7, 0.00001, 3)
        log_str = 'Best score {} for THR array: {}'.format(best_score, list(searcher))
        print(log_str)
        log_list.append(log_str)

        # Create submission
        tbl = pd.read_csv(INPUT_PATH + "sample_submission_v2.csv")
        ids = tbl['image_name'].values
        files = []
        for id in ids:
            files.append(INPUT_PATH + "test-jpg/" + id + '.jpg')
        files = np.array(files)

        preds = test_preds_full_arr

        sub_file = OUTPUT_PATH + "merger_xgboost_v2_{}.csv".format(best_score)
        out = open(sub_file, "w")
        out.write("image_name,tags\n")
        for i in range(len(files)):
            out.write(ids[i] + ',')
            for j in range(len(indexes)):
                cond = preds[:, i, j] > searcher[j]
                cond = 2 * np.sum(cond, axis=0)
                if cond >= preds.shape[0]:
                    out.write(indexes[j] + ' ')
            out.write("\n")
        out.close()

    # Normal version
    else:
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

        sub_file = OUTPUT_PATH + "merger_xgboost_{}.csv".format(best_score)
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
        sub_file_raw = OUTPUT_PATH + "merger_xgboost_{}_raw_test.csv".format(best_score)
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
        sub_file_raw = OUTPUT_PATH + "merger_xgboost_{}_raw_valid.csv".format(best_score)
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

    # Copy code
    shutil.copy2(__file__, sub_file + ".py")

    # Save log
    out = open(sub_file + ".log", "w")
    for l in log_list:
        out.write(l + '\n')
    out.close()


if __name__ == '__main__':
    start_time = time.time()
    required_iterations = 150
    valid_files, test_files = get_feature_files()
    multiple_gbm_runs(valid_files, test_files, required_iterations)
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))
