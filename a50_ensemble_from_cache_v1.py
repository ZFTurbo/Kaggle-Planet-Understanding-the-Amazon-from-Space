# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import shutil
from a00_common_functions import *
from a00_common_functions_scores import *
from scipy.stats.mstats import gmean
from a42_gbm_blender import get_optimal_score_very_fast, get_optimal_score_very_fast_for_full_array

INPUT_PATH = "../input/"
OUTPUT_PATH = "../subm/"
CACHE_PATH = "../cache/"


def ensemble_from_cache_v1(restore_from_cache_file):

    indexes = get_indexes()
    lbl = get_train_label_matrix()
    train = pd.read_csv(INPUT_PATH + 'train_v2.csv')
    log_list = []

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
        scores_full_arr = scores_full_arr[0:only_1_best_subm]
        validation_full_arr = validation_full_arr[0:only_1_best_subm]
        test_preds_full_arr = test_preds_full_arr[0:only_1_best_subm]
        params_full_arr = params_full_arr[0:only_1_best_subm]
        thr_full_arr = thr_full_arr[0:only_1_best_subm]

    for i in range(1, len(restore_from_cache_file)):
        validation_full_arr1, test_preds_full_arr1, scores_full_arr1, params_full_arr1, thr_full_arr1 = load_from_file(restore_from_cache_file[i])
        scores_full_arr1 = np.array(scores_full_arr1)
        validation_full_arr1 = np.array(validation_full_arr1)
        test_preds_full_arr1 = np.array(test_preds_full_arr1)
        params_full_arr1 = np.array(params_full_arr1)
        thr_full_arr1 = np.array(thr_full_arr1)

        # Experiment with top 1 score:
        if only_1_best_subm:
            scores_full_arr1 = scores_full_arr1[0:only_1_best_subm]
            validation_full_arr1 = validation_full_arr1[0:only_1_best_subm]
            test_preds_full_arr1 = test_preds_full_arr1[0:only_1_best_subm]
            params_full_arr1 = params_full_arr1[0:only_1_best_subm]
            thr_full_arr1 = thr_full_arr1[0:only_1_best_subm]

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
    best_score, searcher = get_optimal_score_very_fast(indexes, lbl, validation_arr, 7, 0.000001, 3)
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

    sub_file = OUTPUT_PATH + "merger_final_{}.csv".format(best_score)
    out = open(sub_file, "w")
    out.write("image_name,tags\n")
    for i in range(len(files)):
        out.write(ids[i] + ',')
        for j in range(len(indexes)):
            if preds[i][j] > searcher[j]:
                out.write(indexes[j] + ' ')
        out.write("\n")
    out.close()
    print('File with predictions was written in {}'.format(sub_file))

    # Save raw test for analysis
    sub_file_raw = OUTPUT_PATH + "merger_cache_{}_raw_test.csv".format(best_score)
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
    sub_file_raw = OUTPUT_PATH + "merger_cache_{}_raw_valid.csv".format(best_score)
    out = open(sub_file_raw, "w")
    out.write("image_name")
    for i in indexes:
        out.write("," + str(i))
    out.write("\n")

    ids = list(train['image_name'].values)
    for i in range(validation_arr.shape[0]):
        out.write(ids[i])
        for j in range(len(validation_arr[i])):
            out.write(',' + str(validation_arr[i][j]))
        out.write("\n")
    out.close()


if __name__ == '__main__':
    start_time = time.time()

    # Currently used 150 XGBoost + 100 Keras runs. But there can be more
    # effective proportions like 150 vs 150 or 150 vs 80 etc.

    cache_files = glob.glob(CACHE_PATH + 'gbm_cache_iter_150_score*.pklz') + \
                  glob.glob(CACHE_PATH + 'keras_cache_iter_100_score*.pklz')

    ensemble_from_cache_v1(cache_files)
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))
