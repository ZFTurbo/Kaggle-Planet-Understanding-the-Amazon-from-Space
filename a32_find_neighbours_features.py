# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import shutil
os.environ["THEANO_FLAGS"] = "device=cpu"
from a00_common_functions import *
from multiprocessing import Process, Manager


INPUT_PATH = "../input/"
OUTPUT_PATH = "../modified_data/"
FEATURES_PATH = "../features/"
if not os.path.isdir(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)


def get_list_of_CNN_features():
    files = glob.glob(FEATURES_PATH + 'valid_*.csv')
    train_files = []
    test_files = []
    for f in files:
        if '_land' in f:
            continue
        if '_weather' in f:
            continue
        if '_single_class' in f:
            continue
        train_files.append(f)
        test_files.append(f.replace('valid_', 'test_'))
        # print(train_files[-1])
        # print(test_files[-1])
    return train_files, test_files


def get_features_for_table(t):
    features = list(set(t.columns.values) - {'image_name'})
    return features


def create_4_neighbours_features(prefix):
    indexes = get_indexes()
    train = pd.read_csv("../input/train_v2.csv")[['image_name']]
    test = pd.read_csv("../input/sample_submission_v2.csv")[['image_name']]
    for i in indexes:
        train[i] = -1
        test[i] = -1
    ne = pd.read_csv('../modified_data/' + prefix + '.csv')
    median = [-1]*len(indexes)

    train_files, test_files = get_list_of_CNN_features()

    print('Reading data...')
    train_s = pd.read_csv(train_files[0])
    for i in range(1, len(train_files)):
        s = pd.read_csv(train_files[i])
        train_s[indexes] += s[indexes]
    train_s[indexes] /= len(train_files)

    test_s = pd.read_csv(test_files[0])
    for i in range(1, len(test_files)):
        s = pd.read_csv(test_files[i])
        test_s[indexes] += s[indexes]
    test_s[indexes] /= len(test_files)

    full = pd.concat((train_s, test_s), axis=0)
    full.reset_index(drop=True, inplace=True)

    for i in range(len(indexes)):
        nm = indexes[i]
        median[i] = np.float64(full[[nm]].median())

    for index, row in ne.iterrows():
        id = row['id']
        val = np.zeros(len(indexes))
        count = 0
        for nm in ['ntop', 'nbottom', 'nleft', 'nright']:
            if row[nm] != 'fake':
                v = full[full['image_name'] == row[nm]][indexes].as_matrix()
                val += v[0]
                count += 1

        # print(id, count)
        if count == 0:
            if 'train_' in id:
                train.loc[train['image_name'] == id, indexes] = median
            else:
                test.loc[test['image_name'] == id, indexes] = median
        else:
            val /= count
            if 'train_' in id:
                train.loc[train['image_name'] == id, indexes] = val
            else:
                test.loc[test['image_name'] == id, indexes] = val

        if (index + 1) % 1000 == 0:
            print('Completed {} from {}'.format(index + 1, len(full)))

    train.to_csv(FEATURES_PATH + "feature_neighbours_4_mean_" + prefix + "_train.csv", index=False)
    test.to_csv(FEATURES_PATH + "feature_neighbours_4_mean_" + prefix + "_test.csv", index=False)


def create_8_neighbours_features(prefix):
    indexes = get_indexes()
    train = pd.read_csv("../input/train_v2.csv")[['image_name']]
    test = pd.read_csv("../input/sample_submission_v2.csv")[['image_name']]
    for i in indexes:
        train[i] = -1
        test[i] = -1
    ne = pd.read_csv('../modified_data/' + prefix + '_8.csv')
    median = [-1]*len(indexes)

    train_files, test_files = get_list_of_CNN_features()

    print('Reading data...')
    train_s = pd.read_csv(train_files[0])
    for i in range(1, len(train_files)):
        s = pd.read_csv(train_files[i])
        train_s[indexes] += s[indexes]
    train_s[indexes] /= len(train_files)

    test_s = pd.read_csv(test_files[0])
    for i in range(1, len(test_files)):
        s = pd.read_csv(test_files[i])
        test_s[indexes] += s[indexes]
    test_s[indexes] /= len(test_files)

    full = pd.concat((train_s, test_s), axis=0)
    full.reset_index(drop=True, inplace=True)

    for i in range(len(indexes)):
        nm = indexes[i]
        median[i] = np.float64(full[[nm]].median())

    for index, row in ne.iterrows():
        id = row['id']
        val = np.zeros(len(indexes))
        count = 0
        for nm in ['ntop', 'nbottom', 'nleft', 'nright', 'ntl', 'ntr', 'nbl', 'nbr']:
            if row[nm] != 'fake':
                v = full[full['image_name'] == row[nm]][indexes].as_matrix()
                val += v[0]
                count += 1

        # print(id, count)
        if count == 0:
            if 'train_' in id:
                train.loc[train['image_name'] == id, indexes] = median
            else:
                test.loc[test['image_name'] == id, indexes] = median
        else:
            val /= count
            if 'train_' in id:
                train.loc[train['image_name'] == id, indexes] = val
            else:
                test.loc[test['image_name'] == id, indexes] = val

        if (index + 1) % 1000 == 0:
            print('Completed {} from {}'.format(index + 1, len(full)))

    train.to_csv(FEATURES_PATH + "feature_neighbours_8_mean_" + prefix + "_train.csv", index=False)
    test.to_csv(FEATURES_PATH + "feature_neighbours_8_mean_" + prefix + "_test.csv", index=False)


def create_group_average_features(prefix):
    indexes = get_indexes()
    train = pd.read_csv("../input/train_v2.csv")[['image_name']]
    test = pd.read_csv("../input/sample_submission_v2.csv")[['image_name']]
    ne = pd.read_csv(OUTPUT_PATH + prefix + '.csv')[['id', 'group_id']]

    train_files, test_files = get_list_of_CNN_features()

    train_s = pd.read_csv(train_files[0])
    for i in range(1, len(train_files)):
        s = pd.read_csv(train_files[i])
        train_s[indexes] += s[indexes]
    train_s[indexes] /= len(train_files)

    test_s = pd.read_csv(test_files[0])
    for i in range(1, len(test_files)):
        s = pd.read_csv(test_files[i])
        test_s[indexes] += s[indexes]
    test_s[indexes] /= len(test_files)

    full = pd.concat((train_s, test_s), axis=0)
    full.reset_index(drop=True, inplace=True)

    ne['image_name'] = ne['id']
    ne = pd.merge(ne, full, on='image_name', left_index=True)
    ne.reset_index(drop=True, inplace=True)

    gr = ne.groupby(['group_id']).median().reset_index()
    final_table = pd.merge(ne[['image_name', 'group_id']], gr, on='group_id', left_index=True)
    final_table.reset_index(drop=True, inplace=True)
    final_table.drop('group_id', axis=1, inplace=True)

    train = pd.merge(train, final_table, on='image_name', left_index=True)
    test = pd.merge(test, final_table, on='image_name', left_index=True)

    train.to_csv(FEATURES_PATH + "feature_neighbours_g_mean_" + prefix + "_train.csv", index=False)
    test.to_csv(FEATURES_PATH + "feature_neighbours_g_mean_" + prefix + "_test.csv", index=False)


if __name__ == '__main__':
    start_time = time.time()
    create_4_neighbours_features('neighbours_merged')
    create_8_neighbours_features('neighbours_merged')
    create_4_neighbours_features('neighbours_merged_cleaned')
    create_8_neighbours_features('neighbours_merged_cleaned')
    create_group_average_features('neighbours_merged_with_groups')
    create_group_average_features('neighbours_merged_cleaned')
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))


'''
JPG: Fakes in 251668 positions out of 406680 (61.88%)
TIFF: Fakes in 247976 positions out of 406680 (60.98%)
Merge: Fakes in 181335 positions out of 406680 (44.59%)
Merge cleaned: Fakes in 259874 positions out of 406680 (63.9%)
'''


