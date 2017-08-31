# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import cv2
import numpy as np
import tifffile
import glob
import pandas as pd
import time
from a00_common_functions import save_in_file, load_from_file
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Process, cpu_count


INPUT_PATH = "../input/"
OUTPUT_PATH = "../modified_data/"
FEATURES_PATH = "../features/"
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
if not os.path.isdir(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)
if not os.path.isdir(OUTPUT_PATH + 'csim'):
    os.mkdir(OUTPUT_PATH + 'csim')
if not os.path.isdir(OUTPUT_PATH + 'csim_tiff'):
    os.mkdir(OUTPUT_PATH + 'csim_tiff')

# Set to 1 in case you want images to be generated
CREATE_PANORAMAS_AND_OTHER_IMAGES = 0
THREADS_NUM = 5

BOTTOM = 0
TOP = 1
LEFT = 2
RIGHT = 3


def get_borders(files):
    cache_file = OUTPUT_PATH + "borders.pklz"
    if not os.path.isfile(cache_file):
        result = np.zeros((4, len(files), 3*256), dtype=np.uint8)
        for i in range(len(files)):
            f = files[i]
            im2 = cv2.imread(f)

            pixel_top = im2[0, :, :].flatten()
            pixel_bottom = im2[im2.shape[0] - 1, :, :].flatten()
            pixel_left = im2[:, 0, :].flatten()
            pixel_right = im2[:, im2.shape[1] - 1, :].flatten()

            result[TOP, i, :] = pixel_top
            result[BOTTOM, i, :] = pixel_bottom
            result[LEFT, i, :] = pixel_left
            result[RIGHT, i, :] = pixel_right
        save_in_file(result, cache_file)
    else:
        print('Restore borders from cache...')
        result = load_from_file(cache_file)
    return result


def get_borders_tiff(files):
    cache_file = OUTPUT_PATH + "borders_tiff.pklz"
    if not os.path.isfile(cache_file):
        result = np.zeros((4, len(files), 4*256), dtype=np.uint16)
        for i in range(len(files)):
            f = files[i]
            print(f)
            im2 = tifffile.imread(f)

            pixel_top = im2[0, :, :].flatten()
            pixel_bottom = im2[im2.shape[0] - 1, :, :].flatten()
            pixel_left = im2[:, 0, :].flatten()
            pixel_right = im2[:, im2.shape[1] - 1, :].flatten()

            result[TOP, i, :] = pixel_top
            result[BOTTOM, i, :] = pixel_bottom
            result[LEFT, i, :] = pixel_left
            result[RIGHT, i, :] = pixel_right
        save_in_file(result, cache_file)
    else:
        print('Restore borders from cache...')
        result = load_from_file(cache_file)
    return result


# Based on cosine sim
def fill_neighbours_v1(result, i):
    cs = [0] * 4
    cs[TOP] = cosine_similarity(result[TOP][i:i+1], result[BOTTOM])
    cs[BOTTOM] = cosine_similarity(result[BOTTOM][i:i+1], result[TOP])
    cs[RIGHT] = cosine_similarity(result[RIGHT][i:i+1], result[LEFT])
    cs[LEFT] = cosine_similarity(result[LEFT][i:i+1], result[RIGHT])
    return cs


# Based on minimum absolute difference
def fill_neighbours_v2(result, i):
    cs = [0] * 4
    cs[TOP] = np.expand_dims(-np.sum(np.abs(result[BOTTOM] - result[TOP][i]), axis=1), axis=0)
    cs[BOTTOM] = np.expand_dims(-np.sum(np.abs(result[TOP] - result[BOTTOM][i]), axis=1), axis=0)
    cs[RIGHT] = np.expand_dims(-np.sum(np.abs(result[LEFT] - result[RIGHT][i]), axis=1), axis=0)
    cs[LEFT] = np.expand_dims(-np.sum(np.abs(result[RIGHT] - result[LEFT][i]), axis=1), axis=0)
    return cs


def find_neightbours_part(csv_file, files, result, start, end):
    print('Start writing: {}'.format(csv_file))
    out = open(csv_file, "w")
    out.write('id,ntop,nbottom,nleft,nright,val_top,val_bottom,val_left,val_right\n')
    nindex = np.zeros((4, len(files)), dtype=np.int64)
    nvalue = np.zeros((4, len(files)), dtype=np.float64)
    part = 1
    for i in range(start, end, part):
        # if 'file_' not in files[i]:
        #     continue
        cs = fill_neighbours_v2(result, i)
        for j in range(4):
            nindex[j, i * part:(i + 1) * part] = np.argmax(cs[j], axis=1)
            nvalue[j, i * part:(i + 1) * part] = np.max(cs[j], axis=1)
        for j in range(part):
            if 0:
                print(nvalue[:, i + j],
                  os.path.basename(files[i + j]),
                  os.path.basename(files[nindex[BOTTOM, i + j]]),
                  os.path.basename(files[nindex[TOP, i + j]]),
                  os.path.basename(files[nindex[LEFT, i + j]]),
                  os.path.basename(files[nindex[RIGHT, i + j]])
                  )
            out.write(os.path.basename(files[i + j])[:-4])
            out.write(',' + os.path.basename(files[nindex[TOP, i + j]])[:-4])
            out.write(',' + os.path.basename(files[nindex[BOTTOM, i + j]])[:-4])
            out.write(',' + os.path.basename(files[nindex[LEFT, i + j]])[:-4])
            out.write(',' + os.path.basename(files[nindex[RIGHT, i + j]])[:-4])
            out.write(',' + str(nvalue[TOP, i + j]))
            out.write(',' + str(nvalue[BOTTOM, i + j]))
            out.write(',' + str(nvalue[LEFT, i + j]))
            out.write(',' + str(nvalue[RIGHT, i + j]))
            out.write('\n')

            if CREATE_PANORAMAS_AND_OTHER_IMAGES:
                initial_image = cv2.imread(files[i + j])
                closest_bottom = cv2.imread(files[nindex[BOTTOM, i + j]])
                closest_top = cv2.imread(files[nindex[TOP, i + j]])
                closest_left = cv2.imread(files[nindex[LEFT, i + j]])
                closest_right = cv2.imread(files[nindex[RIGHT, i + j]])
                dimg = np.zeros((256 * 3, 256 * 3, 3), dtype=np.uint8)
                dimg[256:2 * 256, 256:2 * 256, :] = initial_image
                dimg[0:256, 256:2 * 256, :] = closest_top
                dimg[2 * 256:, 256:2 * 256, :] = closest_bottom
                dimg[256:2 * 256, :256, :] = closest_left
                dimg[256:2 * 256, 2 * 256:, :] = closest_right
                cv2.imwrite(OUTPUT_PATH + "csim/" + os.path.basename(files[i + j])[:-4] + '.jpg', dimg)

    out.close()


def jpg_from_tiff(tiff):
    if 'train' in tiff:
        return INPUT_PATH + "train-jpg/" + os.path.basename(tiff)[:-4] + '.jpg'
    return INPUT_PATH + "test-jpg/" + os.path.basename(tiff)[:-4] + '.jpg'


def find_neightbours_part_tiff(csv_file, files, result, start, end):
    print('Start writing: {}'.format(csv_file))
    out = open(csv_file, "w")
    out.write('id,ntop,nbottom,nleft,nright,val_top,val_bottom,val_left,val_right\n')
    nindex = np.zeros((4, len(files)), dtype=np.int64)
    nvalue = np.zeros((4, len(files)), dtype=np.float64)
    part = 1
    for i in range(start, end, part):
        # if 'file_' not in files[i]:
        #     continue
        cs = fill_neighbours_v2(result, i)
        for j in range(4):
            nindex[j, i * part:(i + 1) * part] = np.argmax(cs[j], axis=1)
            nvalue[j, i * part:(i + 1) * part] = np.max(cs[j], axis=1)
        for j in range(part):
            if 0:
                print(nvalue[:, i + j],
                  os.path.basename(files[i + j]),
                  os.path.basename(files[nindex[BOTTOM, i + j]]),
                  os.path.basename(files[nindex[TOP, i + j]]),
                  os.path.basename(files[nindex[LEFT, i + j]]),
                  os.path.basename(files[nindex[RIGHT, i + j]])
                  )
            out.write(os.path.basename(files[i + j])[:-4])
            out.write(',' + os.path.basename(files[nindex[TOP, i + j]])[:-4])
            out.write(',' + os.path.basename(files[nindex[BOTTOM, i + j]])[:-4])
            out.write(',' + os.path.basename(files[nindex[LEFT, i + j]])[:-4])
            out.write(',' + os.path.basename(files[nindex[RIGHT, i + j]])[:-4])
            out.write(',' + str(nvalue[TOP, i + j]))
            out.write(',' + str(nvalue[BOTTOM, i + j]))
            out.write(',' + str(nvalue[LEFT, i + j]))
            out.write(',' + str(nvalue[RIGHT, i + j]))
            out.write('\n')

            if CREATE_PANORAMAS_AND_OTHER_IMAGES:
                initial_image = cv2.imread(jpg_from_tiff(files[i + j]))
                closest_bottom = cv2.imread(jpg_from_tiff(files[nindex[BOTTOM, i + j]]))
                closest_top = cv2.imread(jpg_from_tiff(files[nindex[TOP, i + j]]))
                closest_left = cv2.imread(jpg_from_tiff(files[nindex[LEFT, i + j]]))
                closest_right = cv2.imread(jpg_from_tiff(files[nindex[RIGHT, i + j]]))
                dimg = np.zeros((256 * 3, 256 * 3, 3), dtype=np.uint8)
                dimg[256:2 * 256, 256:2 * 256, :] = initial_image
                dimg[0:256, 256:2 * 256, :] = closest_top
                dimg[2 * 256:, 256:2 * 256, :] = closest_bottom
                dimg[256:2 * 256, :256, :] = closest_left
                dimg[256:2 * 256, 2 * 256:, :] = closest_right
                cv2.imwrite(OUTPUT_PATH + "csim_tiff/" + os.path.basename(files[i + j])[:-4] + '.jpg', dimg)

    out.close()


def find_neightbours():
    threads = max(1, cpu_count()-1)
    # threads = THREADS_NUM
    print('Number of threads:', threads)
    files = glob.glob(INPUT_PATH + '/train-jpg/' + '*.jpg') + glob.glob(INPUT_PATH + '/test-jpg/' + '*.jpg')
    result = get_borders(files)
    result = result.astype(np.int64)

    p = dict()
    for i in range(threads):
        csv_file = OUTPUT_PATH + "neighbours_{}.csv".format(i)
        start = i*len(files) // threads
        end = (i + 1) * len(files) // threads
        if i == threads-1:
            end = len(files)
        p[i] = Process(target=find_neightbours_part, args=(csv_file, files, result, start, end))
        p[i].start()
    for i in range(threads):
        p[i].join()


def concat_nfiles():
    files = glob.glob(OUTPUT_PATH + "neighbours_*.csv")
    lst = []
    for f in files:
        s = pd.read_csv(f)
        lst.append(s)
    s = pd.concat(lst, axis=0)
    s.to_csv(OUTPUT_PATH + "neighbours.csv", index=False)


def concat_nfiles_tiff():
    files = glob.glob(OUTPUT_PATH + "neighbours_tiff_*.csv")
    lst = []
    for f in files:
        s = pd.read_csv(f)
        lst.append(s)
    s = pd.concat(lst, axis=0)
    s.to_csv(OUTPUT_PATH + "neighbours_tiff.csv", index=False)


def find_neightbours_tiff():
    threads = max(1, cpu_count() - 1)
    # threads = THREADS_NUM

    files = glob.glob(INPUT_PATH + '/train-tif-v2/' + '*.tif') + glob.glob(INPUT_PATH + '/test-tif-v2/' + '*.tif')
    result = get_borders_tiff(files)
    result = result.astype(np.int64)

    p = dict()
    for i in range(threads):
        csv_file = OUTPUT_PATH + "neighbours_tiff_{}.csv".format(i)
        start = i*len(files) // threads
        end = (i + 1) * len(files) // threads
        if i == threads-1:
            end = len(files)
        p[i] = Process(target=find_neightbours_part_tiff, args=(csv_file, files, result, start, end))
        p[i].start()
    for i in range(threads):
        p[i].join()


def get_fake_count(t):
    l = 0
    for i in ['ntop', 'nbottom', 'nleft', 'nright']:
        l += len(t[t[i] == 'fake'])
    return l


def create_groups_from_tiff_and_jpg_at_once():
    n_jpg = pd.read_csv(OUTPUT_PATH + 'neighbours.csv')
    n_tiff = pd.read_csv(OUTPUT_PATH + 'neighbours_tiff.csv')

    n_tiff.rename(columns={'ntop': 'ntop_tiff', 'nbottom': 'nbottom_tiff', 'nleft': 'nleft_tiff', 'nright': 'nright_tiff'}, inplace=True)
    n = pd.DataFrame(pd.concat((n_jpg, n_tiff[['ntop_tiff', 'nbottom_tiff', 'nleft_tiff', 'nright_tiff']]), axis=1))
    # n = pd.concat((n_jpg, n_tiff[['ntop_tiff', 'nbottom_tiff', 'nleft_tiff', 'nright_tiff']]), axis=1)

    n['ntop_diff'] = (n['ntop'] != n['ntop_tiff'])
    n['nbottom_diff'] = (n['nbottom'] != n['nbottom_tiff'])
    n['nleft_diff'] = (n['nleft'] != n['nleft_tiff'])
    n['nright_diff'] = (n['nright'] != n['nright_tiff'])

    n.loc[n['ntop_diff'] == True, 'ntop'] = 'fake'
    n.loc[n['nbottom_diff'] == True, 'nbottom'] = 'fake'
    n.loc[n['nleft_diff'] == True, 'nleft'] = 'fake'
    n.loc[n['nright_diff'] == True, 'nright'] = 'fake'

    fc = get_fake_count(n)
    print('Fakes in {} positions out of {} ({}%)'.format(get_fake_count(n), 4*len(n), round(100*fc/(4*len(n)), 2)))

    # Remove cases where file is more than one time is top neighbour for some other files
    vc = n['ntop'].value_counts().to_frame()
    vc = vc[vc['ntop'] > 1]
    remove_list = list(vc.index.values)
    remove_list.remove('fake')
    print('Remove {} top additional fakes'.format(len(remove_list)))
    n.loc[n['ntop'].isin(remove_list), 'ntop'] = 'fake'

    # Multiple bottom removal
    vc = n['nbottom'].value_counts().to_frame()
    vc = vc[vc['nbottom'] > 1]
    remove_list = list(vc.index.values)
    remove_list.remove('fake')
    print('Remove {} bottom additional fakes'.format(len(remove_list)))
    n.loc[n['nbottom'].isin(remove_list), 'nbottom'] = 'fake'

    # Multiple left removal
    vc = n['nleft'].value_counts().to_frame()
    vc = vc[vc['nleft'] > 1]
    remove_list = list(vc.index.values)
    remove_list.remove('fake')
    print('Remove {} left additional fakes'.format(len(remove_list)))
    n.loc[n['nleft'].isin(remove_list), 'nleft'] = 'fake'

    # Multiple right removal
    vc = n['nright'].value_counts().to_frame()
    vc = vc[vc['nright'] > 1]
    remove_list = list(vc.index.values)
    remove_list.remove('fake')
    print('Remove {} right additional fakes'.format(len(remove_list)))
    n.loc[n['nright'].isin(remove_list), 'nright'] = 'fake'

    # Remove cases where image is neighbour for itself
    for i in ['ntop', 'nbottom', 'nleft', 'nright']:
        print('Cases where id is equal to neighbour: ', len(n[n[i] == n['id']]))
        n.loc[n[i] == n['id'], i] = 'fake'

    fc = get_fake_count(n)
    print('Fakes in {} positions out of {} ({}%)'.format(get_fake_count(n), 4*len(n), round(100*fc/(4*len(n)), 2)))

    n[['id', 'ntop', 'nbottom', 'nleft', 'nright']].to_csv(OUTPUT_PATH + 'neighbours_merged.csv', index=False)


def remove_bad_neighbours_and_create_groups(prefix):
    if 1:
        s = pd.read_csv(OUTPUT_PATH + prefix + ".csv")
        s['group_id'] = s.index
        s['proc'] = 0
    else:
        s = pd.read_csv(OUTPUT_PATH + prefix + "_cleaned_intermediate.csv")

    total = 0
    for index, row in s.iterrows():
        total += 1

        if row['proc'] == 1:
            continue

        # print(row['ntop'], row['nbottom'], row['nleft'], row['nright'])
        if row['ntop'] == 'fake' and row['nbottom'] == 'fake' and row['nleft'] == 'fake' and row['nright'] == 'fake':
            continue

        # rw = s[s['id'] == 'file_33']
        id = str(row['id'])
        group_id = row['group_id']
        rw = row
        top_m = rw['ntop']
        if top_m == 'fake':
            top_m = 'f'
        bottom_m = rw['nbottom']
        if bottom_m == 'fake':
            bottom_m = 'f'
        left_m = rw['nleft']
        if left_m == 'fake':
            left_m = 'f'
        right_m = rw['nright']
        if right_m == 'fake':
            right_m = 'f'


        rw_tl = s[(s['nright'] == top_m) & (s['nbottom'] == left_m)]
        rw_tr = s[(s['nleft'] == top_m) & (s['nbottom'] == right_m)]
        rw_bl = s[(s['nright'] == bottom_m) & (s['ntop'] == left_m)]
        rw_br = s[(s['nleft'] == bottom_m) & (s['ntop'] == right_m)]

        # rw_far_top = s[(s['nbottom'] == top_m)]
        # rw_far_bottom = s[(s['ntop'] == bottom_m)]
        # rw_far_left = s[(s['nright'] == left_m)]
        # rw_far_right = s[(s['nleft'] == right_m)]

        fake_list = []
        if rw_tl.empty and rw_tr.empty:
            fake_list.append('top')
            s.loc[index, 'ntop'] = 'fake'
        if rw_tl.empty and rw_bl.empty:
            fake_list.append('left')
            s.loc[index, 'nleft'] = 'fake'
        if rw_bl.empty and rw_br.empty:
            fake_list.append('bottom')
            s.loc[index, 'nbottom'] = 'fake'
        if rw_tr.empty and rw_br.empty:
            fake_list.append('right')
            s.loc[index, 'nright'] = 'fake'
        print('ID: {} Fake list: {}'.format(id, fake_list))

        # Put to group upper element
        if not rw_tl.empty or not rw_tr.empty:
            # print('Put top in group')
            condition = (s['id'] == top_m)
            elem_old_group = s[condition]['group_id'].values[0]
            if elem_old_group != group_id:
                s.loc[condition, 'group_id'] = group_id
                s.loc[s['group_id'] == elem_old_group, 'group_id'] = group_id

        # Put to group left element
        if not rw_tl.empty or not rw_bl.empty:
            # print('Put left in group')
            condition = (s['id'] == left_m)
            elem_old_group = s[condition]['group_id'].values[0]
            if elem_old_group != group_id:
                s.loc[condition, 'group_id'] = group_id
                s.loc[s['group_id'] == elem_old_group, 'group_id'] = group_id

        # Put to group bottom element
        if not rw_bl.empty or not rw_br.empty:
            # print('Put bottom in group')
            condition = (s['id'] == bottom_m)
            elem_old_group = s[condition]['group_id'].values[0]
            if elem_old_group != group_id:
                s.loc[condition, 'group_id'] = group_id
                s.loc[s['group_id'] == elem_old_group, 'group_id'] = group_id

        # Put to group right element
        if not rw_tr.empty or not rw_br.empty:
            # print('Put right in group')
            condition = (s['id'] == right_m)
            elem_old_group = s[condition]['group_id'].values[0]
            if elem_old_group != group_id:
                s.loc[condition, 'group_id'] = group_id
                s.loc[s['group_id'] == elem_old_group, 'group_id'] = group_id

        # Put to group diagonal element (up-left)
        if not rw_tl.empty:
            # print('Put top-left in group')
            condition = rw_tl.index
            elem_old_group = rw_tl['group_id'].values[0]
            if elem_old_group != group_id:
                s.loc[condition, 'group_id'] = group_id
                s.loc[s['group_id'] == elem_old_group, 'group_id'] = group_id

        # Put to group diagonal element (up-right)
        if not rw_tr.empty:
            # print('Put top-right in group')
            condition = rw_tr.index
            elem_old_group = rw_tr['group_id'].values[0]
            if elem_old_group != group_id:
                s.loc[condition, 'group_id'] = group_id
                s.loc[s['group_id'] == elem_old_group, 'group_id'] = group_id

        # Put to group diagonal element (bottom-right)
        if not rw_br.empty:
            # print('Put bottom-right in group')
            condition = rw_br.index
            elem_old_group = rw_br['group_id'].values[0]
            if elem_old_group != group_id:
                s.loc[condition, 'group_id'] = group_id
                s.loc[s['group_id'] == elem_old_group, 'group_id'] = group_id

        # Put to group diagonal element (bottom-left)
        if not rw_bl.empty:
            # print('Put bottom-left in group')
            condition = rw_bl.index
            elem_old_group = rw_bl['group_id'].values[0]
            if elem_old_group != group_id:
                s.loc[condition, 'group_id'] = group_id
                s.loc[s['group_id'] == elem_old_group, 'group_id'] = group_id

        # img = cv2.imread("../modified_data/csim/" + id + '.jpg')
        # show_resized_image(img)

        s.loc[index, 'proc'] = 1

        if total % 1000 == 0:
            print('Completed: {} out of {}'.format(total, len(s)))
            s.to_csv(OUTPUT_PATH + prefix + "_cleaned_intermediate.csv", index=False)

    s.to_csv(OUTPUT_PATH + prefix + "_cleaned.csv", index=False)
    return


def only_create_groups(prefix):
    if 1:
        s = pd.read_csv(OUTPUT_PATH + prefix + ".csv")
        s['group_id'] = s.index
        s['proc'] = 0
    else:
        s = pd.read_csv(OUTPUT_PATH + prefix + "_with_groups_intermediate.csv")

    total = 0
    for index, row in s.iterrows():
        total += 1

        if row['proc'] == 1:
            continue

        if row['ntop'] == 'fake' and row['nbottom'] == 'fake' and row['nleft'] == 'fake' and row['nright'] == 'fake':
            continue

        # id = str(row['id'])
        group_id = row['group_id']
        rw = row
        top_m = rw['ntop']
        bottom_m = rw['nbottom']
        left_m = rw['nleft']
        right_m = rw['nright']

        for n in [top_m, bottom_m, left_m, right_m]:
            if n == 'fake':
                continue
            condition = (s['id'] == n)
            elem_old_groups = list(s[condition]['group_id'].values)
            for elem_old_group in elem_old_groups:
                if elem_old_group != group_id:
                    s.loc[condition, 'group_id'] = group_id
                    s.loc[s['group_id'] == elem_old_group, 'group_id'] = group_id

        s.loc[index, 'proc'] = 1

        if total % 1000 == 0:
            print('Completed: {} out of {}'.format(total, len(s)))
            s.to_csv(OUTPUT_PATH + prefix + "_with_groups_intermediate.csv", index=False)

    s.to_csv(OUTPUT_PATH + prefix + "_with_groups.csv", index=False)
    return


def create_features_with_group_id_v1(fname):
    t = pd.read_csv(OUTPUT_PATH + fname + ".csv")
    t['image_name'] = t['id']
    train = pd.read_csv(INPUT_PATH + "train_v2.csv")
    subm = pd.read_csv(INPUT_PATH + "sample_submission_v2.csv")

    train = pd.merge(train[['image_name']], t[['image_name', 'group_id']], on='image_name', left_index=True)
    train.reset_index(drop=True, inplace=True)
    subm = pd.merge(subm[['image_name']], t[['image_name', 'group_id']], on='image_name', left_index=True)
    subm.reset_index(drop=True, inplace=True)

    u = t['group_id'].unique()
    res = dict()
    for g in u:
        part = t[t['group_id'] == g]
        if len(part) <= 1:
            continue
        res[g] = len(part)

    rename = dict()
    total = 0
    for r in sorted(res.keys()):
        rename[r] = total
        total += 1

    train['count'] = -1
    for index, row in train.iterrows():
        gid = row['group_id']
        if gid in rename:
            train.loc[index, 'group_id'] = rename[gid]
            train.loc[index, 'count'] = res[gid]
        else:
            train.loc[index, 'group_id'] = -1
    train.to_csv(FEATURES_PATH + fname + '_train.csv', index=False)

    subm['count'] = -1
    for index, row in subm.iterrows():
        gid = row['group_id']
        if gid in rename:
            subm.loc[index, 'group_id'] = rename[gid]
            subm.loc[index, 'count'] = res[gid]
        else:
            subm.loc[index, 'group_id'] = -1
    subm.to_csv(FEATURES_PATH + fname + '_test.csv', index=False)


# Remove group IDs which only exists in train or in test data
def clean_group_ids(prefix):
    train = pd.read_csv(FEATURES_PATH + prefix + '_train.csv')
    test = pd.read_csv(FEATURES_PATH + prefix + '_test.csv')

    uni_train = set(train['group_id'].unique())
    uni_test = set(test['group_id'].unique())
    common = uni_train & uni_test
    print('Unique train:', len(uni_train))
    print('Unique test:', len(uni_test))
    print('Common groups:', len(common))
    only_train = list(uni_train - common)
    print('Only train groups:', len(only_train))
    only_test = list(uni_test - common)
    print('Only test groups:', len(only_test))
    train.loc[train['group_id'].isin(only_train), 'group_id'] = -1
    test.loc[test['group_id'].isin(only_test), 'group_id'] = -1
    train.to_csv(FEATURES_PATH + prefix + '_fix_train.csv', index=False)
    test.to_csv(FEATURES_PATH + prefix + '_fix_test.csv', index=False)


def create_panorama_for_group(t, group_id, panorama_path):
    part = t[t['group_id'] == group_id]
    if len(part) <= 1:
        return
    print('Go for group: {} Length: {}'.format(group_id, len(part)))
    placed = dict()
    for index, row in part.iterrows():
        id = row['id']
        placed[id] = [0, 0]
        break

    while 1:
        change = 0
        for index, row in part.iterrows():
            id = row['id']
            top_m = row['ntop']
            bottom_m = row['nbottom']
            left_m = row['nleft']
            right_m = row['nright']
            if id not in placed:
                if top_m in placed:
                    placed[id] = [placed[top_m][0] + 1, placed[top_m][1]]
                    change += 1
                if bottom_m in placed:
                    placed[id] = [placed[bottom_m][0] - 1, placed[bottom_m][1]]
                    change += 1
                if left_m in placed:
                    placed[id] = [placed[left_m][0], placed[left_m][1] + 1]
                    change += 1
                if right_m in placed:
                    placed[id] = [placed[right_m][0], placed[right_m][1] - 1]
                    change += 1
        if change == 0:
            break
    if len(placed) != len(part):
        print('Some problems with placemement! {} != {}'.format(len(placed), len(part)))
        # exit()


    # Find min max and move position
    min_pos_0 = 10000000
    max_pos_0 = -1000000
    min_pos_1 = 10000000
    max_pos_1 = -1000000
    for el in placed:
        if placed[el][0] < min_pos_0:
            min_pos_0 = placed[el][0]
        if placed[el][0] > max_pos_0:
            max_pos_0 = placed[el][0]
        if placed[el][1] < min_pos_1:
            min_pos_1 = placed[el][1]
        if placed[el][1] > max_pos_1:
            max_pos_1 = placed[el][1]

    print('Positions: {} - {} {} - {}'.format(min_pos_0, max_pos_0, min_pos_1, max_pos_1))
    for el in placed:
        placed[el][0] -= min_pos_0
        placed[el][1] -= min_pos_1

    max_0 = max_pos_0 - min_pos_0 + 1
    max_1 = max_pos_1 - min_pos_1 + 1

    panorama = np.zeros((max_0*256, max_1*256, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for el in placed:
        if 'train_' in el:
            img = cv2.imread(INPUT_PATH + "train-jpg/" + el + ".jpg")
        else:
            img = cv2.imread(INPUT_PATH + "test-jpg/" + el + ".jpg")
        cv2.putText(img, el, (30, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        p0 = placed[el][0]*256
        p1 = placed[el][1]*256
        panorama[p0:p0+256, p1:p1+256, :] = img.copy()

    # show_resized_image(panorama)
    cv2.imwrite(panorama_path, panorama)
    print('Shape of panorama: {} x {}. Number of images: {}'.format(max_0, max_1, len(placed)))


def create_panoramas_merged():
    t = pd.read_csv(OUTPUT_PATH + "neighbours_merged_cleaned.csv")

    uni = t['group_id'].unique()
    store_dir = OUTPUT_PATH + "panoramas_merged/"
    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)
    for group_id in uni:
        panorama_path = store_dir + str(group_id) + '.jpg'
        if os.path.isfile(panorama_path):
            print('Skip {}'.format(group_id))
            continue
        create_panorama_for_group(t, group_id, panorama_path)


def get_8_neighbours(prefix):
    s = pd.read_csv('../modified_data/' + prefix + '_cleaned.csv')
    s['ntl'] = ''
    s['ntr'] = ''
    s['nbl'] = ''
    s['nbr'] = ''
    for index, row in s.iterrows():
        top_m = row['ntop']
        if top_m == 'fake':
            top_m = 'f'
        bottom_m = row['nbottom']
        if bottom_m == 'fake':
            bottom_m = 'f'
        left_m = row['nleft']
        if left_m == 'fake':
            left_m = 'f'
        right_m = row['nright']
        if right_m == 'fake':
            right_m = 'f'

        # print('Go {}'.format(row['id']))
        rw_tl = s[(s['nright'] == top_m) & (s['nbottom'] == left_m)]
        rw_tr = s[(s['nleft'] == top_m) & (s['nbottom'] == right_m)]
        rw_bl = s[(s['nright'] == bottom_m) & (s['ntop'] == left_m)]
        rw_br = s[(s['nleft'] == bottom_m) & (s['ntop'] == right_m)]
        if not rw_tl.empty:
            s.loc[index, 'ntl'] = rw_tl['id'].values[0]
            if row['group_id'] != rw_tl['group_id'].values[0]:
                print('Strange!!!', row['group_id'], rw_tl)
        if not rw_tr.empty:
            s.loc[index, 'ntr'] = rw_tr['id'].values[0]
            if row['group_id'] != rw_tr['group_id'].values[0]:
                print('Strange!!!', row['group_id'], rw_tr)
        if not rw_bl.empty:
            s.loc[index, 'nbl'] = rw_bl['id'].values[0]
            if row['group_id'] != rw_bl['group_id'].values[0]:
                print('Strange!!!', row['group_id'], rw_bl)
        if not rw_br.empty:
            s.loc[index, 'nbr'] = rw_br['id'].values[0]
            if row['group_id'] != rw_br['group_id'].values[0]:
                print('Strange!!!', row['group_id'], rw_br)
        if (index + 1) % 1000 == 0:
            print('Completed {} from {}'.format(index + 1, len(s)))

    s.to_csv('../modified_data/' + prefix + '_cleaned_8.csv', index=False)


def get_8_neighbours_v2(prefix):
    s = pd.read_csv('../modified_data/' + prefix + '.csv')
    s['ntl'] = 'fake'
    s['ntr'] = 'fake'
    s['nbl'] = 'fake'
    s['nbr'] = 'fake'
    for index, row in s.iterrows():
        top_m = row['ntop']
        if top_m == 'fake':
            top_m = 'f'
        bottom_m = row['nbottom']
        if bottom_m == 'fake':
            bottom_m = 'f'
        left_m = row['nleft']
        if left_m == 'fake':
            left_m = 'f'
        right_m = row['nright']
        if right_m == 'fake':
            right_m = 'f'

        # print('Go {}'.format(row['id']))
        rw_tl = s[(s['nright'] == top_m) | (s['nbottom'] == left_m)]
        rw_tr = s[(s['nleft'] == top_m) | (s['nbottom'] == right_m)]
        rw_bl = s[(s['nright'] == bottom_m) | (s['ntop'] == left_m)]
        rw_br = s[(s['nleft'] == bottom_m) | (s['ntop'] == right_m)]
        if len(rw_tl) > 1:
            print('More than one neighbour TL!', row['id'])
        if len(rw_tr) > 1:
            print('More than one neighbour TR!', row['id'])
        if len(rw_bl) > 1:
            print('More than one neighbour BL!', row['id'])
        if len(rw_br) > 1:
            print('More than one neighbour BR!', row['id'])
        if len(rw_tl) == 1:
            s.loc[index, 'ntl'] = rw_tl['id'].values[0]
        if len(rw_tr):
            s.loc[index, 'ntr'] = rw_tr['id'].values[0]
        if len(rw_bl):
            s.loc[index, 'nbl'] = rw_bl['id'].values[0]
        if len(rw_br):
            s.loc[index, 'nbr'] = rw_br['id'].values[0]
        if (index + 1) % 1000 == 0:
            print('Completed {} from {}'.format(index + 1, len(s)))

    s.to_csv('../modified_data/' + prefix + '_8.csv', index=False)


if __name__ == '__main__':
    start_time = time.time()

    find_neightbours()
    concat_nfiles()
    find_neightbours_tiff()
    concat_nfiles_tiff()
    create_groups_from_tiff_and_jpg_at_once()

    # Pessimistic variant
    remove_bad_neighbours_and_create_groups('neighbours_merged')
    # Optimistic variant
    only_create_groups('neighbours_merged')

    create_features_with_group_id_v1('neighbours_merged_cleaned')
    clean_group_ids('neighbours_merged_cleaned')

    create_features_with_group_id_v1('neighbours_merged_with_groups')
    clean_group_ids('neighbours_merged_with_groups')

    get_8_neighbours_v2('neighbours_merged_cleaned')
    get_8_neighbours_v2('neighbours_merged')

    if CREATE_PANORAMAS_AND_OTHER_IMAGES:
        create_panoramas_merged()

    print("Elapsed time overall: %s seconds" % (time.time() - start_time))
