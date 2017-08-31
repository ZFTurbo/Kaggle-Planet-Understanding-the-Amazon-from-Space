# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *

random.seed(2017)
np.random.seed(2017)


def get_f2_score(thr_arr, p, lbl, indexes):
    preds = p.copy()
    for i in range(len(indexes)):
        preds[:, i][preds[:, i] > thr_arr[i]] = 1
        preds[:, i][preds[:, i] <= thr_arr[i]] = 0
    score = f2_score(lbl, preds)
    # print(score)
    # print(thr_arr)
    return -score


def test_1():
    p = np.arange(150.).reshape(5, 10, 3)
    print(p[:, :, 0])
    cond = p[:, :, 0] > 80
    print(cond)
    cond = np.sum(cond, axis=0)
    set_1 = cond*2 > p.shape[0]
    set_0 = cond*2 < p.shape[0]
    print(set_1)
    print(set_0)
    exit()


def get_f2_score_full_arr(thr_arr, p, lbl, indexes):
    preds = np.zeros(p.shape[1:])
    for i in range(len(indexes)):
        cond = p[:, :, i] > thr_arr[i]
        cond = 2 * np.sum(cond, axis=0)
        preds[:, i][cond >= p.shape[0]] = 1
        preds[:, i][cond < p.shape[0]] = 0
    score = f2_score(lbl, preds)
    # print(score)
    # print(thr_arr)
    return -score


def f2beta_loss(Y_true, Y_pred):
    from keras import backend as K
    eps = 0.000001
    false_positive = K.sum(Y_pred * (1 - Y_true), axis=-1)
    false_negative = K.sum((1 - Y_pred) * Y_true, axis=-1)
    true_positive = K.sum(Y_true * Y_pred, axis=-1)
    p = (true_positive + eps) / (true_positive + false_positive + eps)
    r = (true_positive + eps) / (true_positive + false_negative + eps)
    out = (5*p*r + eps) / (4*p + r + eps)
    return -K.mean(out)


def get_optimal_score_normal(indexes, lbl, validation_arr, koeff=100):
    start_time = time.time()
    # Step 1 (Low precision)
    best_score1 = -1
    best_index = -1
    no_improve = 0
    for i in range(koeff):
        thr = i / koeff
        score = abs(get_f2_score(np.array([thr] * len(indexes)), validation_arr, lbl, indexes))
        # print(thr, score)
        if score > best_score1:
            best_score1 = score
            best_index = i
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= koeff // 5:
            break

    # Step 2 (High precision)
    best_score = -1
    best_thr = -1
    no_improve = 0
    for i in range(max(koeff * (best_index - 10), 0), koeff*(best_index + 10)):
        thr = i / (koeff*koeff)
        score = abs(get_f2_score(np.array([thr] * len(indexes)), validation_arr, lbl, indexes))
        # print(thr, score)
        if score > best_score:
            best_score = score
            best_thr = thr
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= koeff*2:
            break

    # Try to find optimum independently for each class
    searcher = np.array([best_thr] * len(indexes))
    best_score = -1
    for j in range(len(indexes)):
        best_thr1 = searcher[j]

        # Step 1 (Low precision)
        best_score1 = -1
        best_index = -1
        no_improve = 0
        for i in range(koeff):
            thr = i / koeff
            searcher[j] = thr
            score = abs(get_f2_score(searcher, validation_arr, lbl, indexes))
            # print(thr, score)
            if score > best_score1:
                best_score1 = score
                best_index = i
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= koeff // 5:
                break

        # Step 2 (High precision)
        best_score1 = -1
        no_improve = 0
        for i in range(max(koeff * (best_index - 10), 0), koeff * (best_index + 10)):
            thr = i / (koeff*koeff)
            searcher[j] = thr
            score = abs(get_f2_score(searcher, validation_arr, lbl, indexes))
            # print(thr, score)
            if score > best_score:
                best_score = score
                best_thr1 = thr
            if score > best_score1:
                best_score1 = score
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= koeff:
                break

        searcher[j] = best_thr1
        print('Class: {} Best score {} for THR: {}'.format(j, best_score, best_thr1))
    print("Time search thr: %s sec" % (round(time.time() - start_time, 0)))
    return best_score, searcher


def get_optimal_score_fast(indexes, lbl, validation_arr, koeff=100):
    start_time = time.time()
    # Step 1 (Low precision)
    best_score1 = -1
    best_thr = -1
    no_improve = 0
    for i in range(koeff):
        thr = i / koeff
        score = abs(get_f2_score(np.array([thr] * len(indexes)), validation_arr, lbl, indexes))
        if score > best_score1:
            best_score1 = score
            best_thr = thr
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= koeff // 10:
            break

    # Try to find optimum independently for each class
    searcher = np.array([best_thr] * len(indexes))
    best_score = -1
    for j in range(len(indexes)):
        best_thr1 = searcher[j]
        best_score1 = -1
        no_improve = 0
        for i in range(koeff):
            thr = i / koeff
            searcher[j] = thr
            score = abs(get_f2_score(searcher, validation_arr, lbl, indexes))
            if score > best_score:
                best_score = score
                best_thr1 = thr

            if score > best_score1:
                best_score1 = score
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= koeff // 10:
                break
        searcher[j] = best_thr1
        print('Class: {} Best score {} for THR: {}'.format(j, best_score, best_thr1))
    print("Time search thr: %s seconds" % (round(time.time() - start_time, 0)))
    return best_score, searcher


def get_optimal_score_slow(indexes, lbl, validation_arr, koeff=100):
    start_time = time.time()
    # Step 1 (Low precision)
    best_score = -1
    best_thr = -1
    no_improve = 0
    for i in range(koeff):
        thr = i / koeff
        score = abs(get_f2_score(np.array([thr] * len(indexes)), validation_arr, lbl, indexes))
        if score > best_score:
            best_score = score
            best_thr = thr
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= koeff:
            break

    # Try to find optimum independently for each class
    searcher = np.array([best_thr] * len(indexes))
    for j in range(len(indexes)):
        best_thr1 = searcher[j]
        best_score1 = -1
        no_improve = 0
        for i in range(koeff):
            thr = i / koeff
            searcher[j] = thr
            score = abs(get_f2_score(searcher, validation_arr, lbl, indexes))
            if score > best_score:
                best_score = score
                best_thr1 = thr

            if score > best_score1:
                best_score1 = score
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= koeff:
                break
        searcher[j] = best_thr1
        print('Class: {} Best score {} for THR: {}'.format(j, best_score, best_thr1))

    # Try to find optimum independently for each class
    koeff *= 100
    for j in range(len(indexes)):
        best_thr1 = searcher[j]
        best_score1 = -1
        no_improve = 0
        for i in range(koeff):
            thr = i / koeff
            searcher[j] = thr
            score = abs(get_f2_score(searcher, validation_arr, lbl, indexes))
            if score > best_score:
                best_score = score
                best_thr1 = thr

            if score > best_score1:
                best_score1 = score
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= koeff:
                break
        searcher[j] = best_thr1
        print('Class: {} Best score {} for THR: {}'.format(j, best_score, best_thr1))
    print("Time search thr: %s seconds" % (round(time.time() - start_time, 0)))
    return best_score, searcher


def get_optimal_score_very_fast(indexes, lbl, validation_arr, splits=7, eps=0.01, iterations=2, log=True):
    start_time = time.time()

    best_score = -1
    best_thr = -1
    start = 0.0
    stop = 1.0
    while 1:
        scores = np.zeros(splits)
        thr_list = np.zeros(splits)
        for i in range(splits):
            thr = start + i*(stop - start)/splits
            thr_list[i] = thr
            scores[i] = abs(get_f2_score(np.array([thr] * len(indexes)), validation_arr, lbl, indexes))
        index = int(np.argmax(scores))
        score = scores[index]
        if score > best_score:
            best_score = score
            best_thr = thr_list[index]
        if index > 0:
            start = thr_list[index - 1]
        if index < splits-1:
            stop = thr_list[index + 1]
        if stop - start < eps:
            break
    if log:
        print('Best score {} for single THR: {}'.format(best_score, best_thr))

    # Try to find optimum independently for each class
    searcher = np.array([best_thr] * len(indexes))
    orig_split = splits
    for z in range(iterations):
        if z > 0:
            splits = random.randint(orig_split-3, orig_split+3)
            if splits == orig_split:
                splits = random.randint(orig_split - 3, orig_split + 3)
            if log:
                print('Iteration: {} New split: {}'.format(z, splits))
        for j in range(len(indexes)):
            best_thr1 = searcher[j]
            start = 0.0
            stop = 1.0
            while 1:
                scores = np.zeros(splits)
                thr_list = np.zeros(splits)
                for i in range(splits):
                    thr = start + i * (stop - start) / splits
                    thr_list[i] = thr
                    searcher[j] = thr
                    scores[i] = abs(get_f2_score(searcher, validation_arr, lbl, indexes))
                index = int(np.argmax(scores))
                score = scores[index]
                if score > best_score:
                    best_score = score
                    best_thr1 = thr_list[index]
                if index > 0:
                    start = thr_list[index - 1]
                if index < splits - 1:
                    stop = thr_list[index + 1]
                if stop - start < eps:
                    break
            searcher[j] = best_thr1
            if log:
                print('Class: {} Best score {} for THR: {}'.format(j, best_score, best_thr1))
    if log:
        print("Time search thr: %s seconds" % (round(time.time() - start_time, 0)))
    return best_score, searcher


def get_optimal_score_very_fast_for_full_array(indexes, lbl, validation_arr, splits=7, eps=0.01, iterations=2):
    start_time = time.time()

    best_score = -1
    best_thr = -1
    start = 0.0
    stop = 1.0
    while 1:
        scores = np.zeros(splits)
        thr_list = np.zeros(splits)
        for i in range(splits):
            thr = start + i*(stop - start)/splits
            thr_list[i] = thr
            scores[i] = abs(get_f2_score_full_arr(np.array([thr] * len(indexes)), validation_arr, lbl, indexes))
        index = int(np.argmax(scores))
        score = scores[index]
        if score > best_score:
            best_score = score
            best_thr = thr_list[index]
        if index > 0:
            start = thr_list[index - 1]
        if index < splits-1:
            stop = thr_list[index + 1]
        if stop - start < eps:
            break
    print('Best score {} for single THR: {}'.format(best_score, best_thr))

    # Try to find optimum independently for each class
    searcher = np.array([best_thr] * len(indexes))
    orig_split = splits
    for z in range(iterations):
        if z > 0:
            splits = random.randint(orig_split-3, orig_split+3)
            if splits == orig_split:
                splits = random.randint(orig_split - 3, orig_split + 3)
            print('Iteration: {} New split: {}'.format(z, splits))
        for j in range(len(indexes)):
            best_thr1 = searcher[j]
            start = 0.0
            stop = 1.0
            while 1:
                scores = np.zeros(splits)
                thr_list = np.zeros(splits)
                for i in range(splits):
                    thr = start + i * (stop - start) / splits
                    thr_list[i] = thr
                    searcher[j] = thr
                    scores[i] = abs(get_f2_score_full_arr(searcher, validation_arr, lbl, indexes))
                index = int(np.argmax(scores))
                score = scores[index]
                if score > best_score:
                    best_score = score
                    best_thr1 = thr_list[index]
                if index > 0:
                    start = thr_list[index - 1]
                if index < splits - 1:
                    stop = thr_list[index + 1]
                if stop - start < eps:
                    break
            searcher[j] = best_thr1
            print('Class: {} Best score {} for THR: {}'.format(j, best_score, best_thr1))
    print("Time search thr: %s seconds" % (round(time.time() - start_time, 0)))
    return best_score, searcher
