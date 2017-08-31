# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *
from sklearn.metrics import log_loss, roc_auc_score

GPU_TO_USE = 0
USE_THEANO = 1

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


def get_validation_score_single_class(nfolds, class_id, cnn_type):
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
        files.append("../input/train-jpg/" + id + '.jpg')
    files = np.array(files)

    lbl = np.zeros((len(labels), len(indexes)))
    for j in range(len(labels)):
        l = labels[j]
        for i in range(len(indexes)):
            if indexes[i] in l:
                lbl[j][i] = 1

    print(lbl.shape)
    stat = []

    kf = KFold(n_splits=nfolds, shuffle=True, random_state=get_random_state(cnn_type))
    num_fold = 0
    result = np.zeros(len(labels))
    for train_ids, valid_ids in kf.split(range(len(files))):
        num_fold += 1
        start_time = time.time()
        cache_file = CACHE_PATH + '{}_valid_fold_{}_single_class_{}'.format(cnn_type, num_fold, class_id)
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(train_ids))
        print('Split valid: ', len(valid_ids))
        valid_files = files[valid_ids]
        valid_labels = lbl[valid_ids, class_id]
        if not (os.path.isfile(cache_file) and restore_from_cache):
            final_model_path = MODELS_PATH + '{}_fold_{}_single_class_{}.h5'.format(cnn_type, num_fold, class_id)
            print('Loading model {}...'.format(final_model_path))
            model = get_pretrained_model(cnn_type, 1, final_layer_activation='sigmoid')
            model.load_weights(final_model_path)
            preds = get_raw_predictions_for_images(model, cnn_type, valid_files, 8)
            save_in_file(preds, cache_file)
        else:
            preds = load_from_file(cache_file)

        if len(preds[np.isnan(preds)]) > 0:
            print('There are {} NANs in pred! Fix them with mean'.format(len(preds[np.isnan(preds)])))
            mean = preds[~np.isnan(preds)].mean()
            preds[np.isnan(preds)] = mean
        result[valid_ids] = preds
        print(preds.shape)
        print(valid_labels.shape)

        best_score = log_loss(valid_labels, preds)
        print('Best log loss: {}'.format(best_score))
        best_score = roc_auc_score(valid_labels, preds)
        print('Best AUC: {}'.format(best_score))
        print('Fold time: {} seconds'.format(time.time() - start_time))

    best_score = log_loss(lbl[:, class_id], result)
    print('Best log loss score: {}'.format(best_score))
    best_score = roc_auc_score(lbl[:, class_id], result)
    print('Best AUC score: {}'.format(best_score))

    # Save validation file
    out = open(FEATURES_PATH + "valid_{}_single_class_{}_score_{}.csv".format(cnn_type, class_id, best_score), "w")
    # out = open(FEATURES_PATH + "valid_{}_single_class_{}.csv".format(cnn_type, class_id), "w")
    out.write("image_name")
    out.write("," + indexes[class_id])
    out.write("\n")
    ids = tbl['image_name'].values
    for i in range(len(result)):
        out.write(ids[i])
        out.write("," + str(result[i]))
        out.write("\n")
    out.close()

    return best_score


def process_test_single_class(nfolds, cnn_type, class_id, score):
    global FOLD_TO_CALC
    from keras.models import load_model
    from keras import backend as K

    if K.backend() == 'tensorflow':
        print('Update dim ordering to "tf"')
        K.set_image_dim_ordering('tf')

    restore_from_cache = 1

    tbl = pd.read_csv(INPUT_PATH + "sample_submission_v2.csv")
    indexes = get_indexes()
    ids = tbl['image_name'].values

    files = []
    for id in ids:
        files.append("../input/test-jpg/" + id + '.jpg')
    files = np.array(files)
    # files = files[:100]

    preds = []
    for num_fold in range(1, nfolds+1):
        start_time = time.time()

        if 'FOLD_TO_CALC' in globals():
            if num_fold not in FOLD_TO_CALC:
                continue

        cache_file = CACHE_PATH + '{}_test_fold_{}_single_class_{}'.format(cnn_type, num_fold, class_id)
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        if os.path.isfile(cache_file) and restore_from_cache:
            print('Restore from cache...')
            p = load_from_file(cache_file)
        else:
            final_model_path = MODELS_PATH + '{}_fold_{}_single_class_{}.h5'.format(cnn_type, num_fold, class_id)
            print('Loading model {}...'.format(final_model_path))
            model = get_pretrained_model(cnn_type, 1, final_layer_activation='sigmoid')
            model.load_weights(final_model_path)
            p = get_raw_predictions_for_images(model, cnn_type, files, 8)
            save_in_file(p, cache_file)
        preds.append(p)
    preds = np.array(preds)
    print(preds.shape)
    preds = np.mean(preds, axis=0)

    # Save raw file
    out = open(FEATURES_PATH + "test_{}_single_class_{}_score_{}.csv".format(cnn_type, class_id, score), "w")
    out.write("image_name")
    out.write("," + indexes[class_id])
    out.write("\n")
    ids = tbl['image_name'].values
    for i in range(len(preds)):
        out.write(ids[i])
        out.write("," + str(preds[i][0]))
        out.write("\n")
    out.close()


if __name__ == '__main__':
    num_folds = 5
    for class_id in range(1, 17):
        for cnn in ['DENSENET_121']:
            best_score = get_validation_score_single_class(num_folds, class_id, cnn)
            process_test_single_class(num_folds, cnn, class_id, best_score)


'''
Validation results:

Class 0 (24 augment):
Fold 1:
Best log loss: 0.22772516145361263
Best AUC: 0.9681611676805534
Fold 2:
Best log loss: 0.24578336946242255
Best AUC: 0.9651904325699159
Fold 3:
Best log loss: 0.24523281635632213
Best AUC: 0.9648608595328034
Fold 4:
Best log loss: 0.2655470102476548
Best AUC: 0.9635775854593888
Fold 5:
Best log loss: 0.24445596844166212
Best AUC: 0.9635897933788119
Best log loss score: 0.24574889719927384
Best AUC score: 0.9643620758364253

New validation (8 augment):

Class 4:
Best log loss: 0.19271503292580192
Best AUC: 0.9154468633958038
Best log loss: 0.20040894672624915
Best AUC: 0.954901232680224
Best log loss: 0.15068588557553242
Best AUC: 0.938265249604139
Best log loss: 0.2462618427549412
Best AUC: 0.9301456936283611
Best log loss: 0.23098790161785845
Best AUC: 0.9130177596192206
Best log loss score: 0.2042112602743943
Best AUC score: 0.9274156010934623

Class 1:
Best log loss: 0.03811951119234441
Best AUC: 0.9951909464526258
Best log loss: 0.048754306974963635
Best AUC: 0.9984422903143837
Best log loss: 0.06174141446384239
Best AUC: 0.9921879692151934
Best log loss: 0.030328301687537473
Best AUC: 0.9989694385216773
Best log loss: 0.04053232867061804
Best AUC: 0.9966355784605692
Fold time: 728.634740114212 seconds
Best log loss score: 0.04389525566509881
Best AUC score: 0.9967844843931196

Class 5:
Best log loss: 0.13754383867745576
Best AUC: 0.9898741534977363
Best log loss: 0.1407097081301429
Best AUC: 0.9884055385718593
Best log loss: 0.14993728358783198
Best AUC: 0.9891054665926355
Best log loss: 0.1479958416072093
Best AUC: 0.9901369790230676
Best log loss: 0.13113664577419135
Best AUC: 0.9887792855989967
Fold time: 803.063668012619 seconds
Best log loss score: 0.14146491865284122
Best AUC score: 0.9891344529494603

Class 6:
Best log loss: 0.1013346719765783
Best AUC: 0.9932364226284082
Best log loss: 0.11845386073228246
Best AUC: 0.9939338697510687
Best log loss: 0.10418842342835029
Best AUC: 0.9923096942759018
Best log loss: 0.0936938813620891
Best AUC: 0.9953129506560875
Best log loss: 0.0887781298560271
Best AUC: 0.9955662606053148
Fold time: 690.0189571380615 seconds
Best log loss score: 0.10129010250387299
Best AUC score: 0.9939780821432701

Class 2:
Best log loss: 0.3193028522943171
Best AUC: 0.9168173985473191
Best log loss: 0.29456970135771376
Best AUC: 0.9214497759336571
Best log loss: 0.2815473452201511
Best AUC: 0.9249727894921321
Best log loss: 0.29912306827751317
Best AUC: 0.9347493342197668
Best log loss: 0.29637484576337675
Best AUC: 0.938331896766066
Fold time: 617.7306578159332 seconds
Best log loss score: 0.2981836069756931
Best AUC score: 0.9272408748804607

Class 3:
Best log loss: 0.2915650288312323
Best AUC: 0.9327988659329712
Best log loss: 0.29132380419460807
Best AUC: 0.9346460885945589
Best log loss: 0.2261809924541458
Best AUC: 0.9743947653799936
Best log loss: 0.2695016681096027
Best AUC: 0.9528340647768844
Best log loss: 0.3026091752105957
Best AUC: 0.9159740381023533
Fold time: 607.7529680728912 seconds
Best log loss score: 0.27623548212075444
Best AUC score: 0.9445849004906965

Class 7:
Best log loss: 0.10770234185427274
Best AUC: 0.9885339753014611
Best log loss: 0.08665580671766984
Best AUC: 0.9938273514851486
Best log loss: 0.10820953582724398
Best AUC: 0.9902734810048262
Best log loss: 0.06758468670264607
Best AUC: 0.9873491347971499
Best log loss: 0.040493334328091525
Best AUC: 0.9953300728228552
Fold time: 738.5424618721008 seconds
Best log loss score: 0.08213016964691523
Best AUC score: 0.9898691151341044

Class 12:
Best log loss: 0.1762134442716055
Best AUC: 0.9853955942900918
Best log loss: 0.17752719105933168
Best AUC: 0.9843507889270774
Best log loss: 0.17138769259719164
Best AUC: 0.9830966247941769
Best log loss: 0.1590568591193237
Best AUC: 0.9874772203614748
Best log loss: 0.13316565771373035
Best AUC: 0.986358030102203
Fold time: 619.8796629905701 seconds
Best log loss score: 0.16347091746764283
Best AUC score: 0.9845900937304197

Class 10:
Best log loss: 0.21512476057973134
Best AUC: 0.9713505690415105
Best log loss: 0.20589928443105426
Best AUC: 0.973395825689021
Best log loss: 0.2008245884089307
Best AUC: 0.9709878360815531
Best log loss: 0.21708911002298573
Best AUC: 0.9738312645732162
Best log loss: 0.2076587643386939
Best AUC: 0.973662198972203
Fold time: 598.0848441123962 seconds
Best log loss score: 0.2093193424692287
Best AUC score: 0.9725594023611087

Class 11:
Best log loss: 0.09679505293445312
Best AUC: 0.9934267337278585
Best log loss: 0.10177069504633214
Best AUC: 0.9938068291132358
Best log loss: 0.09597090466566603
Best AUC: 0.9936317460069493
Best log loss: 0.1074384845116223
Best AUC: 0.9936855283653945
Best log loss: 0.13955825222181115
Best AUC: 0.9909820947620634
Fold time: 682.8519070148468 seconds
Best log loss score: 0.10830590562199588
Best AUC score: 0.9930041927993318

Class 14:
Best log loss: 0.18723967045451514
Best AUC: 0.9522946870213888
Best log loss: 0.2737099454922427
Best AUC: 0.9381282085033567
Best log loss: 0.26238505813365587
Best AUC: 0.9538068585643212
Best log loss: 0.23004463341357106
Best AUC: 0.9678736277445109
Best log loss: 0.22609733187008046
Best AUC: 0.9602489110143124
Fold time: 705.9609940052032 seconds
Best log loss score: 0.23589556996675196
Best AUC score: 0.953874770466746

Class 8:
Best log loss: 0.42778759389878684
Best AUC: 0.9231615100013267
Best log loss: 0.3170926169958293
Best AUC: 0.9254201946965472
Best log loss: 0.39044749693707115
Best AUC: 0.9225318030973451
Best log loss: 0.3933927935731148
Best AUC: 0.9361093947609957
Best log loss: 0.318176923364811
Best AUC: 0.9225128061653657
Fold time: 802.30384516716 seconds
Best log loss score: 0.3693807498092545
Best AUC score: 0.9239076007702497

Class 9:
Best log loss: 0.19298712914387076
Best AUC: 0.9671239105721865
Best log loss: 0.21576608148377555
Best AUC: 0.969628411966739
Best log loss: 0.17453910143247506
Best AUC: 0.9733059192056234
Best log loss: 0.18491012337760004
Best AUC: 0.9690910716472565
Best log loss: 0.19258769059213018
Best AUC: 0.9689034194174346
Fold time: 668.7590019702911 seconds
Best log loss score: 0.19215801456555895
Best AUC score: 0.9693925215613167

Class 13:
Best log loss: 0.18746864205839933
Best AUC: 0.9768615485494193
Best log loss: 0.20173310505902728
Best AUC: 0.9749429994128367
Best log loss: 0.2061426603520332
Best AUC: 0.9790132993037375
Best log loss: 0.19074217667342347
Best AUC: 0.9743706430198277
Best log loss: 0.2011499023623359
Best AUC: 0.9761030213325245
Fold time: 670.308308839798 seconds
Best log loss score: 0.1974472058461548
Best AUC score: 0.97613310930843

Class 15:
Best log loss: 0.23077802701389574
Best AUC: 0.9445833250004999
Best log loss: 0.22811947313801614
Best AUC: 0.9491929005852426
Best log loss: 0.24251130014276662
Best AUC: 0.9489624751491053
Best log loss: 0.26617685795212054
Best AUC: 0.9587954435605022
Best log loss: 0.19164812884391486
Best AUC: 0.9563842333954066
Fold time: 606.2063620090485 seconds
Best log loss score: 0.23184775060083418
Best AUC score: 0.9507165151970609

Class 16:
Best log loss: 0.256431130564825
Best AUC: 0.9644717785585375
Best log loss: 0.22385005951884268
Best AUC: 0.9635620465641866
Best log loss: 0.25767038116700974
Best AUC: 0.9654633224892433
Best log loss: 0.22202993941841004
Best AUC: 0.964412043352981
Best log loss: 0.2630350409576729
Best AUC: 0.9581374413632833
Fold time: 603.998811006546 seconds
Best log loss score: 0.24460285483104532
Best AUC score: 0.9628241034772262
'''