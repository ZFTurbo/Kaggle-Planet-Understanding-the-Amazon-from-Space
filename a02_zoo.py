# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import h5py
import numpy as np
import sys
sys.setrecursionlimit(5000)

WEIGHTS_PATH = '../weights/'


def f2beta_loss(Y_true, Y_pred):
    from keras import backend as K
    eps = 0.001
    false_positive = K.sum(Y_pred * (1 - Y_true), axis=-1)
    false_negative = K.sum((1 - Y_pred) * Y_true, axis=-1)
    true_positive = K.sum(Y_true * Y_pred, axis=-1)
    p = (true_positive + eps) / (true_positive + false_positive + eps)
    r = (true_positive + eps) / (true_positive + false_negative + eps)
    out = (5*p*r + eps) / (4*p + r + eps)
    return -K.mean(out)


def get_learning_rate(cnn_type):
    if cnn_type == 'VGG16' or cnn_type == 'VGG16_DROPOUT' or cnn_type == 'VGG16_MULTI':
        return 0.00004
    elif cnn_type == 'VGG16_ADDITIONAL_INPUT_LAYER':
        return 0.00001
    elif cnn_type == 'VGG16_MULTIPLY_INPUT_LAYER':
        return 0.00005
    elif cnn_type == 'VGG16_KERAS':
        return 0.00005
    elif cnn_type == 'VGG19':
        return 0.00004
    elif cnn_type == 'VGG19_KERAS':
        return 0.00005
    elif cnn_type == 'RESNET50' or cnn_type == 'RESNET50_DENSE_LAYERS':
        return 0.00003
    elif cnn_type == 'RESNET101':
        return 0.00003
    elif cnn_type == 'RESNET152':
        return 0.00003
    elif cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V3_DENSE_LAYERS':
        return 0.00003
    elif cnn_type == 'INCEPTION_V4':
        return 0.00003
    elif cnn_type == 'XCEPTION':
        return 0.00004
    elif cnn_type == 'SQUEEZE_NET':
        return 0.004
    elif cnn_type == 'DENSENET_121':
        return 0.00003
    elif cnn_type == 'DENSENET_169':
        return 0.00003
    elif cnn_type == 'DENSENET_161':
        return 0.00003
    else:
        print('Error Unknown CNN type for learning rate!!')
        exit()
    return 0.00005


def get_optim(cnn_type, optim_type, learning_rate=-1):
    from keras.optimizers import SGD
    from keras.optimizers import Adam

    if learning_rate == -1:
        lr = get_learning_rate(cnn_type)
    else:
        lr = learning_rate
    if optim_type == 'Adam':
        optim = Adam(lr=lr)
    else:
        optim = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    return optim


def get_random_state_v1(cnn_type):
    if cnn_type == 'VGG16' or cnn_type == 'VGG16_DROPOUT':
        return 51
    elif cnn_type == 'VGG16_MULTI' or cnn_type == 'VGG16_ADDITIONAL_INPUT_LAYER':
        return 151
    elif cnn_type == 'VGG16_MULTIPLY_INPUT_LAYER':
        return 152
    elif cnn_type == 'VGG19_KERAS':
        return 52
    elif cnn_type == 'RESNET50' or cnn_type == 'RESNET50_DENSE_LAYERS':
        return 53
    elif cnn_type == 'RESNET101':
        return 67
    elif cnn_type == 'RESNET152':
        return 68
    elif cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V3_DENSE_LAYERS':
        return 54
    elif cnn_type == 'VGG16_KERAS':
        return 55
    elif cnn_type == 'VGG19':
        return 56
    elif cnn_type == 'SQUEEZE_NET':
        return 66
    elif cnn_type == 'DENSENET_121':
        return 69
    elif cnn_type == 'DENSENET_169':
        return 70
    elif cnn_type == 'DENSENET_161':
        return 71
    elif cnn_type == 'INCEPTION_V4':
        return 72
    elif cnn_type == 'XCEPTION':
        return 73
    else:
        print('Error Unknown CNN Type for random state!!')
        exit()
    return 0


def get_random_state(cnn_type):
    return 69


def get_input_shape(cnn_type):
    if cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V3_DENSE_LAYERS' or cnn_type == 'INCEPTION_V4' or cnn_type == 'XCEPTION':
        return (299, 299)
    elif cnn_type == 'SQUEEZE_NET':
        return (227, 227)
    return (224, 224)


# Tuned for 6 GB of GPU memory
def get_batch_size(cnn_type):
    if cnn_type == 'VGG19' or cnn_type == 'VGG19_KERAS' or cnn_type == 'VGG19_MULTI' or cnn_type == 'VGG16_ADDITIONAL_INPUT_LAYER':
        return 20
    if cnn_type == 'VGG16_MULTIPLY_INPUT_LAYER':
        return 20
    if cnn_type == 'VGG16_DROPOUT':
        return 20
    if cnn_type == 'VGG16' or cnn_type == 'VGG16_KERAS':
        return 20
    if cnn_type == 'RESNET50' or cnn_type == 'RESNET50_DENSE_LAYERS':
        return 20
    if cnn_type == 'RESNET101':
        return 20
    if cnn_type == 'RESNET152':
        return 16
    if cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V3_DENSE_LAYERS':
        return 20
    if cnn_type == 'INCEPTION_V4':
        return 20
    if cnn_type == 'SQUEEZE_NET':
        return 20
    if cnn_type == 'DENSENET_121':
        return 20
    if cnn_type == 'DENSENET_169':
        return 16
    if cnn_type == 'DENSENET_161':
        return 15
    if cnn_type == 'XCEPTION':
        return 20
    return -1


def normalize_image_vgg16(img):
    img[:, 0, :, :] -= 103.939
    img[:, 1, :, :] -= 116.779
    img[:, 2, :, :] -= 123.68
    return img


def normalize_image_inception(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def normalize_image_densenet(img):
    img[:, 0, :, :] = (img[:, 0, :, :] - 103.94) * 0.017
    img[:, 1, :, :] = (img[:, 1, :, :] - 116.78) * 0.017
    img[:, 2, :, :] = (img[:, 2, :, :] - 123.68) * 0.017
    return img


def preprocess_input_overall(cnn_type, x):
    if cnn_type == 'INCEPTION_V3' or cnn_type == 'INCEPTION_V3_DENSE_LAYERS' or cnn_type == 'INCEPTION_V4' or cnn_type == 'XCEPTION':
        return normalize_image_inception(x.astype(np.float32))
    if 'DENSENET' in cnn_type:
        return normalize_image_densenet(x.astype(np.float32))
    return normalize_image_vgg16(x.astype(np.float32))


def VGG_16(classes_number):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

    # VGG16: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    f = h5py.File(WEIGHTS_PATH + 'vgg16_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.add(Dense(classes_number, activation='sigmoid'))
    return model


def VGG16_MULTI(classes_number, input_ch):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

    # VGG16: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(input_ch, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes_number, activation='sigmoid'))
    return model


# MIN: 0.98 Time: 130 sec
def VGG_16_KERAS(classes_number):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.vgg16 import VGG16
    from keras.models import Model

    base_model = VGG16(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    vgg16 = Model(input=base_model.input, output=x)
    return vgg16


# MIN: 1.00 Fast: 60 sec
def VGG_16_2_v2(classes_number):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.vgg16 import VGG16
    from keras.models import Model
    from keras.layers import Input

    input_tensor = Input(shape=(3, 224, 224))
    base_model = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    vgg16 = Model(input=base_model.input, output=x)
    return vgg16


def VGG_19(classes_number):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

    # VGG19: https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    f = h5py.File(WEIGHTS_PATH + 'vgg19_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.add(Dense(classes_number, activation='sigmoid'))
    return model


def VGG_19_KERAS(classes_number):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.vgg19 import VGG19
    from keras.models import Model

    base_model = VGG19(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    return model


def RESNET_50(classes_number):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.resnet50 import ResNet50
    from keras.models import Model

    base_model = ResNet50(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    return model


def RESNET50_DENSE_LAYERS(classes_number):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.resnet50 import ResNet50
    from keras.models import Model

    base_model = ResNet50(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    return model


# Batch 40 OK
def Inception_V3(classes_number):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model

    base_model = InceptionV3(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    return model


def Inception_V3_DENSE_LAYERS(classes_number):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model

    base_model = InceptionV3(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    return model


def Xception_wrapper(classes_number):
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.applications.xception import Xception
    from keras.models import Model

    # Only tensorflow
    base_model = Xception(include_top=True, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    return model


def Squeeze_Net(classes_number):
    from a01_squeezenet import get_squeezenet
    model = get_squeezenet(classes_number, dim_ordering='th')
    return model


def VGG16_WITH_DROPOUTS(classes_number, dropout=0.1):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
    base_model = VGG_16(classes_number)

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', weights=base_model.layers[1].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', weights=base_model.layers[3].get_weights()))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', weights=base_model.layers[6].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', weights=base_model.layers[8].get_weights()))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[11].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[13].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[15].get_weights()))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[18].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[20].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[22].get_weights()))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[25].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[27].get_weights()))
    model.add(Dropout(dropout))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[29].get_weights()))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', weights=base_model.layers[32].get_weights()))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', weights=base_model.layers[34].get_weights()))
    model.add(Dropout(0.5))
    model.add(Dense(classes_number, activation='sigmoid'))

    return model


def ResNet101(classes_number):
    from a01_resnet_101 import get_resnet101
    model = get_resnet101(classes_number, dim_ordering='th')
    return model


def ResNet152(classes_number):
    from a01_resnet_152 import get_resnet152
    model = get_resnet152(classes_number, dim_ordering='th')
    return model


def DenseNet121(classes_number, final_layer_activation):
    from a01_densenet_121 import DenseNet_121
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Model

    base_model = DenseNet_121(reduction=0.5, weights_path=WEIGHTS_PATH + 'densenet121_weights_th.h5')
    x = base_model.layers[-3].output
    del base_model.layers[-2:]
    x = Dense(classes_number, activation=final_layer_activation, name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    # print(model.summary())
    return model


def DenseNet169(classes_number):
    from a01_densenet_169 import DenseNet_169
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Model

    base_model = DenseNet_169(reduction=0.5, weights_path=WEIGHTS_PATH + 'densenet169_weights_th.h5')
    x = base_model.layers[-3].output
    del base_model.layers[-2:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    # print(model.summary())
    return model


def DenseNet161(classes_number):
    from a01_densenet_161 import DenseNet_161
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Model
    import keras.backend as K

    if K.image_dim_ordering() == 'th':
        model_path = WEIGHTS_PATH + 'densenet161_weights_th.h5'
    else:
        model_path = WEIGHTS_PATH + 'densenet161_weights_tf.h5'

    base_model = DenseNet_161(reduction=0.5, weights_path=model_path)
    x = base_model.layers[-3].output
    del base_model.layers[-2:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    # print(model.summary())
    return model


def VGG16_ADDITIONAL_INPUT_LAYER(classes_number, input_ch, trainable=False):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
    base_model = VGG_16(classes_number)

    feature_layers = [
        Convolution2D(3, 1, 1, activation='relu', input_shape=(input_ch, 224, 224)),
        Convolution2D(64, 3, 3, activation='relu', weights=base_model.layers[1].get_weights()),
        Convolution2D(64, 3, 3, activation='relu', weights=base_model.layers[3].get_weights()),
        Convolution2D(128, 3, 3, activation='relu', weights=base_model.layers[6].get_weights()),
        Convolution2D(128, 3, 3, activation='relu', weights=base_model.layers[8].get_weights()),
        Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[11].get_weights()),
        Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[13].get_weights()),
        Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[15].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[18].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[20].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[22].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[25].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[27].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[29].get_weights())
    ]
    classification_layers = [
        Dense(4096, activation='relu', weights=base_model.layers[32].get_weights()),
        Dense(4096, activation='relu', weights=base_model.layers[34].get_weights()),
        Dense(classes_number, activation='sigmoid'),
    ]

    if trainable == False:
        for i in range(1, len(feature_layers)):
            feature_layers[i].trainable = False
        for i in range(0, len(classification_layers) - 1):
            classification_layers[i].trainable = False

    model = Sequential()
    model.add(feature_layers[0])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[1])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[2])
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[3])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[4])
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[5])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[6])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[7])
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[8])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[9])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[10])
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[11])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[12])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[13])
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(classification_layers[0])
    model.add(Dropout(0.5))
    model.add(classification_layers[1])
    model.add(Dropout(0.5))
    model.add(classification_layers[2])

    return model


def VGG16_MULTIPLY_INPUT_LAYER(classes_number, input_ch, trainable=False):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
    base_model = VGG_16(classes_number)

    (weights, bias) = base_model.layers[1].get_weights()
    new_weights = np.zeros((weights.shape[0], input_ch, weights.shape[2], weights.shape[3]))
    for i in range(input_ch):
        new_weights[:, i, :, :] = weights[:, input_ch%3, :, :]
    new_weights = new_weights*3/input_ch

    feature_layers = [
        Convolution2D(64, 3, 3, activation='relu', weights=(new_weights, bias)),
        Convolution2D(64, 3, 3, activation='relu', weights=base_model.layers[3].get_weights()),
        Convolution2D(128, 3, 3, activation='relu', weights=base_model.layers[6].get_weights()),
        Convolution2D(128, 3, 3, activation='relu', weights=base_model.layers[8].get_weights()),
        Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[11].get_weights()),
        Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[13].get_weights()),
        Convolution2D(256, 3, 3, activation='relu', weights=base_model.layers[15].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[18].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[20].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[22].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[25].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[27].get_weights()),
        Convolution2D(512, 3, 3, activation='relu', weights=base_model.layers[29].get_weights())
    ]
    classification_layers = [
        Dense(4096, activation='relu', weights=base_model.layers[32].get_weights()),
        Dense(4096, activation='relu', weights=base_model.layers[34].get_weights()),
        Dense(classes_number, activation='sigmoid'),
    ]

    if trainable == False:
        for i in range(0, len(feature_layers)):
            feature_layers[i].trainable = False
        for i in range(0, len(classification_layers) - 1):
            classification_layers[i].trainable = False

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(input_ch, 224, 224)))
    model.add(feature_layers[0])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[1])
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[2])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[3])
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[4])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[5])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[6])
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[7])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[8])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[9])
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[10])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[11])
    model.add(ZeroPadding2D((1, 1)))
    model.add(feature_layers[12])
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(classification_layers[0])
    model.add(Dropout(0.5))
    model.add(classification_layers[1])
    model.add(Dropout(0.5))
    model.add(classification_layers[2])

    return model


def Inception_v4(classes_number):
    from a01_inception_v4 import inception_v4
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Model

    base_model = inception_v4(classes_number, 0.1, 'imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    x = Dense(classes_number, activation='sigmoid', name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    # print(model.summary())
    return model


def get_pretrained_model(cnn_type, classes_number, optim_name='Adam', learning_rate=-1, final_layer_activation='sigmoid'):
    import keras
    K = keras.backend.backend()
    if K == 'tensorflow':
        print('Update dim ordering to "tf"')
        keras.backend.set_image_dim_ordering('tf')

    if cnn_type == 'VGG16':
        model = VGG_16(classes_number)
    elif cnn_type == 'VGG16_DROPOUT':
        model = VGG16_WITH_DROPOUTS(classes_number, 0.1)
    elif cnn_type == 'VGG16_MULTI':
        model = VGG16_MULTI(classes_number, 7)
    elif cnn_type == 'VGG16_ADDITIONAL_INPUT_LAYER':
        model = VGG16_ADDITIONAL_INPUT_LAYER(classes_number, 7)
    elif cnn_type == 'VGG16_MULTIPLY_INPUT_LAYER':
        model = VGG16_MULTIPLY_INPUT_LAYER(classes_number, 7)
    elif cnn_type == 'VGG19':
        model = VGG_19(classes_number)
    elif cnn_type == 'VGG16_KERAS':
        model = VGG_16_KERAS(classes_number)
    elif cnn_type == 'VGG19_KERAS':
        model = VGG_19_KERAS(classes_number)
    elif cnn_type == 'RESNET50':
        model = RESNET_50(classes_number)
    elif cnn_type == 'RESNET50_DENSE_LAYERS':
        model = RESNET50_DENSE_LAYERS(classes_number)
    elif cnn_type == 'RESNET101':
        model = ResNet101(classes_number)
    elif cnn_type == 'RESNET152':
        model = ResNet152(classes_number)
    elif cnn_type == 'INCEPTION_V3':
        model = Inception_V3(classes_number)
    elif cnn_type == 'INCEPTION_V3_DENSE_LAYERS':
        model = Inception_V3_DENSE_LAYERS(classes_number)
    elif cnn_type == 'SQUEEZE_NET':
        model = Squeeze_Net(classes_number)
    elif cnn_type == 'DENSENET_121':
        model = DenseNet121(classes_number, final_layer_activation)
    elif cnn_type == 'DENSENET_161':
        model = DenseNet161(classes_number)
    elif cnn_type == 'DENSENET_169':
        model = DenseNet169(classes_number)
    elif cnn_type == 'INCEPTION_V4':
        model = Inception_v4(classes_number)
    elif cnn_type == 'XCEPTION':
        model = Xception_wrapper(classes_number)
    else:
        model = None
        print('Unknown CNN type: {}'.format(cnn_type))
        exit()

    optim = get_optim(cnn_type, optim_name, learning_rate)
    # model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f2beta_loss, 'accuracy'])
    # model.compile(optimizer=optim, loss='binary_crossentropy')
    if final_layer_activation == 'sigmoid':
        model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f2beta_loss])
    else:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['binary_crossentropy'])
    # print(model.summary())

    return model

'''
DenseNet: https://github.com/flyyufelix/DenseNet-Keras
ResNet-101: https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294
ResNet-152: https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6
SqueezeNet: https://github.com/rcmalli/keras-squeezenet
Inception v4: https://github.com/titu1994/Inception-v4/releases
VGG16: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
VGG19: https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
Other Keras models: https://keras.io/applications/
'''