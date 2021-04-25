
# conv8 =
# se1 = Conv(1, (1,1))(conv8)#, activation = ?)
# se1 = sigmoid(se1)
# sig_conv8 = sigmoid(conv8)
# mul = multiply([sig_conv8, se1])

import nibabel as nib
import glob as glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
# import numpy as np
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import *
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
# from tensorflow.keras import backend as K
# from tensorflow.python.keras.layers.normalization import BatchNormalization as bn
# from tensorflow.keras import regularizers
# from tensorflow.keras.preprocessing.image import *
# from tensorflow.keras.activations import *
# import pickle
# from sklearn.utils import shuffle
#
# from tensorflow.keras.utils import multi_gpu_model


from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.layers.normalization import BatchNormalization as bn
from keras import regularizers
from keras.preprocessing.image import *

from sklearn.utils import shuffle
from keras.utils import multi_gpu_model


os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# path = '/home/rutu_g/thesis/iSeg-2019-Training/iSeg-2019-Training'
# label_path = '/home/rutu_g/thesis/iSeg-2019-Training/iSeg-2019-Training'
# val_path = '/home/rutu_g/thesis/iSeg-2019-Validation'

#
path = '/home/rutu/thesis/iSeg-2019-Training'
label_path = '/home/rutu/thesis/iSeg-2019-Training'
val_path = '/home/rutu/thesis/iSeg-2019-Validation'


smooth = 1.
def dice_coef(y_true, y_pred):
    """ The dice coef is a metric to calculate the similarilty
    (intersection) between the true values and the predictions"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def load_data(path, i):

    sub_T1 = '/subject-%d-T1.img' %i
    sub_T2 = '/subject-%d-T2.img' %i
    sub_label = '/subject-%d-label.img' %i
    inter_train = nib.load(path + sub_T1)
    train_T1=inter_train.get_data()

    inter_train = nib.load(path + sub_T2)
    train_T2=inter_train.get_data()

    inter_label = nib.load(path + sub_label)
    label=inter_label.get_data()


    return train_T1, train_T2, label#, val
train_T1, train_T2, label = load_data(path, 1)

def load_data_val(path, i):

    sub_T1 = '/subject-%d-T1.img' %i
    sub_T2 = '/subject-%d-T2.img' %i
    #sub_label = '/subject-%d-label.img' %i
    inter_train = nib.load(path + sub_T1)
    train_T1=inter_train.get_data()

    inter_train = nib.load(path + sub_T2)
    train_T2=inter_train.get_data()

    # inter_label = nib.load(path + sub_label)
    # label=inter_label.get_data()


    return train_T1, train_T2#, label#, val
val_T1, val_T2 = load_data_val(path, 1)

#
# train_filenames = glob.glob('/home/rutu_g/thesis/iSeg-2019-Training/iSeg-2019-Training/*T1.img')
# val_filenames = glob.glob('/home/rutu_g/thesis/iSeg-2019-Validation/*T1.img')

train_filenames = glob.glob('/home/rutu/thesis/iSeg-2019-Training/*T1.img')
val_filenames = glob.glob('/home/rutu/thesis/iSeg-2019-Validation/*T1.img')
len(val_filenames)#np.save('np_train_T1_chunked', np_train_T1_chunked)
np_train_T1 = np.ndarray(shape = (1, 144, 192, 256, 1))
np_train_T2 = np.ndarray(shape = (1, 144, 192, 256, 1))
np_label = np.ndarray(shape = (1, 144, 192, 256, 1))
for i in range(1,len(train_filenames)+1):
    print("i......", i)
    train_T1, train_T2, labels = load_data(path, i)
    train_T1_exp = np.expand_dims(train_T1, axis=0)
    train_T2_exp = np.expand_dims(train_T2, axis=0)
    label_exp = np.expand_dims(labels, axis=0)
    np_train_T1 = np.concatenate((np_train_T1, train_T1_exp), axis=0)
    #print(np_train_T1.shape)

    np_train_T2 = np.concatenate((np_train_T2, train_T2_exp), axis=0)
    np_label = np.concatenate((np_label, label_exp), axis=0)
np_train_T1 = np.array(np_train_T1[1:])
np_train_T2 = np.array(np_train_T2[1:])
np_label = np.array(np_label[1:])
print("np_train_T1", np_train_T1.shape)
print("[[[[[[[[np_label]]]]]]]]", np_label.shape)
print("unique np_label", np.unique(np_label))

# one_hot = np.load('/home/rutu_g/thesis/one_hot_encoded_newer.npy')
one_hot = np.load('/home/rutu/thesis/one_hot_encoded_newer.npy')

y_l = one_hot

# X_train = np_train_T1[:8]
print("......................np_train_T1",len(np_train_T1))
X_train = np.concatenate((np_train_T1[0:6], np_train_T1[8:]))
X_train = X_train/1000
# X_test = np.concatenate((np_train_T1[0], np_train_T1[9]))
X_test = np_train_T1[6:8]
X_test = X_test/1000
print("np_train_T1 unique", np.unique(np_train_T1[0]))
print("y one hot uni", np.unique(y_l[0]))
print("((((((((((((((((((((np_train_T1", X_train.shape)
print("X_train", X_train.shape)
# y_train  = y_l[:8]
y_train = np.concatenate((y_l[0:6], y_l[8:]))
# y_test = np.concatenate((y_l[0], y_l[9]))
y_test = y_l[6:8]
print("y", y_l.shape)
X1 = X_train.shape
ys = y_train.shape
X_train = np.reshape(X_train, (X1[0]*X1[1], X1[2], X1[3], X1[4]))
y_train = np.reshape(y_train, (ys[0]*ys[1], ys[2], ys[3], ys[4]))


"""Normalize!!!!"""

def UNet(input_shape_1):
    new_inputs = Input(input_shape_1)
    # new_inputs_2 = Input(input_shape_2)

    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size = 3



    sess = K.get_session()
    # combine = concatenate([pwc, new_inputs_2], name = 'combine')

    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(new_inputs)

    conv1 = bn()(conv1)

    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv1)

    conv1 = bn()(conv1)

    se1 = Conv2D(1, (1,1), activation = 'softmax')(conv1)#, activation = ?)
    print("se1 shape....", K.int_shape(se1))
    mul_1 = multiply([conv1, se1], name = 'mul_1')
    print("mul_1 shape....", K.int_shape(mul_1))

    init_cs = K.int_shape(conv1)
    print("Look here..........init_cs", init_cs)

    # conv1_int = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], 1)))(conv1)
    # cs = K.int_shape(conv1_int)
    # print("Look here..........cs", cs)
    # se1 = Conv2D(1, (int(cs[1]), 1), activation = 'relu')(conv1_int)
    # print("H*1 filter", se1.shape)
    # se1_r = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], init_cs[2],init_cs[3])))(se1)
    # print("After reshaping wc", K.int_shape(se1_r))
    # se1_r_shape = K.int_shape(se1_r)
    # print("se1_r_shape", se1_r_shape)
    # se1 = Conv2D(32, (1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    conv1_int = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], 1)))(conv1)
    cs = K.int_shape(conv1_int)
    print("Look here..........cs", cs)
    se1 = DepthwiseConv2D((int(cs[1]), 1), activation = 'relu')(conv1_int)
    print("H*1 filter", se1.shape)
    se1_r = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], init_cs[2],init_cs[3])))(se1)

    print("After reshaping wc", K.int_shape(se1_r))

    se1_r_shape = K.int_shape(se1_r)
    print("se1_r_shape", se1_r_shape)
    se1_r = Lambda(lambda x: tf.reverse(x, [1]))(se1_r)
    se1 = DepthwiseConv2D((1, int(init_cs[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)

    print("************se1", se1.shape)
    # se1 = DepthwiseConv2D((1, int(init_cs[2])), (1, init_cs[2]), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = Permute((1, 3, 2))(se1)
    # print("...............se1 permuted", se1.shape)

    # se1 = GlobalAveragePooling2D()(conv1)
    # print("new GAP", se1.shape)
    # se1 = Dense(16, activation = 'relu')(se1)
    # print("new D", se1.shape)
    # se1 = Dense(32, activation = 'sigmoid')(se1)
    # print("************bn1", conv1.shape)
    # print("************se1", se1.shape)

    mul = multiply([conv1, se1])
    print("mul shape......", K.int_shape(mul))

    mul_scse = Add()([mul_1, mul])
    print("mul_scse shape......", K.int_shape(mul_scse))
    print("*********************sen1 shape: ",K.int_shape(se1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(mul_scse)

    pool1 = Dropout(DropP)(pool1)

    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool1)

    conv2 = bn()(conv2)

    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv2)

    conv2 = bn()(conv2)

    se1 = Conv2D(1, (1,1), activation = 'softmax')(conv2)#, activation = ?)
    print("se1 shape....", K.int_shape(se1))
    init_cs = K.int_shape(conv2)
    print("Look here..........init_cs", init_cs)
    # se1 = sigmoid(se1)
    # sig_conv2 = sigmoid(conv2)
    mul_1 = multiply([conv2, se1])




    print("mul_1 shape....", K.int_shape(mul_1))
    conv1_int = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], 1)))(conv2)
    cs = K.int_shape(conv1_int)
    print("Look here..........cs", cs)
    se1 = DepthwiseConv2D((int(cs[1]), 1), activation = 'relu')(conv1_int)
    print("H*1 filter", se1.shape)
    se1_r = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], init_cs[2],init_cs[3])))(se1)
    print("After reshaping wc", K.int_shape(se1_r))
    se1_r_shape = K.int_shape(se1_r)
    print("se1_r_shape", se1_r_shape)
    se1_r = Lambda(lambda x: tf.reverse(x, [1]))(se1_r)

    se1 = DepthwiseConv2D((1, int(init_cs[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)

    print("************se1", se1.shape)
    # se1 = Permute((1, 3, 2))(se1)
    # print("...............se1 permuted", se1.shape)




    # cs = K.int_shape(conv2)
    # print("cs", cs)
    # se1 = Conv2D(64, (int(cs[1]), 1), activation = 'relu')(conv2)
    # print("H*1 filter", se1.shape)
    #
    # se1 = Conv2D(64, (1, int(cs[2])), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = GlobalAveragePooling2D()(conv2)
    # print("new GAP", se1.shape)
    # se1 = Dense(32, activation = 'relu')(se1)
    # print("new D", se1.shape)
    # se1 = Dense(64, activation = 'sigmoid')(se1)
    # print("************bn1", conv2.shape)
    # print("************se1", se1.shape)

    mul = multiply([conv2, se1])
    print("mul shape......", K.int_shape(mul))
    mul_scse = Add()([mul_1, mul])
    print("mul_scse shape......", K.int_shape(mul_scse))
    print("*********************sen1 shape: ",K.int_shape(se1))

    pool2 = MaxPooling2D(pool_size=(2, 2))(mul_scse)

    pool2 = Dropout(DropP)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool2)

    conv3 = bn()(conv3)

    conv3 = Conv2D(128, (3, 3), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv3)

    conv3 = bn()(conv3)

    se1 = Conv2D(1, (1,1), activation = 'softmax')(conv3)#, activation = ?)
    # se1 = sigmoid(se1)
    # sig_conv3 = sigmoid(conv3)
    init_cs = K.int_shape(conv3)
    print("Look here..........init_cs", init_cs)
    mul_1 = multiply([conv3, se1])
    conv1_int = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], 1)))(conv3)
    cs = K.int_shape(conv1_int)
    print("Look here..........cs", cs)
    se1 = DepthwiseConv2D((int(cs[1]), 1), activation = 'relu')(conv1_int)
    print("H*1 filter", se1.shape)
    se1_r = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], init_cs[2],init_cs[3])))(se1)
    print("After reshaping wc", K.int_shape(se1_r))
    se1_r_shape = K.int_shape(se1_r)
    print("se1_r_shape", se1_r_shape)
    se1_r = Lambda(lambda x: tf.reverse(x, [1]))(se1_r)

    se1 = DepthwiseConv2D((1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)

    print("************se1", se1.shape)
    # se1 = DepthwiseConv2D((1, int(init_cs[2])), (1, init_cs[2]), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = Permute((1, 3, 2))(se1)
    # print("...............se1 permuted", se1.shape)

        # cs = K.int_shape(conv3)
    # print("cs", cs)
    # se1 = Conv2D(128, (int(cs[1]), 1), activation = 'relu')(conv3)
    # print("H*1 filter", se1.shape)
    #
    # se1 = Conv2D(128, (1, int(cs[2])), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = GlobalAveragePooling2D()(conv3)
    # print("new GAP", se1.shape)
    # se1 = Dense(64, activation = 'relu')(se1)
    # print("new D", se1.shape)
    # se1 = Dense(128, activation = 'sigmoid')(se1)
    # print("************bn1", conv3.shape)
    # print("************se1", se1.shape)
    mul = multiply([conv3, se1])

    mul_scse = Add()([mul_1, mul])

    print("*********************sen1 shape: ",K.int_shape(se1))

    pool3 = MaxPooling2D(pool_size=(2, 2))(mul_scse)

    pool3 = Dropout(DropP)(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool3)
    conv4 = bn()(conv4)

    conv4 = Conv2D(256, (3, 3), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv4)

    conv4 = bn()(conv4)

    se1 = Conv2D(1, (1,1), activation='softmax')(conv4)#, activation = ?)
    # se1 = sigmoid(se1)
    # sig_conv4 = sigmoid(conv4)
    init_cs = K.int_shape(conv4)
    print("Look here..........init_cs", init_cs)
    mul_1 = multiply([conv4, se1])
    conv1_int = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], 1)))(conv4)
    cs = K.int_shape(conv1_int)
    print("Look here..........cs", cs)
    se1 = DepthwiseConv2D((int(cs[1]), 1), activation = 'relu')(conv1_int)
    print("H*1 filter", se1.shape)
    se1_r = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], init_cs[2],init_cs[3])))(se1)
    print("After reshaping wc", K.int_shape(se1_r))
    se1_r_shape = K.int_shape(se1_r)
    print("se1_r_shape", se1_r_shape)
    se1_r = Lambda(lambda x: tf.reverse(x, [1]))(se1_r)

    se1 = DepthwiseConv2D((1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)

    print("************se1", se1.shape)
    # se1 = DepthwiseConv2D((1, int(init_cs[2])), (1, init_cs[2]), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = Permute((1, 3, 2))(se1)
    # print("...............se1 permuted", se1.shape)

    # cs = K.int_shape(conv4)
    # print("cs", cs)
    # se1 = Conv2D(256, (int(cs[1]), 1), activation = 'relu')(conv4)
    # print("H*1 filter", se1.shape)
    #
    # se1 = Conv2D(256, (1, int(cs[2])), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = GlobalAveragePooling2D()(conv4)
    # print("new GAP", se1.shape)
    # se1 = Dense(128, activation = 'relu')(se1)
    # print("new D", se1.shape)
    # se1 = Dense(256, activation = 'sigmoid')(se1)
    # print("************bn1", conv4.shape)
    # print("************se1", se1.shape)
    mul = multiply([conv4, se1])

    mul_scse = Add()([mul_1, mul])

    print("*********************sen1 shape: ",K.int_shape(se1))

    pool4 = MaxPooling2D(pool_size=(2, 2))(mul_scse)

    pool4 = Dropout(DropP)(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool4)

    conv5 = bn()(conv5)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv5)

    conv5 = bn()(conv5)

    # remember.append(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2),
                                       padding='same')(conv5), conv4], name='up6', axis=3)

    up6 = Dropout(DropP)(up6)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up6)

    conv6 = bn()(conv6)

    conv6 = Conv2D(256, (3, 3), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv6)

    conv6 = bn()(conv6)

    se1 = Conv2D(1, (1,1), activation = 'softmax')(conv6)#, activation = ?)
    # se1 = sigmoid(se1)
    # sig_conv6 = sigmoid(conv6)
    init_cs = K.int_shape(conv6)
    print("Look here..........init_cs", init_cs)
    mul_1 = multiply([conv6, se1])
    conv1_int = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], 1)))(conv6)
    cs = K.int_shape(conv1_int)
    print("Look here..........cs", cs)
    se1 = DepthwiseConv2D((int(cs[1]), 1), activation = 'relu')(conv1_int)
    print("H*1 filter", se1.shape)
    se1_r = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], init_cs[2],init_cs[3])))(se1)
    print("After reshaping wc", K.int_shape(se1_r))
    se1_r_shape = K.int_shape(se1_r)
    print("se1_r_shape", se1_r_shape)
    se1_r = Lambda(lambda x: tf.reverse(x, [1]))(se1_r)

    se1 = DepthwiseConv2D((1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)

    print("************se1", se1.shape)
    # se1 = DepthwiseConv2D((1, int(init_cs[2])), (1, init_cs[2]), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = Permute((1, 3, 2))(se1)
    # print("...............se1 permuted", se1.shape)

    # cs = K.int_shape(conv6)
    # print("cs", cs)
    # se1 = Conv2D(256, (int(cs[1]), 1), activation = 'relu')(conv6)
    # print("H*1 filter", se1.shape)
    #
    # se1 = Conv2D(256, (1, int(cs[2])), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = GlobalAveragePooling2D()(conv6)
    # print("new GAP", se1.shape)
    # se1 = Dense(128, activation = 'relu')(se1)
    # print("new D", se1.shape)
    # se1 = Dense(256, activation = 'sigmoid')(se1)
    # print("************bn1", conv6.shape)
    # print("************se1", se1.shape)
    mul = multiply([conv6, se1])

    mul_scse = Add()([mul_1, mul])

    print("*********************sen1 shape: ",K.int_shape(se1))
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(mul_scse), conv3], name='up7', axis=3)

    up7 = Dropout(DropP)(up7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up7)

    conv7 = bn()(conv7)

    conv7 = Conv2D(128, (3, 3), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv7)

    conv7 = bn()(conv7)

    se1 = Conv2D(1, (1,1), activation='softmax')(conv7)#, activation = ?)
    # se1 = sigmoid(se1)
    # sig_conv7 = sigmoid(conv7)
    init_cs = K.int_shape(conv7)
    print("\n\n\nLook here..........init_cs", init_cs)
    mul_1 = multiply([conv7, se1])
    print("conv7 shape....", K.int_shape(conv7))
    print("se1 shape....", K.int_shape(se1))
    print("mul_1 shape....", K.int_shape(mul_1))
    conv1_int = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], 1)))(conv7)
    cs = K.int_shape(conv1_int)
    print("Look here..........cs", cs)
    se1 = DepthwiseConv2D((int(cs[1]), 1), activation = 'relu')(conv1_int)
    print("H*1 filter", se1.shape)
    se1_r = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], init_cs[2],init_cs[3])))(se1)
    print("After reshaping wc", K.int_shape(se1_r))
    se1_r_shape = K.int_shape(se1_r)
    print("se1_r_shape", se1_r_shape)
    se1_r = Lambda(lambda x: tf.reverse(x, [1]))(se1_r)

    se1 = DepthwiseConv2D((1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)

    print("************se1", se1.shape)
    # se1 = DepthwiseConv2D((1, int(init_cs[2])), (1, init_cs[2]), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = Permute((1, 3, 2))(se1)
    # print("...............se1 permuted", se1.shape)

    # cs = K.int_shape(conv7)
    # print("cs", cs)
    # se1 = Conv2D(128, (int(cs[1]), 1), activation = 'relu')(conv7)
    # print("H*1 filter", se1.shape)
    #
    # se1 = Conv2D(128, (1, int(cs[2])), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = GlobalAveragePooling2D()(conv7)
    # print("new GAP", se1.shape)
    # se1 = Dense(64, activation = 'relu')(se1)
    # print("new D", se1.shape)
    # se1 = Dense(128, activation = 'sigmoid')(se1)
    # print("************bn1", conv7.shape)
    # print("************se1", se1.shape)

    mul = multiply([conv7, se1])
    print("conv7................", K.int_shape(conv7))
    print("se1..............", K.int_shape(se1))
    print("mul...............", K.int_shape(mul))

    mul_scse = Add()([mul_1, mul])

    print("*********************sen1 shape: ",K.int_shape(se1))
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(mul_scse), conv2], name='up8', axis=3)

    up8 = Dropout(DropP)(up8)

    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up8)

    conv8 = bn()(conv8)

    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv8)

    conv8 = bn()(conv8)

    se1 = Conv2D(1, (1,1), activation='softmax')(conv8)#, activation = ?)
    # se1 = sigmoid(se1)
    # sig_conv8 = sigmoid(conv8)
    init_cs = K.int_shape(conv8)
    print("Look here..........init_cs", init_cs)
    mul_1 = multiply([conv8, se1])
    print("conv8 shape....", K.int_shape(conv8))
    print("se1 shape....", K.int_shape(se1))
    print("mul_1 shape....", K.int_shape(mul_1))
    conv1_int = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], 1)))(conv8)
    cs = K.int_shape(conv1_int)
    print("Look here..........cs", cs)
    se1 = DepthwiseConv2D((int(cs[1]), 1), activation = 'relu')(conv1_int)
    print("H*1 filter", se1.shape)
    se1_r = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], init_cs[2],init_cs[3])))(se1)
    print("After reshaping wc", K.int_shape(se1_r))
    se1_r_shape = K.int_shape(se1_r)
    print("se1_r_shape", se1_r_shape)
    se1_r = Lambda(lambda x: tf.reverse(x, [1]))(se1_r)

    se1 = DepthwiseConv2D((1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)

    print("************se1", se1.shape)
    # se1 = DepthwiseConv2D((1, int(init_cs[2])), (1, init_cs[2]), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = Permute((1, 3, 2))(se1)
    # print("...............se1 permuted", se1.shape)

    # cs = K.int_shape(conv8)
    # print("cs", cs)
    # se1 = Conv2D(64, (int(cs[1]), 1), activation = 'relu')(conv8)
    # print("H*1 filter", se1.shape)
    #
    # se1 = Conv2D(64, (1, int(cs[2])), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = GlobalAveragePooling2D()(conv8)
    # print("new GAP", se1.shape)
    # se1 = Dense(32, activation = 'relu')(se1)
    # print("new D", se1.shape)
    # se1 = Dense(64, activation = 'sigmoid')(se1)
    # print("************bn1", conv8.shape)
    # print("************se1", se1.shape)
    mul = multiply([conv8, se1])
    print("conv8 shape....", K.int_shape(conv8))
    print("se1 shape....", K.int_shape(se1))
    print("mul shape....", K.int_shape(mul))
    mul_scse = Add()([mul_1, mul])

    print("*********************sen1 shape: ",K.int_shape(se1))
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mul_scse), conv1], name='up9', axis=3)

    up9 = Dropout(DropP)(up9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up9)

    conv9 = bn()(conv9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv9)

    conv9 = bn()(conv9)

    se1 = Conv2D(1, (1,1), activation='softmax')(conv9)#, activation = ?)
    # se1 = sigmoid(se1)
    # sig_conv9 = sigmoid(conv9)
    init_cs = K.int_shape(conv9)
    print("Look here..........init_cs", init_cs)
    mul_1 = multiply([conv9, se1])
    conv1_int = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], 1)))(conv9)
    cs = K.int_shape(conv1_int)
    print("Look here..........cs", cs)
    se1 = DepthwiseConv2D((int(cs[1]), 1), activation = 'relu')(conv1_int)
    print("H*1 filter", se1.shape)
    se1_r = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], init_cs[2],init_cs[3])))(se1)
    print("After reshaping wc", K.int_shape(se1_r))
    se1_r_shape = K.int_shape(se1_r)
    print("se1_r_shape", se1_r_shape)
    se1_r = Lambda(lambda x: tf.reverse(x, [1]))(se1_r)

    se1 = DepthwiseConv2D((1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)

    print("************se1", se1.shape)
    # se1 = DepthwiseConv2D((1, int(init_cs[2])), (1, init_cs[2]), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #
    # print("************se1", se1.shape)
    # se1 = Permute((1, 3, 2))(se1)
    # print("...............se1 permuted", se1.shape)

    # cs = K.int_shape(conv9)
    # print("cs", cs)
    # se1 = Conv2D(32, (int(cs[1]), 1), activation = 'relu')(conv9)
    # print("H*1 filter", se1.shape)
    #
    # se1 = Conv2D(32, (1, int(cs[2])), activation = 'softmax')(se1)
    # print("W*1 filter", se1.shape)
    #


    mul = multiply([conv9, se1])

    mul_scse = Add()([mul_1, mul])

    # print("*********************sen1 shape: ",K.int_shape(se1))

    conv22 = Conv2D(1, (1, 1), activation = 'sigmoid', name='conv22')(mul_scse) #

    # print("conv22: ", conv22.type())

    conv23 = Conv2D(1, (1, 1), activation = 'sigmoid', name = 'conv23')(mul_scse)

    conv24 = Conv2D(1, (1, 1), activation = 'sigmoid', name = 'conv24')(mul_scse)

    # # conv25 = Conv2D(1, (1, 1), activation = 'softmax', name = 'conv25')(lr18)
    # #
    model = Model(inputs = [new_inputs], outputs = [conv22, conv23, conv24])
    #model = Model(inputs = new_inputs, outputs = conv22)


    # model.compile(optimizer = Adam(lr = 5e-5), loss = {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss, 'conv25': dice_coef_loss}, metrics = [dice_coef])

    model.compile(optimizer = Adam(lr = 5e-5), loss = {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss}, metrics = [dice_coef])

    model.summary()

    # if(pretrained_weights):
    #   model.load_weights(pretrained_weights)

    return model
    # print(model.input.shape)




# print("Validation data", X_test[0:144].shape)
# model = load_model("experiment_2_unet_256_200epochs.h5", custom_objects={'dice_coef_loss':dice_coef_loss,'dice_coef' : dice_coef})

model = UNet(( 192, 256, 1))
# model=load_model('newarch_unet_concurrent_144_300.h5', custom_objects={'dice_coef_loss':dice_coef_loss,'dice_coef' : dice_coef, 'tf':tf})
# print("plain_unet_ordering_raunak_parameters_test_2_500.h5")
# model = load_weights('/home/rutu_g/thesis/plain_unet_ordering_75_again.h5')
es = EarlyStopping(monitor = 'val_loss', min_delta = -0.001 , patience = 5)
mc = ModelCheckpoint("concurrent_144_model_checkpoint.h5", monitor = 'val_loss', period = 100)
# parallel_model = multi_gpu_model(model, gpus = 3)
model.compile(optimizer = Adam(lr = 5e-5), loss =  {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss}, metrics = [dice_coef] )
history = model.fit([X_train], [y_train[:, :,:,0],y_train[ :,:,:,1],y_train[ :,:,:,2]],  batch_size=3, epochs=300, verbose = 2, shuffle=True)#, validation_data = (X_test[144:288], (y_test[144:288, :, :, 0], y_test[144:288, :, :, 1], y_test[144:288, :, :, 2] )) )
model.save("newarch_reshape_reverse_300_trial2_fold2.h5")

#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig('plain_unet_concurrent_144_300.png')




X1 = X_test.shape
ys = y_test.shape
print("X1", X1)
print("ys", ys)
X_test = np.reshape(X_test, (X1[0]*X1[1], X1[2], X1[3], X1[4]))
y_test = np.reshape(y_test, (ys[0]*ys[1], ys[2], ys[3], ys[4]))


y_pred = model.predict([X_test])
y_test = y_test
y_pred = np.array(y_pred)
# y_pred[y_pred <= 0.5 ] = 0
# y_pred[y_pred > 0.5] = 1

print("y_test", y_test.shape)
print("y_pred", y_pred.shape)
print("y_pred unique", np.unique(y_pred))
d1 = 0
d2 = 0
d3 = 0
count = 0
for i in range(0, 288):
    count = count + 1
    dl = dice_coef_loss(y_test[i][ : ,: , 0].astype(np.float32), y_pred[0][ i, :, :, :])
    dl1 = dice_coef_loss(y_test[i][ : ,: , 1].astype(np.float32), y_pred[1][ i, :, :, :])
    dl2 = dice_coef_loss(y_test[i][ : ,: , 2].astype(np.float32), y_pred[2][ i, :, :, :])
    d1 = d1 + dl
    d2 = d2 + dl1
    d3 = d3 + dl2

print("count1 ", count)
sess = K.get_session()
print("------------all------dice")
print("newarch_reshape_reverse_300_trial2_fold2.....fold2")
print(sess.run(d1/count))
print(sess.run(d2/count))
print(sess.run(d3/count))

final_dice_1=0
final_dice_2=0
final_dice_3=0

count=0
# for i in range(0,len(X_test)):
for i in range(0, 288):
	count=count+1
	final_dice_1=final_dice_1+dice_coef_loss(y_pred[0][i],y_test[i,:,:,0].astype(np.float32))
	final_dice_2=final_dice_2+dice_coef_loss(y_pred[1][i],y_test[i,:,:,1].astype(np.float32))
	final_dice_3=final_dice_3+dice_coef_loss(y_pred[2][i],y_test[i,:,:,2].astype(np.float32))
print("count2", count)
final_dice_1=final_dice_1/count
final_dice_2=final_dice_2/count
final_dice_3=final_dice_3/count
print(final_dice_1,final_dice_2,final_dice_3)
file1 = open('myfile_trial3.txt', 'w')

sess = K.get_session()
# print(sess.run(final_dice_1))
# print(sess.run(final_dice_2))
# print(sess.run(final_dice_3))
file1.write(str(sess.run(final_dice_1)))
file1.write(str(sess.run(final_dice_2)))
file1.write(str(sess.run(final_dice_3)))
file1.close()

import cv2
y_pred = y_pred*255
y_test = y_test*255
# for i in range(0, 256):
#     cv2.imwrite("plain_%d_pred.png" %i, y_pred[0][i])
#     cv2.imwrite("plain_%d_test.png" %i, y_test[i,:,:,0])
