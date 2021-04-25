import nibabel as nib
import glob as glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.layers.normalization import BatchNormalization as bn
from keras import regularizers
from keras.preprocessing.image import *
#
# from sklearn.utils import shuffle
from keras.utils import multi_gpu_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2


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

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

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
train_filenames = glob.glob('/home/rutu/thesis/iSeg-2019-Training/*T1.img')
val_filenames = glob.glob('/home/rutu/thesis/iSeg-2019-Validation/*T1.img')

len(val_filenames)
np_train_T1 = np.ndarray(shape = (1, 144, 192, 256, 1))
np_train_T2 = np.ndarray(shape = (1, 144, 192, 256, 1))
np_label = np.ndarray(shape = (1, 144, 192, 256, 1))
for i in range(1,len(train_filenames)+1):
    print("Running 500 epochs*********** ")

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


#
def UNet(input_shape_1, input_shape_2):
    new_inputs = Input(input_shape_1)
    new_inputs_2 = Input(input_shape_2)

    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size = 3

    #Change shape from (144, 192, 256) to (192, 256, 144)
    new_inputs_permuted = Permute((2, 3, 1))(new_inputs)
    print("new_inputs_permuted", new_inputs_permuted.shape)

    #Channel squeeze, Spatial Excite
    se1 = Conv2D(1, (1,1), activation = 'softmax')(new_inputs_permuted)
    print("se1 shape....", K.int_shape(se1))                                             #
    mul_1 = multiply([new_inputs_permuted, se1], name = 'mul_1')                         #
    print("mul_1 shape....", K.int_shape(mul_1))                                         #

    init_cs = K.int_shape(new_inputs_permuted)
    print("Look here..........init_cs", init_cs)

    #Spatial squeeze, Channel excite (ssce)
    conv1_int = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], 1)))(new_inputs_permuted)
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
    new_inputs_p = Permute( (2, 3, 1))(new_inputs)                                     #
    print("************new_inputs", new_inputs_p.shape)
    print("************se1", se1.shape)


    mul = multiply([new_inputs_p, se1])
    print("mul.........", mul.shape)
    mul_scse = Add()([mul_1, mul])                                                      #
    print("mul_scse........", mul_scse.shape)
    # mul = Permute( (3, 2, 1))(mul)
    mul = Permute( (3, 2, 1))(mul_scse)
    print("*****mul",mul.shape)

    mul1 = Permute((2, 3, 1))(mul)
    print("!!!!!!!!!!!!!!!!!!!!mul1:", mul1.shape)

    # pwc = Conv2D(1, (1, 1),  activation = 'relu', name = 'pwc')(mul1)
    pwc = Conv2D(1, (1, 1),  activation = 'relu', name = 'pwc')(mul_scse)
    print("!!!!!!!!!!!!!!!!!pwc_before:", pwc.shape)


    #2d attention-augmented UNet begins with two slices combined together as input
    #Downsample layer 1
    combine = concatenate([pwc, new_inputs_2], name = 'combine')
    sess = K.get_session()
    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(combine)

    conv1 = bn()(conv1)

    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv1)

    conv1 = bn()(conv1)

    #MSE left branch layer1
    se1 = Conv2D(1, (1,1), activation = 'softmax')(conv1)#, activation = ?)
    print("se1 shape....", K.int_shape(se1))
    mul_1 = multiply([conv1, se1], name = 'mul_11')
    print("mul_1 shape....", K.int_shape(mul_1))

    init_cs = K.int_shape(conv1)
    print("Look here..........init_cs", init_cs)

    #MSE right branch layer1
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
    se1 = DepthwiseConv2D((1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)
    print("************se1", se1.shape)
    mul = multiply([conv1, se1])
    print("mul shape......", K.int_shape(mul))
    mul_scse = Add()([mul_1, mul])
    print("mul_scse shape......", K.int_shape(mul_scse))
    print("*********************sen1 shape: ",K.int_shape(se1))

    #Downsample layer 2
    pool1 = MaxPooling2D(pool_size=(2, 2))(mul_scse)

    pool1 = Dropout(DropP)(pool1)

    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool1)

    conv2 = bn()(conv2)

    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv2)

    conv2 = bn()(conv2)

    #MSE left branch layer2
    se1 = Conv2D(1, (1,1), activation = 'softmax')(conv2)#, activation = ?)
    print("se1 shape....", K.int_shape(se1))
    init_cs = K.int_shape(conv2)
    print("Look here..........init_cs", init_cs)
    mul_1 = multiply([conv2, se1])
    print("mul_1 shape....", K.int_shape(mul_1))
    #MSE right branch layer1
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
    se1 = DepthwiseConv2D((1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)
    print("************se1", se1.shape)
    mul = multiply([conv2, se1])
    print("mul shape......", K.int_shape(mul))
    mul_scse = Add()([mul_1, mul])
    print("mul_scse shape......", K.int_shape(mul_scse))
    print("*********************sen1 shape: ",K.int_shape(se1))

    #Downsample layer 3
    pool2 = MaxPooling2D(pool_size=(2, 2))(mul_scse)

    pool2 = Dropout(DropP)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool2)

    conv3 = bn()(conv3)

    conv3 = Conv2D(128, (3, 3), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv3)

    conv3 = bn()(conv3)

    se1 = Conv2D(1, (1,1), activation = 'softmax')(conv3)

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
    mul = multiply([conv3, se1])
    mul_scse = Add()([mul_1, mul])

    print("*********************sen1 shape: ",K.int_shape(se1))

    #Downsample layer 4
    pool3 = MaxPooling2D(pool_size=(2, 2))(mul_scse)

    pool3 = Dropout(DropP)(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool3)
    conv4 = bn()(conv4)

    conv4 = Conv2D(256, (3, 3), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv4)

    conv4 = bn()(conv4)

    se1 = Conv2D(1, (1,1), activation='softmax')(conv4)#, activation = ?)
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
    mul = multiply([conv4, se1])
    mul_scse = Add()([mul_1, mul])
    print("*********************sen1 shape: ",K.int_shape(se1))

    #Bottom layer
    pool4 = MaxPooling2D(pool_size=(2, 2))(mul_scse)

    pool4 = Dropout(DropP)(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool4)

    conv5 = bn()(conv5)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv5)

    conv5 = bn()(conv5)

    #Upsample layer 1
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2),
                                       padding='same')(conv5), conv4], name='up6', axis=3)

    up6 = Dropout(DropP)(up6)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up6)

    conv6 = bn()(conv6)

    conv6 = Conv2D(256, (3, 3), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv6)

    conv6 = bn()(conv6)

    #MSE left branch
    se1 = Conv2D(1, (1,1), activation = 'softmax')(conv6)#, activation = ?)
    init_cs = K.int_shape(conv6)
    print("Look here..........init_cs", init_cs)
    mul_1 = multiply([conv6, se1])
    #MSE right branch
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
    mul = multiply([conv6, se1])
    mul_scse = Add()([mul_1, mul])
    print("*********************sen1 shape: ",K.int_shape(se1))

    #Upsample layer 2
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(mul_scse), conv3], name='up7', axis=3)

    up7 = Dropout(DropP)(up7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up7)

    conv7 = bn()(conv7)

    conv7 = Conv2D(128, (3, 3), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv7)

    conv7 = bn()(conv7)

    #MSE left branch
    se1 = Conv2D(1, (1,1), activation='softmax')(conv7)#, activation = ?)
    init_cs = K.int_shape(conv7)
    print("\n\n\nLook here..........init_cs", init_cs)
    mul_1 = multiply([conv7, se1])
    print("conv7 shape....", K.int_shape(conv7))
    print("se1 shape....", K.int_shape(se1))
    print("mul_1 shape....", K.int_shape(mul_1))
    #MSE right branch
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
    mul = multiply([conv7, se1])
    print("conv7................", K.int_shape(conv7))
    print("se1..............", K.int_shape(se1))
    print("mul...............", K.int_shape(mul))
    mul_scse = Add()([mul_1, mul])
    print("*********************sen1 shape: ",K.int_shape(se1))

    #Upsample layer 3
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(mul_scse), conv2], name='up8', axis=3)

    up8 = Dropout(DropP)(up8)

    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up8)

    conv8 = bn()(conv8)

    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv8)

    conv8 = bn()(conv8)

    #MSE left branch
    se1 = Conv2D(1, (1,1), activation='softmax')(conv8)#, activation = ?)
    init_cs = K.int_shape(conv8)
    print("Look here..........init_cs", init_cs)
    mul_1 = multiply([conv8, se1])
    print("conv8 shape....", K.int_shape(conv8))
    print("se1 shape....", K.int_shape(se1))
    print("mul_1 shape....", K.int_shape(mul_1))
    # MSE right branch
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
    mul = multiply([conv8, se1])
    print("conv8 shape....", K.int_shape(conv8))
    print("se1 shape....", K.int_shape(se1))
    print("mul shape....", K.int_shape(mul))
    mul_scse = Add()([mul_1, mul])
    print("*********************sen1 shape: ",K.int_shape(se1))

    # Upsample layer 4
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mul_scse), conv1], name='up9', axis=3)

    up9 = Dropout(DropP)(up9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up9)

    conv9 = bn()(conv9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv9)

    conv9 = bn()(conv9)

    se1 = Conv2D(1, (1,1), activation='softmax')(conv9)#, activation = ?)
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
    mul = multiply([conv9, se1])
    mul_scse = Add()([mul_1, mul])

    #Final block
    conv22 = Conv2D(1, (1, 1), activation = 'sigmoid', name='conv22')(mul_scse) #

    conv23 = Conv2D(1, (1, 1), activation = 'sigmoid', name = 'conv23')(mul_scse)

    conv24 = Conv2D(1, (1, 1), activation = 'sigmoid', name = 'conv24')(mul_scse)

    model = Model(inputs = [new_inputs, new_inputs_2], outputs = [conv22, conv23, conv24])

    model.compile(optimizer = Adam(lr = 5e-5), loss = {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss}, metrics = [dice_coef])

    model.summary()

    return model


# X_train=np.load("8_after_sub_144_new.npy")
# X1_train=np.load("8_x_l_144_new.npy")
# y_train=np.load("8_y_l_144_new.npy")
X_test1=np.load("8_after_sub_144_new.npy")
X_test=np.load("8_x_l_144_new.npy")
y_test=np.load("8_y_l_144_new.npy")

X_train=np.load("2_after_sub_144_new.npy")
X1_train=np.load("2_x_l_144_new.npy")
y_train=np.load("2_y_l_144_new.npy")
print("[[[[[[[[[[[[[[[[[[[[[[[[[y_tets]]]]]]]]]]]]]]]]]]]]]]]]]", np.unique(y_test))


X_train1=np.load("10_after_sub_144_new.npy")
X1_train1=np.load("10_x_l_144_new.npy")
y_train1=np.load("10_y_l_144_new.npy")

#changed
X1_train=np.concatenate((X1_train,X1_train1),axis=0)
X1_train1=[]
y_train=np.concatenate((y_train,y_train1),axis=0)
y_train1=[]
X_train=np.concatenate((X_train,X_train1),axis=0)
X_train1=[]

X_train1=np.load("6_after_sub_144_new.npy")
X1_train1=np.load("6_x_l_144_new.npy")
y_train1=np.load("6_y_l_144_new.npy")

X1_train=np.concatenate((X1_train,X1_train1),axis=0)
X1_train1=[]
y_train=np.concatenate((y_train,y_train1),axis=0)
y_train1=[]
X_train=np.concatenate((X_train,X_train1),axis=0)
X_train1=[]
#
X_train1=np.load("4_after_sub_144_new.npy")
X1_train1=np.load("4_x_l_144_new.npy")
y_train1=np.load("4_y_l_144_new.npy")
X1_train=np.concatenate((X1_train,X1_train1),axis=0)
X1_train1=[]
y_train=np.concatenate((y_train,y_train1),axis=0)
y_train1=[]
X_train=np.concatenate((X_train,X_train1),axis=0)
X_train1=[]

print(np.array(X_train).shape,np.array(X1_train).shape,np.array(y_train).shape)

X_train = X_train/1000
print("X_train",np.unique(X_train))

X1_train = X1_train/1000
print("X1_train", np.unique(X1_train))

X_test1 = X_test1/1000
print("X_test1",np.unique(X_test1))

X_test = X_test/1000
print("X_test",np.unique(X_test))


# X_test = X_test/1000
# print("X_test1",np.unique(X_test))
#
# X1_test = X1_test/1000
# print("X_test",np.unique(X1_test))
model = UNet((10, 192, 256),( 192, 256, 1))
# model=load_model('nrna_reverse_reshape_fold4_mulscse.h5', custom_objects={'dice_coef_loss' : dice_coef_loss, 'tf':tf})
print("************Running 300 epochs with softmax***************")
# print("plain_unet_ordering_raunak_parameters_test_2_500.h5")
# model = load_weights('/home/rutu_g/thesis/plain_unet_ordering_75_again.h5')

es = EarlyStopping(monitor = 'val_loss', min_delta = -0.001 , patience = 5)
mc = ModelCheckpoint("server_no_reduction_modified_concurrent_144_100_3_regions_softmax_2intodice4_{epoch:02d}-{val_loss:.2f}.h5", monitor = 'val_loss', period = 100)
# parallel_model = multi_gpu_model(model, gpus = 3)
model.compile(optimizer = Adam(lr = 5e-5), loss =   {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss}, metrics = [dice_coef_loss] )
history = model.fit([X_train, X1_train], [y_train[ :, :, :, 0],y_train[ :, :, :, 1], y_train[ :, :, :, 2]], verbose = 2, batch_size =3, epochs = 300, shuffle= True)#, validation_data = ([X_val1, X_val], [y_val]))#[ :, :, :, 0], y_val[ :, :, :, 1], y_val[ :, :, :, 2]]))
model.save("nrna_reverse_reshape_fold2_mulscse.h5")


y_pred = model.predict([X_test1, X_test])
y_pred = np.array(y_pred)
# y_pred[y_pred <= 0.5 ] = 0
# y_pred[y_pred > 0.5] = 1

print("y_test", y_test.shape)
print("*********************////////////y_pred", y_pred.shape)
print("y_pred unique", np.unique(y_pred))


def dice_coef(y_true, y_pred):
    """ The dice coef is a metric to calculate the similarilty
    (intersection) between the true values and the predictions"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

"""Preprocessing For softmax"""
import random
"""Preprocess for four sigmoids"""
print("y_test", y_test.shape)
print("y_pred", y_pred.shape)
print("y_pred unique", np.unique(y_pred))
pred = y_pred


print("X_test......", X_test.shape)
# y_test = y_test[144:288]
print("-----------------y_pred", y_pred.shape)
d0 = 0
d1 = 0
d2 = 0
d3 = 0
count = 0
for i in range(0, (X_test.shape)[0]):#(X_test[:144].shape)[0]):
    count = count + 1
    dl0 = dice_coef_loss(y_test[i, : ,: , 0].astype(np.float32), y_pred[0, i, : ,: , :])
    dl1 = dice_coef_loss(y_test[i, : ,: , 1].astype(np.float32), y_pred[1, i, : ,: , :])
    dl2 = dice_coef_loss(y_test[i, : ,: , 2].astype(np.float32), y_pred[2, i, : ,: , :])
    # dl3 = dice_coef_loss(y_test[i, : ,: , 3].astype(np.float64), new[i, : ,: , 3])

    d0 = d0 + dl0
    d1 = d1 + dl1
    d2 = d2 + dl2
    # d3 = d3 + dl3

print("count1 ", count)
sess = K.get_session()
print("------------all------dice")
print("nrna_reverse_reshape_fold2_mulscse")
print(sess.run(d0/count))
print(sess.run(d1/count))
print(sess.run(d2/count))

print(np.unique(y_test))
new = new*255
y_test = y_test*255
for i in range(48,100):
    cv2.imwrite("0_4r_pred_%d.png" %i,new[i][ : ,: , 0])
    cv2.imwrite("0_4r_test_%d.png" %i, y_test[i,:,:,0])
    cv2.imwrite("1_4r_pred_%d.png" %i, new[i][ : ,: , 1])
    cv2.imwrite("1_4r_test_%d.png" %i, y_test[i,:,:,1])
    cv2.imwrite("2_4r_pred_%d.png" %i,new[i][ : ,: , 2])
    cv2.imwrite("2_4r_test_%d.png" %i, y_test[i,:,:,2])
    # cv2.imwrite("3_4r_pred_%d.png" %i,y_pred[i][ : ,: , 3])
    # cv2.imwrite("3_4r_test_%d.png" %i, y_test[i,:,:,3])
