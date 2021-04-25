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
import numpy as np
import time
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import *
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
# from tensorflow.keras import backend as K
# from tensorflow.python.keras.layers.normalization import BatchNormalization as bn
# from tensorflow.keras import regularizers
# from tensorflow.keras.preprocessing.image import *
#
# from sklearn.utils import shuffle
#
# from tensorflow.keras.utils import multi_gpu_model

#
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from keras import backend as K
from keras.layers.normalization import BatchNormalization as bn
from keras import regularizers
from keras.preprocessing.image import *
#
# from sklearn.utils import shuffle
from keras.utils import multi_gpu_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
#
# path = '/home/rutu_g/thesis/iSeg-2019-Training/iSeg-2019-Training'
# label_path = '/home/rutu_g/thesis/iSeg-2019-Training/iSeg-2019-Training'
# val_path = '/home/rutu_g/thesis/iSeg-2019-Validation'


path = '/home/rutu/thesis/iSeg-2019-Training'
label_path = '/home/rutu/thesis/iSeg-2019-Training'
val_path = '/home/rutu/thesis/iSeg-2019-Validation'


smooth = 1.
# def dice_coef(y_true, y_pred):
#     """ The dice coef is a metric to calculate the similarilty
#     (intersection) between the true values and the predictions"""
#     y_true_f0 = K.flatten(y_true[:,:,:,0])
#     y_pred_f0 = K.flatten(y_pred[:,:,:,0])
#
#     y_true_f1 = K.flatten(y_true[:,:,:,1])
#     y_pred_f1 = K.flatten(y_pred[:,:,:,1])
#
#     y_true_f2 = K.flatten(y_true[:,:,:,2])
#     y_pred_f2 = K.flatten(y_pred[:,:,:,2])
#
#     intersection0 = K.sum(y_true_f0 * y_pred_f0)
#     intersection1 = K.sum(y_true_f1 * y_pred_f1)
#     intersection2 = K.sum(y_true_f2 * y_pred_f2)
#     # x = ((2. * intersection0 + smooth) / (K.sum(y_true_f0) + K.sum(y_pred_f0) + smooth))
#     # with tf.Session() as sess:
#     #     print(x.eval())
#     #
#     # print("region_1",(2. * intersection1 + smooth) / (K.sum(y_true_f1) + K.sum(y_pred_f1) + smooth))
#     print("region_2",(2. * intersection2 + smooth) / (K.sum(y_true_f2) + K.sum(y_pred_f2) + smooth))
#     return ((2. * intersection0 + smooth) / (K.sum(y_true_f0) + K.sum(y_pred_f0) + smooth)) + ((2. * intersection1 + smooth) / (K.sum(y_true_f1) + K.sum(y_pred_f1) + smooth)) + ((2. * intersection2 + smooth) / (K.sum(y_true_f2) + K.sum(y_pred_f2) + smooth))

def dice_coef(y_true, y_pred):
    """ The dice coef is a metric to calculate the similarilty
    (intersection) between the true values and the predictions"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# def dice_coef(y_true, y_pred):
#
#     y_t0, y_t1, y_t2, y_t3 = tf.split(y_true, 4, axis = -1 )
#     y_p0, y_p1, y_p2, y_p3 =  tf.split(y_pred, 4, axis = -1 )
#     y_true_f0 = K.flatten(y_t0)
#     y_pred_f0 = K.flatten(y_p0)
#
#     intersection0 = K.sum(y_true_f0 * y_pred_f0)
#
#     y_true_f1 = K.flatten(y_t1)
#
#     y_pred_f1 = K.flatten(y_p1)
#
#     intersection1 = K.sum(y_true_f1 * y_pred_f1)
#
#     y_true_f2 = K.flatten(y_t2)
#
#     y_pred_f2 = K.flatten(y_p2)
#
#     intersection2 = K.sum(y_true_f2 * y_pred_f2)
#
#     y_true_f3 = K.flatten(y_t3)
#
#     y_pred_f3 = K.flatten(y_p3)
#
#     intersection3 = K.sum(y_true_f3 * y_pred_f3)
#
#
#     x = (2. * intersection0 + smooth) / (K.sum(y_true_f0) + K.sum(y_pred_f0) + smooth) + (2. * intersection1 + smooth) / (K.sum(y_true_f1) + K.sum(y_pred_f1) + smooth) + (2. * intersection2 + smooth) / (K.sum(y_true_f2) + K.sum(y_pred_f2) + smooth) + (2. * intersection3 + smooth) / (K.sum(y_true_f3) + K.sum(y_pred_f3) + smooth)
#
#
#     return 1-(x/4)

#
# def dice_coef_loss(y_true, y_pred): #Dice 4 still working partially
#
#     nc = 4
#
#     y_t0, y_t1, y_t2, y_t3 = tf.split(y_true, nc, axis = -1 )
#     y_p0, y_p1, y_p2, y_p3 =  tf.split(y_pred, nc, axis = -1 )
#     y_true_f0 = K.flatten(y_t0)
#     y_pred_f0 = K.flatten(y_p0)
#
#     intersection0 = K.sum(y_true_f0 * y_pred_f0)
#
#     y_true_f1 = K.flatten(y_t1)
#
#     y_pred_f1 = K.flatten(y_p1)
#
#     intersection1 = K.sum(y_true_f1 * y_pred_f1)
#
#     y_true_f2 = K.flatten(y_t2)
#
#     y_pred_f2 = K.flatten(y_p2)
#
#     intersection2 = K.sum(y_true_f2 * y_pred_f2)
#
#     y_true_f3 = K.flatten(y_t3)
#
#     y_pred_f3 = K.flatten(y_p3)
#
#     intersection3 = K.sum(y_true_f3 * y_pred_f3)
#
#
#     x =  ((intersection0 + smooth) / (K.sum(y_true_f0) + K.sum(y_pred_f0) + smooth)) + (( intersection1 + smooth) / (K.sum(y_true_f1) + K.sum(y_pred_f1) + smooth)) + (( intersection2 + smooth) / (K.sum(y_true_f2) + K.sum(y_pred_f2) + smooth)) #+ (( intersection3 + smooth) / (K.sum(y_true_f3) + K.sum(y_pred_f3) + smooth))
#
#     return (1 - (2. * x/nc))

    # return -(2. * (x/nc))

####Dice 5#####
# def dice_coef_loss(y_t, y_p):
#
#     y_true_f = K.flatten(y_t)
#     y_pred_f = K.flatten(y_p)
#
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
# def dice_coef(y_true, y_pred):
#     sum_over_batch = 0
#     batch_size = 8
#     for i in range(batch_size):
#         nc = 4
#
#         y_t0, y_t1, y_t2, y_t3 = tf.split(y_true[i], nc, axis = -1 )
#         y_p0, y_p1, y_p2, y_p3 =  tf.split(y_pred[i], nc, axis = -1 )
#
#         x0 = dice_coef_loss(y_t0, y_p0)
#         x1 = dice_coef_loss(y_t1, y_p1)
#         x2 = dice_coef_loss(y_t2, y_p2)
#         x3 = dice_coef_loss(y_t3, y_p3)
#
#         x = -2.*(x0 +x1 + x2 + x3)/4
#         sum_over_batch = sum_over_batch + x
#     return sum_over_batch/8
####Dice 5#####

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

# #
# # """Region 1"""
# train_filenames = glob.glob('/home/rutu_g/thesis/iSeg-2019-Training/iSeg-2019-Training/*T1.img')
# val_filenames = glob.glob('/home/rutu_g/thesis/iSeg-2019-Validation/*T1.img')

train_filenames = glob.glob('/home/rutu/thesis/iSeg-2019-Training/*T1.img')
val_filenames = glob.glob('/home/rutu/thesis/iSeg-2019-Validation/*T1.img')

len(val_filenames)#np.save('np_train_T1_chunked', np_train_T1_chunked)
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


# #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# def difference(X_train):
#
#
#     #print("X_train", X_train.shape)
#
#     n_slices = 5
#     xw = []
#     x_t = X_train.shape
#     # X_train = np.reshape(X_train, (x_t[0], x_t[1], x_t[2]*x_t[3]))
#     for i in range(n_slices, (X_train.shape)[0]-n_slices):
#         # print("&&&&&&&X_train", np.unique(X_train))
#         # print("%%%%%%%%%%%X_train[i, :, :, :]", np.unique(X_train[i, :, :, :]))
#
#         x = np.absolute(np.subtract(X_train[i-n_slices: i+n_slices] , X_train[i, :, :, :]))
#         print("x....",x.shape)
#         # print("x?????????????/", np.unique(x[20, :, :, : ]))
#         xw.append(x)
#     #print("xw: ",(np.array(xw)).shape)
#     xw = np.array(xw)
#     #print("xw1",np.unique(xw[1,11, :, :, :]))
#     return xw
#
# # xw = difference(X_train)
#
# #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# #x1 = xw[10, :, :, :, :]
#
# def ordered(xw):
#     after_sub_ordered = []
#     before_sub_ordered = []
#     for i in range((xw.shape)[0]):
#         x1 = xw[i, :, :, :, :]
#
#         d = {}
#         for i in range((x1.shape)[0]):
#             d[i] = np.sum(x1[i, :, :, :])
#             # print("index", i)
#             # print("dictionary", d)
#
#         sorted_d = sorted(d.items(), key=lambda x: x[1])
#         #print("sorted_d", sorted_d)
#         a = []
#         b = []
#         for s in sorted_d:
#             #print("s", s)
#             a.append(x1[s[0]])
#             #b.append(X_train[s[0]])
#
#         a = np.array(a)
#         #b = np.array(b)
#         after_sub_ordered.append(a)
#         before_sub_ordered.append(b)
#         # print(after_sub_ordered.type(), before_sub_ordered.type())
#
#         # print("after_sub_ordered", after_sub_ordered.shape)
#         # print("before_sub_ordered", before_sub_ordered.shape)
#     after_sub_ordered = np.array(after_sub_ordered)
#     #before_sub_ordered = np.array(before_sub_ordered)
#     return after_sub_ordered#, before_sub_ordered
#
# # a, b = ordered(xw)
# # print("##### a:", a.shape)
# # print("##### b:", b.shape)
# # """ 5 patients """
# # # all_after_sub
# # # for i in range():
#
# difference_all = []
# ranges = [0, 2, 4, 6, 8, 10]
# print(ranges)
# #
# one_hot = np.load("/home/rutu/thesis/one_hot_encoded_newer.npy")
# # one_hot = np.load("/home/rutu_g/thesis/volume_144/server_one_hot_encoded.npy") #newest, background is white
# #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# for j in range(len(ranges)-1):
#     print("%%%%%%%%%%%%%%%%%%%%%%%%%")
#     print(ranges[j], ranges[j+1])
#     print("%%%%%%%%%%%%%%%%%%%%%%%%%")
#     five_after_sub = []
#     five_before_sub = []
#     X_l = []
#     y_l = []
#     for i in range(ranges[j], ranges[j+1]):
#         X =  np_train_T1[i, :, : ,: , :]
#
#         # print("ORRRIGINALLLL X", np.unique(X))
#
#
#         X_l.append(X)
#         s = X.shape
#         zeroes_f = np.zeros((5, s[1], s[2], s[3] ))
#         X = np.concatenate((X, zeroes_f), axis = 0)
#         X = np.concatenate((zeroes_f, X), axis = 0)
#
#
#         y = one_hot[i, :, : ,: , :]
#
#         print("y after transpose", y.shape)
#         y_l.append(y)
#         xw = difference(X)
#
#     # difference_all = np.array(difference_all)
#     # print("difference_all: ",difference_all.shape)
#         after_sub_ordered  = ordered(xw)
#         # print("after_sub_ordered",after_sub_ordered.shape)
#         # print("before_sub_ordered", before_sub_ordered.shape)
#         five_after_sub.append(after_sub_ordered)
#         # five_before_sub.append(before_sub_ordered)
#
#     #print(after_sub_ordered.type(), before_sub_ordered.type())
#     five_after_sub = np.array(five_after_sub)
#     # five_before_sub = np.array(five_before_sub)
#     X_l = np.array(X_l)
#     y_l = np.array(y_l)
#     print("five_after_sub", five_after_sub.shape)
#     # print("five_before_sub", five_before_sub.shape)
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>five_after_sub_unique", np.unique(five_after_sub))
#     a = five_after_sub.shape
#     # b = five_before_sub.shape
#     five_after_sub = np.reshape(five_after_sub, (a[0]* a[1], a[2], a[3], a[4]*a[5]))
#     # five_before_sub = np.reshape(five_before_sub, (b[0]* b[1], b[2], b[3], b[4]*b[5]))
#     c = X_l.shape
#     d = y_l.shape
#     print("c: ", c)
#     print("d: ", d)
#     X_l = np.reshape(X_l, (c[0]*c[1], c[2], c[3], c[4]))
#     y_l = np.reshape(y_l, (d[0]*d[1], d[2], d[3], d[4]))
#
#     from sklearn.utils import shuffle
#     # five_after_sub, X_l, y_l = shuffle(five_after_sub, X_l, y_l)
#
#     #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#     # five_after_sub = five_after_sub/1000
#     # # five_before_sub = five_before_sub/1000
#     # X_l = X_l /1000
#     # y_l = y_l/1000
#
#     np.save("%d_after_sub_192_new.npy" %ranges[j+1], five_after_sub )
#     print("five_after_sub",np.unique(five_after_sub))
#     # np.save("%d_before_sub_256" %ranges[j+1], five_before_sub)
#     np.save("%d_x_l_256_new.npy" %ranges[j+1], X_l)
#     print("X_l",np.unique(X_l))
#     np.save("%d_y_l_256_new.npy" %ranges[j+1], y_l)
#
#     print("a", five_after_sub.shape)
#     # print("b", five_before_sub.shape)
#     print("c", X_l.shape)
#     print("d", y_l.shape)

                                                                                # x1 = np.load("2_after_sub_256_win.npy")
                                                                                # x2 = np.load("2_x_l_256_win.npy")
                                                                                # y = np.load("2_y_l_256_win.npy")
                                                                                # x1=x1*255
                                                                                # x2 = x2*255
                                                                                # y = y*255
                                                                                # # for i in range(0,10):
                                                                                # #     # cv2.imwrite("n_pred_%d.png" %i,y_pred[i][ : ,: , 0])
                                                                                # #     cv2.imwrite("x1_%d.png" %i, x1[100,i,:,:])
                                                                                # y_l = np.load("/home/rutu_g/thesis/volume_144/server_one_hot_encoded.npy")
                                                                                # d = y_l.shape
                                                                                # y_l = np.reshape(y_l, (d[0]*d[1], d[2], d[3], d[4]))
                                                                                #
                                                                                # y_l = y_l*255
                                                                                # for i in range(0,144):
                                                                                #     cv2.imwrite("x2_%d.png" %i, x2[i ,:,:,:])
                                                                                #     cv2.imwrite("y_%d.png" %i, y_l[i, :,:, 0])



# x = np.load("10_after_sub_256_new.npy")
# x1 = np.load("10_x_l_256_new.npy")
# y = np.load("10_y_l_256_new.npy")
#
# print("x",np.unique(x))
# print("x1",np.unique(x1))
# print("y",np.unique(y))
# y = y*255
# import cv2
# for i in range(20, 30):
#     cv2.imwrite("%d_y_trial.png" %i, y[i, :, :, 0])
#     cv2.imwrite("%d_x_trial.png" %i, x[i, :, :, 0])
#     cv2.imwrite("%d_x1_trial.png" %i, x1[i, :, :, 0])

#
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def UNet(input_shape_1, input_shape_2):
    new_inputs = Input(input_shape_1)
    new_inputs_2 = Input(input_shape_2)

    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size = 3

    # se1 = GlobalAveragePooling2D(data_format = 'channels_first')(new_inputs)
    # print("new GAP", se1.shape)
    # se1 = Dense(5, activation = 'relu')(se1)
    # print("new D", se1.shape)
    # se1_huge = Dense(10, activation = 'softmax')(se1)
    # print("se1_huge", se1_huge.shape)

    new_inputs_permuted = Permute((2, 3, 1))(new_inputs)
    print("new_inputs_permuted", new_inputs_permuted.shape)

    se1 = Conv2D(1, (1,1), activation = 'softmax')(new_inputs_permuted)#, activation = ?)#
    print("se1 shape....", K.int_shape(se1))                                             #
    mul_1 = multiply([new_inputs_permuted, se1], name = 'mul_1')                         #
    print("mul_1 shape....", K.int_shape(mul_1))                                         #

    init_cs = K.int_shape(new_inputs_permuted)
    print("Look here..........init_cs", init_cs)
    #
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

    # new_inputs_p = Permute( (3, 2, 1))(new_inputs)
    new_inputs_p = Permute( (2, 3, 1))(new_inputs)                                     #

    print("************new_inputs", new_inputs_p.shape)
    print("************se1", se1.shape)


    # mul = multiply([new_inputs_p, se1_huge])
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



    combine = concatenate([pwc, new_inputs_2], name = 'combine')
    sess = K.get_session()
    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(combine)

    conv1 = bn()(conv1)

    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='softmax', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv1)

    conv1 = bn()(conv1)

    se1 = Conv2D(1, (1,1), activation = 'softmax', name = "se1")(conv1)#, activation = ?)
    print("se1 shape....", K.int_shape(se1))
    mul_1 = multiply([conv1, se1], name = 'mul_11')
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

    se1 = DepthwiseConv2D((1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)

    print("************se1", se1.shape)

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

    se1 = Conv2D(1, (1,1), activation = 'softmax', name = "se2")(conv2)#, activation = ?)
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
    se1 = DepthwiseConv2D((1, int(se1_r_shape[2])), activation = 'softmax')(se1_r)
    print("W*1 filter", se1.shape)

    print("************se1", se1.shape)




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

    se1 = Conv2D(1, (1,1), activation = 'softmax', name = "se3")(conv3)#, activation = ?)
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

    se1 = Conv2D(1, (1,1), activation='softmax', name = "se4")(conv4)#, activation = ?)
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

    se1 = Conv2D(1, (1,1), activation = 'softmax', name = "se5")(conv6)#, activation = ?)
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

    se1 = Conv2D(1, (1,1), activation='softmax', name = "se6")(conv7)#, activation = ?)
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

    se1 = Conv2D(1, (1,1), activation='softmax', name = "se7")(conv8)#, activation = ?)
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

    se1 = Conv2D(1, (1,1), activation='softmax', name = "se8")(conv9)#, activation = ?)
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
    model = Model(inputs = [new_inputs, new_inputs_2], outputs = [conv22, conv23, conv24])
    #model = Model(inputs = new_inputs, outputs = conv22)


    # model.compile(optimizer = Adam(lr = 5e-5), loss = {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss, 'conv25': dice_coef_loss}, metrics = [dice_coef])

    model.compile(optimizer = Adam(lr = 5e-5), loss = {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss}, metrics = [dice_coef])

    model.summary()

    # if(pretrained_weights):
    #   model.load_weights(pretrained_weights)

    return model
    # print(model.input.shape)


# X_train=np.load("8_after_sub_256_new.npy")
# X1_train=np.load("8_x_l_256_new.npy")
# y_train=np.load("8_y_l_256_new.npy")
X_test1=np.load("8_after_sub_192_new.npy")
X_test=np.load("8_x_l_192_new.npy")
y_test=np.load("8_y_l_192_new.npy")

X_train=np.load("2_after_sub_192_new.npy")
# X_train1 = X_trainall[0:144]
# X_test1 = X_trainall[144:288]
X1_train=np.load("2_x_l_192_new.npy")
# X1_train1 = X1_trainall[0:144]
# X_test = X1_trainall[144:288]
y_train=np.load("2_y_l_192_new.npy")
# y_train1 = y_trainall[0:144]
# y_test = y_trainall[144:288]
print("[[[[[[[[[[[[[[[[[[[[[[[[[y_tets]]]]]]]]]]]]]]]]]]]]]]]]]", np.unique(y_test))


# X_test_1=np.concatenate((X_train,X_train1),axis=0)
# X_train1=[]
# X1_test_1=np.concatenate((X1_train,X1_train1),axis=0)
# X1_train1=[]
# y_test_1=np.concatenate((y_train,y_train1),axis=0)
# y_train1=[]
# X1_train=np.concatenate((X1_train,X1_train1),axis=0)
# X1_train1=[]
# y_train=np.concatenate((y_train,y_train1),axis=0)
# y_train1=[]
# X_train=np.concatenate((X_train,X_train1),axis=0)
# X_train1=[]

# print(np.array(X_train).shape,np.array(X1_train).shape,np.array(y_train).shape)

# print("X_test1 shapeeeeee", X_test1.shape)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

X_train1=np.load("6_after_sub_192_new.npy")
# X_test_2 = X_train[0:144]
# X_train = X_train[144:288]
X1_train1=np.load("6_x_l_192_new.npy")
# X1_test_2 = X1_train[0:144]
# X1_train = X1_train[144:288]
y_train1=np.load("6_y_l_192_new.npy")
# y_test_2 = y_train[0:144]
# y_train = y_train[144:288]

#changed
X1_train=np.concatenate((X1_train,X1_train1),axis=0)
X1_train1=[]
y_train=np.concatenate((y_train,y_train1),axis=0)
y_train1=[]
X_train=np.concatenate((X_train,X_train1),axis=0)
X_train1=[]

# X_test = np.concatenate((X_test_1, X_test_2))
# X1_test = np.concatenate((X1_test_1, X1_test_2))
# y_test = np.concatenate((y_test_1, y_test_2))

X_train1=np.load("10_after_sub_192_new.npy")
X1_train1=np.load("10_x_l_192_new.npy")
y_train1=np.load("10_y_l_192_new.npy")

X1_train=np.concatenate((X1_train,X1_train1),axis=0)
X1_train1=[]
y_train=np.concatenate((y_train,y_train1),axis=0)
y_train1=[]
X_train=np.concatenate((X_train,X_train1),axis=0)
X_train1=[]
#
X_train1=np.load("4_after_sub_192_new.npy")
# X_train1 = X_trainall[144:288]
# X_val1 = X_trainall[0:144]
X1_train1=np.load("4_x_l_192_new.npy")
# X1_train1 = X1_trainall[144:288]
# X_val = X1_trainall[0:144]
y_train1=np.load("4_y_l_192_new.npy")
# y_train1 = y_trainall[144:288]
# y_val = y_trainall[0:144]
X1_train=np.concatenate((X1_train,X1_train1),axis=0)
X1_train1=[]
y_train=np.concatenate((y_train,y_train1),axis=0)
y_train1=[]
X_train=np.concatenate((X_train,X_train1),axis=0)
X_train1=[]
# X_train1=np.load("2_after_sub_144_new.npy")
# X_train1 = X_trainall[144:288]
# X_val1 = X_trainall[0:144]
# X1_train1=np.load("2_x_l_144_new.npy")
# X1_train1 = X1_trainall[144:288]
# X_val = X1_trainall[0:144]
# y_train1=np.load("2_y_l_144_new.npy")
# y_train1 = y_trainall[144:288]
# y_val = y_trainall[0:144]
# print(np.unique(X1_train1))
# print(np.unique(y_train1))
# X1_train=np.concatenate((X1_train,X1_train1),axis=0)
# X1_train1=[]
# y_train=np.concatenate((y_train,y_train1),axis=0)
# y_train1=[]
# X_train=np.concatenate((X_train,X_train1),axis=0)
# X_train1=[]

print(np.array(X_train).shape,np.array(X1_train).shape,np.array(y_train).shape)



X_train = X_train/1000
print("X_train",np.unique(X_train))
#
# X_val1 = X_val1/1000
# print("X_val1",np.unique(X_val1))

X1_train = X1_train/1000
print("X1_train", np.unique(X1_train))

# X_val = X_val/1000
# print("X_val",np.unique(X_val))

X_test1 = X_test1/1000
print("X_test1",np.unique(X_test1))

X_test = X_test/1000
print("X_test",np.unique(X_test))


# X_test = X_test/1000
# print("X_test1",np.unique(X_test))
#
# X1_test = X1_test/1000
# print("X_test",np.unique(X1_test))
model = UNet((10, 144, 256),( 144, 256, 1))
# model=load_model('nrna_reverse_reshape_fold4_mulscse.h5', custom_objects={'dice_coef_loss' : dice_coef_loss, 'tf':tf})
print("************Running 300 epochs with softmax***************")
# print("plain_unet_ordering_raunak_parameters_test_2_500.h5")
# model = load_weights('/home/rutu_g/thesis/plain_unet_ordering_75_again.h5')

es = EarlyStopping(monitor = 'val_loss', min_delta = -0.001 , patience = 5)
mc = ModelCheckpoint("server_no_reduction_modified_concurrent_144_100_3_regions_softmax_2intodice4_{epoch:02d}-{val_loss:.2f}.h5", monitor = 'val_loss', period = 100)
# model = multi_gpu_model(model, gpus = 2)
time_callback = TimeHistory()
model.compile(optimizer = Adam(lr = 5e-5), loss =   {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss}, metrics = [dice_coef_loss] )
history = model.fit([X_train, X1_train], [y_train[ :, :, :, 0],y_train[ :, :, :, 1], y_train[ :, :, :, 2]], verbose = 2, callbacks = [time_callback], batch_size =6, epochs = 300, shuffle= True)#, validation_data = ([X_val1, X_val], [y_val]))#[ :, :, :, 0], y_val[ :, :, :, 1], y_val[ :, :, :, 2]]))
model.save("nrna_reverse_reshape_192_fold2_mulscse.h5")
# history = parallel_model.fit([X_test1[:144], X_test[:144]], [y_test[ :144, :, :, 0],y_test[ :144, :, :, 1], y_test[ :144, :, :, 2]], verbose = 2, batch_size =8, epochs = 300, shuffle= True)#, validation_data = ([X_val1, X_val], [y_val]))#[ :, :, :, 0], y_val[ :, :, :, 1], y_val[ :, :, :, 2]]))

times = time_callback.times
print(times)
# # # # #, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss
# # #
# #
# # print("Looook hereeeee", y_train.shape)
# # print("Looook hereeeee",np.unique(y_train))
# model.save("old_144_fold1_SEPWC_8subs.h5")
# model.save("nrna_fold1.h5")
#

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig('no_reduction_modified_concurrent_144_200_joint_dice_4.png')


# X1 = X_test.shape
# ys = y_test.shape
# print("X1", X1)
# print("ys", ys)
# X_test = np.reshape(X_test, (X1[0]*X1[1], X1[2], X1[3], X1[4]))
# y_test = np.reshape(y_test, (ys[0]*ys[1], ys[2], ys[3], ys[4]))


y_pred = model.predict([X_test1, X_test])
y_pred = np.array(y_pred)
# y_pred[y_pred <= 0.5 ] = 0
# y_pred[y_pred > 0.5] = 1

print("y_test", y_test.shape)
print("*********************////////////y_pred", y_pred.shape)
print("y_pred unique", np.unique(y_pred))

# d0 = dice_coef_loss(y_test[0].astype(np.float32), y_pred[0])
# sess = K.get_session()
# print(sess.run(d0))
#
# d1 = dice_coef_loss(y_test[1].astype(np.float32), y_pred[1])
#
# print(sess.run(d1))
#
# d2 = dice_coef_loss(y_test[2].astype(np.float32), y_pred[2])
#
# print(sess.run(d2))

# y_pred = y_pred*255
# print("0000000000000000", np.unique(y_pred))
# y_pred[y_pred<255]=0
# print("0000000000000000", np.unique(y_pred))
#
# y_test = y_test*255
# print("0000000000000000", np.unique(y_test))

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
# print(np.unique(y_pred))
# print("y_pred.....",y_pred.shape)
# new = np.zeros(y_pred.shape)
# pred = y_pred
# for i in range((pred.shape)[0]):
#     for j in range((pred.shape)[1]):
#         for k in range((pred.shape)[2]):
#
#             if pred[i, j, k, 0] > pred[i, j, k, 1] and pred[i, j, k, 0] > pred[i, j, k, 2]:
#                 index = 0
#             # elif pred[0, i, j, k, 0] > pred[1, i, j, k, 0] and pred[0, i, j, k, 0] == pred[2, i, j, k, 0]:
#             #     index = random.choice([0, 2])
#             elif pred[i, j, k, 1] > pred[i, j, k, 0] and pred[i, j, k, 1] > pred[i, j, k, 2]:
#                 index = 1
#             # elif pred[1, i, j, k, 0] > pred[0, i, j, k, 0] and pred[1, i, j, k, 0] == pred[2, i, j, k, 0]:
#             #     index = random.choice([1, 2])
#             elif pred[i, j, k, 2] > pred[i, j, k, 0] and pred[i, j, k, 2] > pred[i, j, k, 1]:
#                 index = 2
#             # elif pred[2, i, j, k, 0] > pred[0, i, j, k, 0] and pred[2, i, j, k, 0] == pred[i, j, k, 1]:
#             #     index = random.choice([2, 1])
#             # else:
#             #     index = random.choice([0, 1, 2])
#             #     count = count +1
#             # index += 1
#             # if pred[0, i, j, k, 0 ] == pred[1, i, j, k, 0 ] == pred[2, i, j, k, 0 ] == 0:
#             #     new[i, j, k, 0] = 0
#             # else:
#             new[i, j, k, index] = 1
import random
"""Preprocess for four sigmoids"""
print("y_test", y_test.shape)
print("y_pred", y_pred.shape)
print("y_pred unique", np.unique(y_pred))
pred = y_pred

# new = np.zeros(( 144, 192, 256, 3))
# count = 0
# for i in range(144):
#     for j in range(192):
#         for k in range(256):
#             if pred[0, i, j, k, 0] < 0.5 and pred[1, i, j, k, 0] < 0.5 and pred[2, i, j, k, 0] < 0.5:
#                 new[i, j, k, 0] = 0
#             else:
#                 if pred[0, i, j, k, 0] > pred[1, i, j, k, 0] and pred[0, i, j, k, 0] > pred[2, i, j, k, 0]:
#                     index = 0
#                 elif pred[0, i, j, k, 0] > pred[1, i, j, k, 0] and pred[0, i, j, k, 0] == pred[2, i, j, k, 0]:
#                     index = random.choice([0, 2])
#                 elif pred[1, i, j, k, 0] > pred[0, i, j, k, 0] and pred[1, i, j, k, 0] > pred[2, i, j, k, 0]:
#                     index = 1
#                 elif pred[1, i, j, k, 0] > pred[0, i, j, k, 0] and pred[1, i, j, k, 0] == pred[2, i, j, k, 0]:
#                     index = random.choice([1, 2])
#                 elif pred[2, i, j, k, 0] > pred[0, i, j, k, 0] and pred[2, i, j, k, 0] > pred[1, i, j, k, 0]:
#                     index = 2
#                 elif pred[2, i, j, k, 0] > pred[0, i, j, k, 0] and pred[2, i, j, k, 0] == pred[1, i, j, k, 0]:
#                     index = random.choice([2, 1])
#                 else:
#                     index = random.choice([0, 1, 2])
#                     count = count +1
#                 # index += 1
#                 # if pred[0, i, j, k, 0 ] == pred[1, i, j, k, 0 ] == pred[2, i, j, k, 0 ] == 0:
#                 #     new[i, j, k, 0] = 0
#                 # else:
#                 new[i, j, k, index] = 1
# print("Number of ties", count)

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
print("nrna_reverse_reshape_192_fold2_mulscse")
print(sess.run(d0/count))
print(sess.run(d1/count))
print(sess.run(d2/count))
# print(sess.run(d3/count))

int_model = Model(inputs = model.input, outputs = model.get_layer('pwc').output)

intermediate_output = int_model.predict([X_train, X1_train])
intermediate_output = np.array(intermediate_output)
print(intermediate_output.shape)

# intermediate_output=(intermediate_output-np.amin(intermediate_output))/((np.amax(intermediate_output))-(np.amin(intermediate_output)))
print(np.unique(intermediate_output, return_counts = True))
intermediate_output = intermediate_output * 2550
print(len(intermediate_output))
for i in range(len(intermediate_output)):
    cv2.imwrite("%d_intermediate_output.png" %i, intermediate_output[i])


# a = np.zeros((144, 192, 3))
# print("-------n1 unique", np.unique(n1))
# print("-------n2 unique", np.unique(n2))
# for j in range(144):
#     for k in range(192):
#         if n1[j, k] == 0 and n2[j, k] >0:
#             a[j, k, 0]  = 2
#             a[j, k, 1]  = 2
#             a[j, k, 2]  = 2
#         # if n2[j, k] == 0 and n1[j, k]>0:
#         #     a[j, k, 0]  = 3
#         #     a[j, k, 1]  = 3
#         #     a[j, k, 2]  = 3

# a = a*255
# print(np.unique(a, return_counts=True))
# cv2.imwrite("paper_image_confusion_matrix_false_negative_no_reduction_144_r2.png", a)
