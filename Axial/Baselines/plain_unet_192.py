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
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.normalization import BatchNormalization as bn
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import *
import pickle
from sklearn.utils import shuffle

from tensorflow.keras.utils import multi_gpu_model

# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
# from keras import backend as K
# from keras.layers.normalization import BatchNormalization as bn
# from keras import regularizers
# from keras.preprocessing.image import *
#
# from sklearn.utils import shuffle
# from keras.utils import multi_gpu_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'


path = '/home/rutu_g/thesis/iSeg-2019-Training/iSeg-2019-Training'
label_path = '/home/rutu_g/thesis/iSeg-2019-Training/iSeg-2019-Training'
val_path = '/home/rutu_g/thesis/iSeg-2019-Validation'

#
# path = '/home/rutu/thesis/iSeg-2019-Training'
# label_path = '/home/rutu/thesis/iSeg-2019-Training'
# val_path = '/home/rutu/thesis/iSeg-2019-Validation'


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


train_filenames = glob.glob('/home/rutu_g/thesis/iSeg-2019-Training/iSeg-2019-Training/*T1.img')
val_filenames = glob.glob('/home/rutu_g/thesis/iSeg-2019-Validation/*T1.img')

# train_filenames = glob.glob('/home/rutu/thesis/iSeg-2019-Training/*T1.img')
# val_filenames = glob.glob('/home/rutu/thesis/iSeg-2019-Validation/*T1.img')
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

one_hot = np.load('/home/rutu_g/thesis/one_hot_encoded_newer.npy')
# one_hot = np.load('/home/rutu/thesis/one_hot_encoded_newer.npy')

X_l = []
y_l = []
for i in range((np_train_T1.shape)[0]):
    X = np_train_T1[i, :, : ,: , :]
    X = np.transpose(X, (1, 0, 2, 3))
    print("X after transpose", X.shape)
    X_l.append(X)
    y = one_hot[i, :, : ,: , :]
    y = np.transpose(y, (1, 0, 2, 3))
    print("y after transpose", y.shape)
    y_l.append(y)
X_l = np.array(X_l)
y_l = np.array(y_l)

X_l = X_l/1000

# X_train = np.concatenate((X_l[0:2], X_l[3:9]))
# X_test = np.concatenate((X_l[2], X_l[9]))
# print("np_train_T1 unique", np.unique(X_l[0]))
# print("y one hot uni", np.unique(y_l[0]))
# print("((((((((((((((((((((np_train_T1", X_l.shape)
# print("X_train", X_train.shape)
# y_train  = np.concatenate((y_l[0:2], y_l[3:9]))
# y_test = np.concatenate((y_l[2], y_l[9]))

# X_train = X_l[5:10]
# X_test = X_l[0:5]
X_train = np.concatenate((X_l[:4], X_l[6:]))
X_test = np.concatenate((X_l[4:5], X_l[5:6]))

print("np_train_T1 unique", np.unique(X_l[0]))
print("y one hot uni", np.unique(y_l[0]))
print("((((((((((((((((((((np_train_T1", X_l.shape)
print("X_train", X_train.shape)
print("y_l_4", y_l[4].shape)
y_train  = np.concatenate((y_l[:4], y_l[6:]))
y_test = np.concatenate((y_l[4:5], y_l[5:6]))
print("y", y_l.shape)
X1 = X_train.shape
ys = y_train.shape
yt = y_test.shape
print("X_train", X1)
print("y_train", ys)
print("y_test", yt)
X_train = np.reshape(X_train, (X1[0]*X1[1], X1[2], X1[3], X1[4]))
y_train = np.reshape(y_train, (ys[0]*ys[1], ys[2], ys[3], ys[4]))
X1 = X_test.shape
ys = y_test.shape
X_test = np.reshape(X_test, (X1[0]*X1[1], X1[2], X1[3], X1[4]))
y_test = np.reshape(y_test, (ys[0]*ys[1], ys[2], ys[3], ys[4]))



"""Normalize!!!!"""

def UNet(input_shape_1):
    new_inputs = Input(input_shape_1)
    # new_inputs_2 = Input(input_shape_2)

    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size = 3



    sess = K.get_session()

    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(new_inputs)

    conv1 = bn()(conv1)

    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv1)

    conv1 = bn()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    pool1 = Dropout(DropP)(pool1)

    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool1)

    conv2 = bn()(conv2)

    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv2)

    conv2 = bn()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    pool2 = Dropout(DropP)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool2)

    conv3 = bn()(conv3)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv3)

    conv3 = bn()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    pool3 = Dropout(DropP)(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(pool3)
    conv4 = bn()(conv4)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv4)

    conv4 = bn()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

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

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv6)

    conv6 = bn()(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], name='up7', axis=3)

    up7 = Dropout(DropP)(up7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up7)

    conv7 = bn()(conv7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv7)

    conv7 = bn()(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], name='up8', axis=3)

    up8 = Dropout(DropP)(up8)

    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up8)

    conv8 = bn()(conv8)

    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv8)

    conv8 = bn()(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], name='up9', axis=3)

    up9 = Dropout(DropP)(up9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(up9)

    conv9 = bn()(conv9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda))(conv9)

    conv9 = bn()(conv9)

    conv22 = Conv2D(1, (1, 1), activation='sigmoid', name='conv22')(conv9) #

    # print("conv22: ", conv22.type())

    conv23 = Conv2D(1, (1, 1), activation = 'sigmoid', name = 'conv23')(conv9)

    conv24 = Conv2D(1, (1, 1), activation = 'sigmoid', name = 'conv24')(conv9)

    # conv25 = Conv2D(1, (1, 1), activation = 'softmax', name = 'conv25')(lr18)
    #
    model = Model(inputs = [new_inputs ], outputs = [conv22, conv23, conv24])
    #model = Model(inputs = new_inputs, outputs = conv22)


    # model.compile(optimizer = Adam(lr = 5e-5), loss = {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss, 'conv25': dice_coef_loss}, metrics = [dice_coef])

    model.compile(optimizer = Adam(lr = 5e-5), loss =  {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss}, metrics = [dice_coef] )

    model.summary()

    # if(pretrained_weights):
    #   model.load_weights(pretrained_weights)

    return model



# model = load_model("experiment_2_unet_256_200epochs.h5", custom_objects={'dice_coef_loss':dice_coef_loss,'dice_coef' : dice_coef})

model = UNet(( 144, 256, 1))
# model=load_model('plain_unet_192_fold1.h5', custom_objects={'dice_coef_loss':dice_coef_loss,'dice_coef' : dice_coef})
# print("plain_unet_ordering_raunak_parameters_test_2_500.h5")
# model = load_weights('/home/rutu_g/thesis/plain_unet_ordering_75_again.h5')


parallel_model = multi_gpu_model(model, gpus = 2)
parallel_model.compile(optimizer = Adam(lr = 5e-5), loss =  {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss}, metrics = [dice_coef] )


model= load_model("old_unet_192_fold3.h5", custom_objects={'dice_coef_loss':dice_coef_loss,'dice_coef' : dice_coef})
# model.compile(optimizer = Adam(lr = 5e-5), loss =  {'conv22':dice_coef_loss, 'conv23': dice_coef_loss, 'conv24': dice_coef_loss}, metrics = [dice_coef] )

history = parallel_model.fit([X_train], [y_train[:, :,:,0],y_train[ :,:,:,1],y_train[ :,:,:,2]], batch_size=16, epochs=1, verbose = 2, shuffle=True)#, validation_data = (X_test[192:384], [y_test[192:384, :, :, 0], y_test[192:384, :, :, 1], y_test[192:384, :, :, 2]]))
# model.save("experiment_2_unet_256_300epochs_8_3gpus_attempt3_diff_test.h5")
# model.save("old_unet_192_fold3.h5")
# parallel_model = multi_gpu_model(model, gpus = 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('plain_unet_192_fold1.png')

X1 = X_test.shape
ys = y_test.shape
print("X1", X1)
print("ys", ys)
# X_test = np.reshape(X_test, (X1[0]*X1[1], X1[2], X1[3], X1[4]))
# y_test = np.reshape(y_test, (ys[0]*ys[1], ys[2], ys[3], ys[4]))

import time
start = time.time()
y_pred = model.predict([X_test])
end = time.time()
print("Inference time", end - start)
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
for i in range(0, 384):
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
print(sess.run(d1/count))
print(sess.run(d2/count))
print(sess.run(d3/count))


print(y_pred.shape)
print(y_test.shape)
for i in range(len(y_test)):
    print("printing")
    cv2.imwrite("plain_%d_pred_192_r2.png" %i, y_pred[2][i])
    cv2.imwrite("plain_%d_test_192_r2.png" %i, y_test[i,:,:,2])


final_dice_1=0
final_dice_2=0
final_dice_3=0

count=0
# for i in range(0,len(X_test)):
for i in range(0, 192):
	count=count+1
	final_dice_1=final_dice_1+dice_coef_loss(y_pred[0][i],y_test[i,:,:,0].astype(np.float32))
	final_dice_2=final_dice_2+dice_coef_loss(y_pred[1][i],y_test[i,:,:,1].astype(np.float32))
	final_dice_3=final_dice_3+dice_coef_loss(y_pred[2][i],y_test[i,:,:,2].astype(np.float32))
print("count2", count)
final_dice_1=final_dice_1/count
final_dice_2=final_dice_2/count
final_dice_3=final_dice_3/count
print(final_dice_1,final_dice_2,final_dice_3)
sess = K.get_session()
print(sess.run(final_dice_1))
print(sess.run(final_dice_2))
print(sess.run(final_dice_3))


final_dice_1=0
final_dice_2=0
final_dice_3=0

count=0
# for i in range(0,len(X_test)):
for i in range(0, 192):
	count=count+1
	final_dice_1=final_dice_1+dice_coef_loss(y_pred[0][i],y_test[i,:,:,0].astype(np.float32))
	final_dice_2=final_dice_2+dice_coef_loss(y_pred[1][i],y_test[i,:,:,1].astype(np.float32))
	final_dice_3=final_dice_3+dice_coef_loss(y_pred[2][i],y_test[i,:,:,2].astype(np.float32))
print("count2", count)
final_dice_1=final_dice_1/count
final_dice_2=final_dice_2/count
final_dice_3=final_dice_3/count
print(final_dice_1,final_dice_2,final_dice_3)
sess = K.get_session()
print(sess.run(final_dice_1))
print(sess.run(final_dice_2))
print(sess.run(final_dice_3))

import cv2
y_pred = y_pred*255
y_test = y_test*255
# for i in range(0, 256):
#     cv2.imwrite("plain_%d_pred_100_r2.png" %i, y_pred[1][i])
#     cv2.imwrite("plain_%d_test_100_r2.png" %i, y_test[i,:,:,1])
