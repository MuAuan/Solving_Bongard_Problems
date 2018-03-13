'''
VGG16
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 32, 32, 64)        1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 32, 32, 64)        36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 16, 16, 64)        0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 16, 16, 128)       73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 16, 16, 128)       147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 8, 8, 128)         0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 8, 8, 256)         295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 8, 8, 256)         590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 8, 8, 256)         590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 4, 4, 256)         0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 4, 4, 512)         1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 4, 4, 512)         2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 4, 4, 512)         2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 2, 2, 512)         0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 2, 2, 512)         2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 2, 2, 512)         2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 2, 2, 512)         2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 1, 1, 512)         0
_________________________________________________________________
sequential_1 (Sequential)    (None, 10)                133898
=================================================================
Total params: 14,848,586
Trainable params: 7,473,482
Non-trainable params: 7,375,104

Cifar10
_________________________________________________________________
for layer in model.layers[7:15]:    layer.trainable = False
1562/1562 [==============================] - 76s - loss: 4.5619e-04 - acc: 0.9999 - val_loss: 1.5312 - val_acc: 0.8400
for layer in model.layers[1:10]:    layer.trainable = False
1562/1562 [==============================] - 78s - loss: 0.0024 - acc: 0.9994 - val_loss: 1.0931 - val_acc: 0.8953
i, ir=  110 8.589934592000007e-06
for layer in model.layers[1:1]:    layer.trainable = False
1562/1562 [==============================] - 104s - loss: 0.0096 - acc: 0.9974 - val_loss: 0.7429 - val_acc: 0.9002
=================================================================

_________________________________________________________________

'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Reshape, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
#from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
import keras.backend as K
#from keras.utils.visualize_util import plot


from getDataSet import getDataSet

#import h5py

def save_history(history, result_file,epochs):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "a") as fp:
        if epochs==0:
            fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\n" % (i+epochs, loss[i], acc[i], val_loss[i], val_acc[i]))
        else:
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\n" % (i+epochs, loss[i], acc[i], val_loss[i], val_acc[i]))


batch_size = 2
num_classes = 300
epochs = 3
data_augmentation = False
img_rows=128
img_cols=128
result_dir="./history"

K.set_image_data_format('channels_last')

# The data, shuffled and split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train,y_train,x_test,y_test = getDataSet(img_rows,img_cols)
    #このままだと読み込んでもらえないので、array型にします。
    #x_train = np.array(x_train).astype(np.float32).reshape((len(x_train),3, 32, 32)) / 255
x_train = np.array(x_train)  #/ 255
y_train = np.array(y_train).astype(np.int32)
x_test = np.array(x_test) #/ 255
y_test = np.array(y_test).astype(np.int32)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=x_train.shape[1:])  #(img_rows, img_cols, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
#vgg19 = VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)
#InceptionV3 = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)
#ResNet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:])) #vgg16,vgg19,InceptionV3,ResNet50
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=top_model(vgg16.output))
#model = Model(input=vgg19.input, output=top_model(vgg19.output))
#model = Model(input=InceptionV3.input, output=top_model(InceptionV3.output))
#model = Model(input=ResNet50.input, output=top_model(ResNet50.output))


# 最後のconv層の直前までの層をfreeze
#trainingするlayerを指定　VGG16では18,15,10,1など 20で全層固定
#trainingするlayerを指定　VGG16では16,11,7,1など 21で全層固定
#trainingするlayerを指定　InceptionV3では310で全層固定
#trainingするlayerを指定　ResNet50では174で全層固定
for layer in model.layers[1:10]:  
    layer.trainable = False

# Fine-tuningのときはSGDの方がよい⇒adamがよかった
lr = 0.00001 #0.00001
opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6) #1e-6
#opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# モデルのサマリを表示
model.summary()

#model.load_weights('params_model_epoch_003.hdf5')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

for i in range(epochs):
    epoch=30
    if not data_augmentation:
        print('Not using data augmentation.')
        """
        history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epoch,
                    verbose=1,
                    validation_split=0.1)
        """
        # 学習履歴をプロット
        #plot_history(history, result_dir)
        
        
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_data=(x_test, y_test),
                  shuffle=True)
        
        # save weights every epoch
        model.save_weights('params_model_epoch_{0:03d}.hdf5'.format(i), True)   
        save_history(history, os.path.join(result_dir, 'history_epoch_{0:03d}.txt'.format(i)),epoch*i)
        save_history(history, os.path.join(result_dir, 'history.txt'),i*epoch)
        
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=2,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epoch,
                            validation_data=(x_test, y_test))
        model.save_weights('params_model_epoch_{0:03d}.hdf5'.format(i), True)   
        save_history(history, os.path.join(result_dir, 'history_epoch_{0:03d}.txt'.format(i)),epoch*i)
        save_history(history, os.path.join(result_dir, 'history.txt'),i*epoch)

    if i%10==0:
        print('i, ir= ',i, lr)
        # save weights every epoch
        model.save_weights('params_model_VGG16L3_i_{0:03d}.hdf5'.format(i), True)
        save_history(history, os.path.join(result_dir, 'history10.txt'),i*epoch)
        
        lr=lr*0.5
        opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6)
        
        # Let's train the model using Adam
        model.compile(loss='categorical_crossentropy',
                  optimizer=opt,metrics=['accuracy'])
    else:
        continue
        
#save_history(history, os.path.join(result_dir, 'history.txt'),i*epoch)

        
"""
 model = Model(input=vgg19.input, output=top_model(vgg19.output))
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 3, 224, 224)       0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 64, 224, 224)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 64, 224, 224)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 64, 112, 112)      0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 128, 112, 112)     73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 128, 112, 112)     147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 128, 56, 56)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 256, 56, 56)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 256, 56, 56)       590080
_________________________________________________________________
10block3_conv3 (Conv2D)        (None, 256, 56, 56)       590080
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 256, 56, 56)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 256, 28, 28)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 512, 28, 28)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 512, 28, 28)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 512, 28, 28)       2359808
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 512, 28, 28)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 512, 14, 14)       0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 512, 14, 14)       2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 512, 14, 14)       2359808
_________________________________________________________________
20block5_conv3 (Conv2D)        (None, 512, 14, 14)       2359808
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 512, 14, 14)       2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 512, 7, 7)         0
_________________________________________________________________
sequential_1 (Sequential)    (None, 10)                6425354
=================================================================
Total params: 26,449,738
Trainable params: 6,425,354
Non-trainable params: 20,024,384
_________________________________________________________________

 model = Model(input=ResNet50.input, output=top_model(ResNet50.output))
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 3, 224, 224)  0
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 64, 112, 112) 9472        input_1[0][0]
__________________________________________________________________________________________________
bn_conv1 (BatchNormalization)   (None, 64, 112, 112) 256         conv1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 64, 112, 112) 0           bn_conv1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 64, 55, 55)   0           activation_1[0][0]
__________________________________________________________________________________________________
res2a_branch2a (Conv2D)         (None, 64, 55, 55)   4160        max_pooling2d_1[0][0]
__________________________________________________________________________________________________
bn2a_branch2a (BatchNormalizati (None, 64, 55, 55)   256         res2a_branch2a[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 64, 55, 55)   0           bn2a_branch2a[0][0]
__________________________________________________________________________________________________
res2a_branch2b (Conv2D)         (None, 64, 55, 55)   36928       activation_2[0][0]
__________________________________________________________________________________________________
10bn2a_branch2b (BatchNormalizati (None, 64, 55, 55)   256         res2a_branch2b[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 64, 55, 55)   0           bn2a_branch2b[0][0]
__________________________________________________________________________________________________
res2a_branch2c (Conv2D)         (None, 256, 55, 55)  16640       activation_3[0][0]
__________________________________________________________________________________________________
res2a_branch1 (Conv2D)          (None, 256, 55, 55)  16640       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
bn2a_branch2c (BatchNormalizati (None, 256, 55, 55)  1024        res2a_branch2c[0][0]
__________________________________________________________________________________________________
bn2a_branch1 (BatchNormalizatio (None, 256, 55, 55)  1024        res2a_branch1[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 256, 55, 55)  0           bn2a_branch2c[0][0]
                                                                 bn2a_branch1[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 256, 55, 55)  0           add_1[0][0]
__________________________________________________________________________________________________
res2b_branch2a (Conv2D)         (None, 64, 55, 55)   16448       activation_4[0][0]
__________________________________________________________________________________________________
bn2b_branch2a (BatchNormalizati (None, 64, 55, 55)   256         res2b_branch2a[0][0]
__________________________________________________________________________________________________
20activation_5 (Activation)       (None, 64, 55, 55)   0           bn2b_branch2a[0][0]
__________________________________________________________________________________________________
res2b_branch2b (Conv2D)         (None, 64, 55, 55)   36928       activation_5[0][0]
__________________________________________________________________________________________________
bn2b_branch2b (BatchNormalizati (None, 64, 55, 55)   256         res2b_branch2b[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 64, 55, 55)   0           bn2b_branch2b[0][0]
__________________________________________________________________________________________________
res2b_branch2c (Conv2D)         (None, 256, 55, 55)  16640       activation_6[0][0]
__________________________________________________________________________________________________
bn2b_branch2c (BatchNormalizati (None, 256, 55, 55)  1024        res2b_branch2c[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 256, 55, 55)  0           bn2b_branch2c[0][0]
                                                                 activation_4[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 256, 55, 55)  0           add_2[0][0]
__________________________________________________________________________________________________
res2c_branch2a (Conv2D)         (None, 64, 55, 55)   16448       activation_7[0][0]
__________________________________________________________________________________________________
bn2c_branch2a (BatchNormalizati (None, 64, 55, 55)   256         res2c_branch2a[0][0]
__________________________________________________________________________________________________
30activation_8 (Activation)       (None, 64, 55, 55)   0           bn2c_branch2a[0][0]
__________________________________________________________________________________________________
res2c_branch2b (Conv2D)         (None, 64, 55, 55)   36928       activation_8[0][0]
__________________________________________________________________________________________________
bn2c_branch2b (BatchNormalizati (None, 64, 55, 55)   256         res2c_branch2b[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 64, 55, 55)   0           bn2c_branch2b[0][0]
__________________________________________________________________________________________________
res2c_branch2c (Conv2D)         (None, 256, 55, 55)  16640       activation_9[0][0]
__________________________________________________________________________________________________
bn2c_branch2c (BatchNormalizati (None, 256, 55, 55)  1024        res2c_branch2c[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 256, 55, 55)  0           bn2c_branch2c[0][0]
                                                                 activation_7[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 256, 55, 55)  0           add_3[0][0]
__________________________________________________________________________________________________
res3a_branch2a (Conv2D)         (None, 128, 28, 28)  32896       activation_10[0][0]
__________________________________________________________________________________________________
bn3a_branch2a (BatchNormalizati (None, 128, 28, 28)  512         res3a_branch2a[0][0]
__________________________________________________________________________________________________
40activation_11 (Activation)      (None, 128, 28, 28)  0           bn3a_branch2a[0][0]
__________________________________________________________________________________________________
res3a_branch2b (Conv2D)         (None, 128, 28, 28)  147584      activation_11[0][0]
__________________________________________________________________________________________________
bn3a_branch2b (BatchNormalizati (None, 128, 28, 28)  512         res3a_branch2b[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 128, 28, 28)  0           bn3a_branch2b[0][0]
__________________________________________________________________________________________________
res3a_branch2c (Conv2D)         (None, 512, 28, 28)  66048       activation_12[0][0]
__________________________________________________________________________________________________
res3a_branch1 (Conv2D)          (None, 512, 28, 28)  131584      activation_10[0][0]
__________________________________________________________________________________________________
bn3a_branch2c (BatchNormalizati (None, 512, 28, 28)  2048        res3a_branch2c[0][0]
__________________________________________________________________________________________________
bn3a_branch1 (BatchNormalizatio (None, 512, 28, 28)  2048        res3a_branch1[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 512, 28, 28)  0           bn3a_branch2c[0][0]
                                                                 bn3a_branch1[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 512, 28, 28)  0           add_4[0][0]
__________________________________________________________________________________________________
50res3b_branch2a (Conv2D)         (None, 128, 28, 28)  65664       activation_13[0][0]
__________________________________________________________________________________________________
bn3b_branch2a (BatchNormalizati (None, 128, 28, 28)  512         res3b_branch2a[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 128, 28, 28)  0           bn3b_branch2a[0][0]
__________________________________________________________________________________________________
res3b_branch2b (Conv2D)         (None, 128, 28, 28)  147584      activation_14[0][0]
__________________________________________________________________________________________________
bn3b_branch2b (BatchNormalizati (None, 128, 28, 28)  512         res3b_branch2b[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 128, 28, 28)  0           bn3b_branch2b[0][0]
__________________________________________________________________________________________________
res3b_branch2c (Conv2D)         (None, 512, 28, 28)  66048       activation_15[0][0]
__________________________________________________________________________________________________
bn3b_branch2c (BatchNormalizati (None, 512, 28, 28)  2048        res3b_branch2c[0][0]
__________________________________________________________________________________________________
add_5 (Add)                     (None, 512, 28, 28)  0           bn3b_branch2c[0][0]
                                                                 activation_13[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 512, 28, 28)  0           add_5[0][0]
__________________________________________________________________________________________________
60res3c_branch2a (Conv2D)         (None, 128, 28, 28)  65664       activation_16[0][0]
__________________________________________________________________________________________________
bn3c_branch2a (BatchNormalizati (None, 128, 28, 28)  512         res3c_branch2a[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 128, 28, 28)  0           bn3c_branch2a[0][0]
__________________________________________________________________________________________________
res3c_branch2b (Conv2D)         (None, 128, 28, 28)  147584      activation_17[0][0]
__________________________________________________________________________________________________
bn3c_branch2b (BatchNormalizati (None, 128, 28, 28)  512         res3c_branch2b[0][0]
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 128, 28, 28)  0           bn3c_branch2b[0][0]
__________________________________________________________________________________________________
res3c_branch2c (Conv2D)         (None, 512, 28, 28)  66048       activation_18[0][0]
__________________________________________________________________________________________________
bn3c_branch2c (BatchNormalizati (None, 512, 28, 28)  2048        res3c_branch2c[0][0]
__________________________________________________________________________________________________
add_6 (Add)                     (None, 512, 28, 28)  0           bn3c_branch2c[0][0]
                                                                 activation_16[0][0]
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 512, 28, 28)  0           add_6[0][0]
__________________________________________________________________________________________________
70res3d_branch2a (Conv2D)         (None, 128, 28, 28)  65664       activation_19[0][0]
__________________________________________________________________________________________________
bn3d_branch2a (BatchNormalizati (None, 128, 28, 28)  512         res3d_branch2a[0][0]
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 128, 28, 28)  0           bn3d_branch2a[0][0]
__________________________________________________________________________________________________
res3d_branch2b (Conv2D)         (None, 128, 28, 28)  147584      activation_20[0][0]
__________________________________________________________________________________________________
bn3d_branch2b (BatchNormalizati (None, 128, 28, 28)  512         res3d_branch2b[0][0]
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 128, 28, 28)  0           bn3d_branch2b[0][0]
__________________________________________________________________________________________________
res3d_branch2c (Conv2D)         (None, 512, 28, 28)  66048       activation_21[0][0]
__________________________________________________________________________________________________
bn3d_branch2c (BatchNormalizati (None, 512, 28, 28)  2048        res3d_branch2c[0][0]
__________________________________________________________________________________________________
add_7 (Add)                     (None, 512, 28, 28)  0           bn3d_branch2c[0][0]
                                                                 activation_19[0][0]
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 512, 28, 28)  0           add_7[0][0]
__________________________________________________________________________________________________
80res4a_branch2a (Conv2D)         (None, 256, 14, 14)  131328      activation_22[0][0]
__________________________________________________________________________________________________
bn4a_branch2a (BatchNormalizati (None, 256, 14, 14)  1024        res4a_branch2a[0][0]
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 256, 14, 14)  0           bn4a_branch2a[0][0]
__________________________________________________________________________________________________
res4a_branch2b (Conv2D)         (None, 256, 14, 14)  590080      activation_23[0][0]
__________________________________________________________________________________________________
bn4a_branch2b (BatchNormalizati (None, 256, 14, 14)  1024        res4a_branch2b[0][0]
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 256, 14, 14)  0           bn4a_branch2b[0][0]
__________________________________________________________________________________________________
res4a_branch2c (Conv2D)         (None, 1024, 14, 14) 263168      activation_24[0][0]
__________________________________________________________________________________________________
res4a_branch1 (Conv2D)          (None, 1024, 14, 14) 525312      activation_22[0][0]
__________________________________________________________________________________________________
bn4a_branch2c (BatchNormalizati (None, 1024, 14, 14) 4096        res4a_branch2c[0][0]
__________________________________________________________________________________________________
bn4a_branch1 (BatchNormalizatio (None, 1024, 14, 14) 4096        res4a_branch1[0][0]
__________________________________________________________________________________________________
90add_8 (Add)                     (None, 1024, 14, 14) 0           bn4a_branch2c[0][0]
                                                                 bn4a_branch1[0][0]
__52sec________________________________________________________________________________________________
activation_25 (Activation)      (None, 1024, 14, 14) 0           add_8[0][0]
__________________________________________________________________________________________________
res4b_branch2a (Conv2D)         (None, 256, 14, 14)  262400      activation_25[0][0]
__________________________________________________________________________________________________
bn4b_branch2a (BatchNormalizati (None, 256, 14, 14)  1024        res4b_branch2a[0][0]
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 256, 14, 14)  0           bn4b_branch2a[0][0]
__________________________________________________________________________________________________
res4b_branch2b (Conv2D)         (None, 256, 14, 14)  590080      activation_26[0][0]
__________________________________________________________________________________________________
bn4b_branch2b (BatchNormalizati (None, 256, 14, 14)  1024        res4b_branch2b[0][0]
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 256, 14, 14)  0           bn4b_branch2b[0][0]
__________________________________________________________________________________________________
res4b_branch2c (Conv2D)         (None, 1024, 14, 14) 263168      activation_27[0][0]
__________________________________________________________________________________________________
bn4b_branch2c (BatchNormalizati (None, 1024, 14, 14) 4096        res4b_branch2c[0][0]
__________________________________________________________________________________________________
100add_9 (Add)                     (None, 1024, 14, 14) 0           bn4b_branch2c[0][0]
                                                                 activation_25[0][0]
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 1024, 14, 14) 0           add_9[0][0]
__________________________________________________________________________________________________
res4c_branch2a (Conv2D)         (None, 256, 14, 14)  262400      activation_28[0][0]
__________________________________________________________________________________________________
bn4c_branch2a (BatchNormalizati (None, 256, 14, 14)  1024        res4c_branch2a[0][0]
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 256, 14, 14)  0           bn4c_branch2a[0][0]
__________________________________________________________________________________________________
res4c_branch2b (Conv2D)         (None, 256, 14, 14)  590080      activation_29[0][0]
__________________________________________________________________________________________________
bn4c_branch2b (BatchNormalizati (None, 256, 14, 14)  1024        res4c_branch2b[0][0]
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 256, 14, 14)  0           bn4c_branch2b[0][0]
__________________________________________________________________________________________________
res4c_branch2c (Conv2D)         (None, 1024, 14, 14) 263168      activation_30[0][0]
__________________________________________________________________________________________________
bn4c_branch2c (BatchNormalizati (None, 1024, 14, 14) 4096        res4c_branch2c[0][0]
__________________________________________________________________________________________________
110add_10 (Add)                    (None, 1024, 14, 14) 0           bn4c_branch2c[0][0]
                                                                 activation_28[0][0]
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 1024, 14, 14) 0           add_10[0][0]
__________________________________________________________________________________________________
res4d_branch2a (Conv2D)         (None, 256, 14, 14)  262400      activation_31[0][0]
__________________________________________________________________________________________________
bn4d_branch2a (BatchNormalizati (None, 256, 14, 14)  1024        res4d_branch2a[0][0]
__________________________________________________________________________________________________
activation_32 (Activation)      (None, 256, 14, 14)  0           bn4d_branch2a[0][0]
__________________________________________________________________________________________________
res4d_branch2b (Conv2D)         (None, 256, 14, 14)  590080      activation_32[0][0]
__________________________________________________________________________________________________
bn4d_branch2b (BatchNormalizati (None, 256, 14, 14)  1024        res4d_branch2b[0][0]
__________________________________________________________________________________________________
activation_33 (Activation)      (None, 256, 14, 14)  0           bn4d_branch2b[0][0]
__________________________________________________________________________________________________
res4d_branch2c (Conv2D)         (None, 1024, 14, 14) 263168      activation_33[0][0]
__________________________________________________________________________________________________
bn4d_branch2c (BatchNormalizati (None, 1024, 14, 14) 4096        res4d_branch2c[0][0]
__________________________________________________________________________________________________
120add_11 (Add)                    (None, 1024, 14, 14) 0           bn4d_branch2c[0][0]
                                                                 activation_31[0][0]
__________________________________________________________________________________________________
activation_34 (Activation)      (None, 1024, 14, 14) 0           add_11[0][0]
__________________________________________________________________________________________________
res4e_branch2a (Conv2D)         (None, 256, 14, 14)  262400      activation_34[0][0]
__________________________________________________________________________________________________
bn4e_branch2a (BatchNormalizati (None, 256, 14, 14)  1024        res4e_branch2a[0][0]
__________________________________________________________________________________________________
activation_35 (Activation)      (None, 256, 14, 14)  0           bn4e_branch2a[0][0]
__________________________________________________________________________________________________
res4e_branch2b (Conv2D)         (None, 256, 14, 14)  590080      activation_35[0][0]
__________________________________________________________________________________________________
bn4e_branch2b (BatchNormalizati (None, 256, 14, 14)  1024        res4e_branch2b[0][0]
__________________________________________________________________________________________________
activation_36 (Activation)      (None, 256, 14, 14)  0           bn4e_branch2b[0][0]
__________________________________________________________________________________________________
res4e_branch2c (Conv2D)         (None, 1024, 14, 14) 263168      activation_36[0][0]
__________________________________________________________________________________________________
bn4e_branch2c (BatchNormalizati (None, 1024, 14, 14) 4096        res4e_branch2c[0][0]
__________________________________________________________________________________________________
130add_12 (Add)                    (None, 1024, 14, 14) 0           bn4e_branch2c[0][0]
                                                                 activation_34[0][0]
__________________________________________________________________________________________________
activation_37 (Activation)      (None, 1024, 14, 14) 0           add_12[0][0]
__________________________________________________________________________________________________
res4f_branch2a (Conv2D)         (None, 256, 14, 14)  262400      activation_37[0][0]
__________________________________________________________________________________________________
bn4f_branch2a (BatchNormalizati (None, 256, 14, 14)  1024        res4f_branch2a[0][0]
__________________________________________________________________________________________________
activation_38 (Activation)      (None, 256, 14, 14)  0           bn4f_branch2a[0][0]
__________________________________________________________________________________________________
res4f_branch2b (Conv2D)         (None, 256, 14, 14)  590080      activation_38[0][0]
__________________________________________________________________________________________________
bn4f_branch2b (BatchNormalizati (None, 256, 14, 14)  1024        res4f_branch2b[0][0]
__________________________________________________________________________________________________
activation_39 (Activation)      (None, 256, 14, 14)  0           bn4f_branch2b[0][0]
__________________________________________________________________________________________________
res4f_branch2c (Conv2D)         (None, 1024, 14, 14) 263168      activation_39[0][0]
__________________________________________________________________________________________________
bn4f_branch2c (BatchNormalizati (None, 1024, 14, 14) 4096        res4f_branch2c[0][0]
__________________________________________________________________________________________________
140add_13 (Add)                    (None, 1024, 14, 14) 0           bn4f_branch2c[0][0]
                                                                 activation_37[0][0]
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 1024, 14, 14) 0           add_13[0][0]
__________________________________________________________________________________________________
res5a_branch2a (Conv2D)         (None, 512, 7, 7)    524800      activation_40[0][0]
__________________________________________________________________________________________________
bn5a_branch2a (BatchNormalizati (None, 512, 7, 7)    2048        res5a_branch2a[0][0]
__________________________________________________________________________________________________
activation_41 (Activation)      (None, 512, 7, 7)    0           bn5a_branch2a[0][0]
__________________________________________________________________________________________________
res5a_branch2b (Conv2D)         (None, 512, 7, 7)    2359808     activation_41[0][0]
__________________________________________________________________________________________________
bn5a_branch2b (BatchNormalizati (None, 512, 7, 7)    2048        res5a_branch2b[0][0]
__________________________________________________________________________________________________
activation_42 (Activation)      (None, 512, 7, 7)    0           bn5a_branch2b[0][0]
__________________________________________________________________________________________________
res5a_branch2c (Conv2D)         (None, 2048, 7, 7)   1050624     activation_42[0][0]
__________________________________________________________________________________________________
res5a_branch1 (Conv2D)          (None, 2048, 7, 7)   2099200     activation_40[0][0]
__________________________________________________________________________________________________
150bn5a_branch2c (BatchNormalizati (None, 2048, 7, 7)   8192        res5a_branch2c[0][0]
__________________________________________________________________________________________________
bn5a_branch1 (BatchNormalizatio (None, 2048, 7, 7)   8192        res5a_branch1[0][0]
__________________________________________________________________________________________________
add_14 (Add)                    (None, 2048, 7, 7)   0           bn5a_branch2c[0][0]
                                                                 bn5a_branch1[0][0]
____152____35-38sec__________________________________________________________________________________________
activation_43 (Activation)      (None, 2048, 7, 7)   0           add_14[0][0]
__________________________________________________________________________________________________
res5b_branch2a (Conv2D)         (None, 512, 7, 7)    1049088     activation_43[0][0]
__________________________________________________________________________________________________
bn5b_branch2a (BatchNormalizati (None, 512, 7, 7)    2048        res5b_branch2a[0][0]
__________________________________________________________________________________________________
activation_44 (Activation)      (None, 512, 7, 7)    0           bn5b_branch2a[0][0]
__________________________________________________________________________________________________
res5b_branch2b (Conv2D)         (None, 512, 7, 7)    2359808     activation_44[0][0]
__________________________________________________________________________________________________
bn5b_branch2b (BatchNormalizati (None, 512, 7, 7)    2048        res5b_branch2b[0][0]
__________________________________________________________________________________________________
activation_45 (Activation)      (None, 512, 7, 7)    0           bn5b_branch2b[0][0]
__________________________________________________________________________________________________
160res5b_branch2c (Conv2D)         (None, 2048, 7, 7)   1050624     activation_45[0][0]
__________________________________________________________________________________________________
bn5b_branch2c (BatchNormalizati (None, 2048, 7, 7)   8192        res5b_branch2c[0][0]
__________________________________________________________________________________________________
add_15 (Add)                    (None, 2048, 7, 7)   0           bn5b_branch2c[0][0]
                                                                 activation_43[0][0]
___162_______________________________________________________________________________________________
activation_46 (Activation)      (None, 2048, 7, 7)   0           add_15[0][0]
__________________________________________________________________________________________________
res5c_branch2a (Conv2D)         (None, 512, 7, 7)    1049088     activation_46[0][0]
__________________________________________________________________________________________________
bn5c_branch2a (BatchNormalizati (None, 512, 7, 7)    2048        res5c_branch2a[0][0]
__________________________________________________________________________________________________
activation_47 (Activation)      (None, 512, 7, 7)    0           bn5c_branch2a[0][0]
__________________________________________________________________________________________________
res5c_branch2b (Conv2D)         (None, 512, 7, 7)    2359808     activation_47[0][0]
__________________________________________________________________________________________________
bn5c_branch2b (BatchNormalizati (None, 512, 7, 7)    2048        res5c_branch2b[0][0]
__________________________________________________________________________________________________
activation_48 (Activation)      (None, 512, 7, 7)    0           bn5c_branch2b[0][0]
__________________________________________________________________________________________________
170res5c_branch2c (Conv2D)         (None, 2048, 7, 7)   1050624     activation_48[0][0]
__________________________________________________________________________________________________
bn5c_branch2c (BatchNormalizati (None, 2048, 7, 7)   8192        res5c_branch2c[0][0]
__________________________________________________________________________________________________
add_16 (Add)                    (None, 2048, 7, 7)   0           bn5c_branch2c[0][0]
                                                                 activation_46[0][0]
____172___32sec___________________________________________________________________________________________
activation_49 (Activation)      (None, 2048, 7, 7)   0           add_16[0][0]
__________________________________________________________________________________________________
avg_pool (AveragePooling2D)     (None, 2048, 1, 1)   0           activation_49[0][0]
__________________________________________________________________________________________________
175sequential_1 (Sequential)       (None, 10)           527114      avg_pool[0][0]
==================================================================================================
for layer in model.layers[1:90]
==================================================================================================
Total params: 24,114,826
Trainable params: 21,096,714
Non-trainable params: 3,018,112
__________________________________________________________________________________________________

  model = Model(input=InceptionV3.input, output=top_model(InceptionV3.output))
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 3, 224, 224)  0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 111, 111) 864         input_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 111, 111) 96          conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 111, 111) 0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 109, 109) 9216        activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 109, 109) 96          conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 109, 109) 0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 109, 109) 18432       activation_2[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 64, 109, 109) 192         conv2d_3[0][0]
__________________________________________________________________________________________________
10 activation_3 (Activation)       (None, 64, 109, 109) 0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 64, 54, 54)   0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 80, 54, 54)   5120        max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 80, 54, 54)   240         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 80, 54, 54)   0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 192, 52, 52)  138240      activation_4[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 192, 52, 52)  576         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 192, 52, 52)  0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 192, 25, 25)  0           activation_5[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 64, 25, 25)   12288       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
20 batch_normalization_9 (BatchNor (None, 64, 25, 25)   192         conv2d_9[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 64, 25, 25)   0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 48, 25, 25)   9216        max_pooling2d_2[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 96, 25, 25)   55296       activation_9[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 48, 25, 25)   144         conv2d_7[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 96, 25, 25)   288         conv2d_10[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 48, 25, 25)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 96, 25, 25)   0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 192, 25, 25)  0           max_pooling2d_2[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 64, 25, 25)   12288       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
30 conv2d_8 (Conv2D)               (None, 64, 25, 25)   76800       activation_7[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 96, 25, 25)   82944       activation_10[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 25, 25)   6144        average_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 64, 25, 25)   192         conv2d_6[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 64, 25, 25)   192         conv2d_8[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 96, 25, 25)   288         conv2d_11[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 32, 25, 25)   96          conv2d_12[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 64, 25, 25)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 64, 25, 25)   0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 96, 25, 25)   0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
40 activation_12 (Activation)      (None, 32, 25, 25)   0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
mixed0 (Concatenate)            (None, 256, 25, 25)  0           activation_6[0][0]
                                                                 activation_8[0][0]
                                                                 activation_11[0][0]
                                                                 activation_12[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 64, 25, 25)   16384       mixed0[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 64, 25, 25)   192         conv2d_16[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 64, 25, 25)   0           batch_normalization_16[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 48, 25, 25)   12288       mixed0[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 96, 25, 25)   55296       activation_16[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 48, 25, 25)   144         conv2d_14[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 96, 25, 25)   288         conv2d_17[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 48, 25, 25)   0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
50 activation_17 (Activation)      (None, 96, 25, 25)   0           batch_normalization_17[0][0]
__________________________________________________________________________________________________
average_pooling2d_2 (AveragePoo (None, 256, 25, 25)  0           mixed0[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 64, 25, 25)   16384       mixed0[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 64, 25, 25)   76800       activation_14[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 96, 25, 25)   82944       activation_17[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 64, 25, 25)   16384       average_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 64, 25, 25)   192         conv2d_13[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 64, 25, 25)   192         conv2d_15[0][0]
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 96, 25, 25)   288         conv2d_18[0][0]
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 64, 25, 25)   192         conv2d_19[0][0]
__________________________________________________________________________________________________
60 activation_13 (Activation)      (None, 64, 25, 25)   0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 64, 25, 25)   0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 96, 25, 25)   0           batch_normalization_18[0][0]
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 64, 25, 25)   0           batch_normalization_19[0][0]
__________________________________________________________________________________________________
mixed1 (Concatenate)            (None, 288, 25, 25)  0           activation_13[0][0]
                                                                 activation_15[0][0]
                                                                 activation_18[0][0]
                                                                 activation_19[0][0]
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 64, 25, 25)   18432       mixed1[0][0]
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 64, 25, 25)   192         conv2d_23[0][0]
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 64, 25, 25)   0           batch_normalization_23[0][0]
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 48, 25, 25)   13824       mixed1[0][0]
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 96, 25, 25)   55296       activation_23[0][0]
__________________________________________________________________________________________________
70 batch_normalization_21 (BatchNo (None, 48, 25, 25)   144         conv2d_21[0][0]
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 96, 25, 25)   288         conv2d_24[0][0]
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 48, 25, 25)   0           batch_normalization_21[0][0]
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 96, 25, 25)   0           batch_normalization_24[0][0]
__________________________________________________________________________________________________
average_pooling2d_3 (AveragePoo (None, 288, 25, 25)  0           mixed1[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 64, 25, 25)   18432       mixed1[0][0]
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 64, 25, 25)   76800       activation_21[0][0]
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 96, 25, 25)   82944       activation_24[0][0]
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 64, 25, 25)   18432       average_pooling2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 64, 25, 25)   192         conv2d_20[0][0]
__________________________________________________________________________________________________
80 batch_normalization_22 (BatchNo (None, 64, 25, 25)   192         conv2d_22[0][0]
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 96, 25, 25)   288         conv2d_25[0][0]
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 64, 25, 25)   192         conv2d_26[0][0]
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 64, 25, 25)   0           batch_normalization_20[0][0]
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 64, 25, 25)   0           batch_normalization_22[0][0]
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 96, 25, 25)   0           batch_normalization_25[0][0]
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 64, 25, 25)   0           batch_normalization_26[0][0]
__________________________________________________________________________________________________
mixed2 (Concatenate)            (None, 288, 25, 25)  0           activation_20[0][0]
                                                                 activation_22[0][0]
                                                                 activation_25[0][0]
                                                                 activation_26[0][0]
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 64, 25, 25)   18432       mixed2[0][0]
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 64, 25, 25)   192         conv2d_28[0][0]
__________________________________________________________________________________________________
90 activation_28 (Activation)      (None, 64, 25, 25)   0           batch_normalization_28[0][0]
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 96, 25, 25)   55296       activation_28[0][0]
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 96, 25, 25)   288         conv2d_29[0][0]
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 96, 25, 25)   0           batch_normalization_29[0][0]
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 384, 12, 12)  995328      mixed2[0][0]
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 96, 12, 12)   82944       activation_29[0][0]
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 384, 12, 12)  1152        conv2d_27[0][0]
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 96, 12, 12)   288         conv2d_30[0][0]
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 384, 12, 12)  0           batch_normalization_27[0][0]
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 96, 12, 12)   0           batch_normalization_30[0][0]
__________________________________________________________________________________________________
100 max_pooling2d_3 (MaxPooling2D)  (None, 288, 12, 12)  0           mixed2[0][0]
__________________________________________________________________________________________________
mixed3 (Concatenate)            (None, 768, 12, 12)  0           activation_27[0][0]
                                                                 activation_30[0][0]
                                                                 max_pooling2d_3[0][0]
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 128, 12, 12)  98304       mixed3[0][0]
__________________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, 128, 12, 12)  384         conv2d_35[0][0]
__________________________________________________________________________________________________
activation_35 (Activation)      (None, 128, 12, 12)  0           batch_normalization_35[0][0]
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 128, 12, 12)  114688      activation_35[0][0]
__________________________________________________________________________________________________
batch_normalization_36 (BatchNo (None, 128, 12, 12)  384         conv2d_36[0][0]
__________________________________________________________________________________________________
activation_36 (Activation)      (None, 128, 12, 12)  0           batch_normalization_36[0][0]
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 128, 12, 12)  98304       mixed3[0][0]
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 128, 12, 12)  114688      activation_36[0][0]
__________________________________________________________________________________________________
110 batch_normalization_32 (BatchNo (None, 128, 12, 12)  384         conv2d_32[0][0]
__________________________________________________________________________________________________
batch_normalization_37 (BatchNo (None, 128, 12, 12)  384         conv2d_37[0][0]
__________________________________________________________________________________________________
activation_32 (Activation)      (None, 128, 12, 12)  0           batch_normalization_32[0][0]
__________________________________________________________________________________________________
activation_37 (Activation)      (None, 128, 12, 12)  0           batch_normalization_37[0][0]
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 128, 12, 12)  114688      activation_32[0][0]
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 128, 12, 12)  114688      activation_37[0][0]
__________________________________________________________________________________________________
batch_normalization_33 (BatchNo (None, 128, 12, 12)  384         conv2d_33[0][0]
__________________________________________________________________________________________________
batch_normalization_38 (BatchNo (None, 128, 12, 12)  384         conv2d_38[0][0]
__________________________________________________________________________________________________
activation_33 (Activation)      (None, 128, 12, 12)  0           batch_normalization_33[0][0]
__________________________________________________________________________________________________
activation_38 (Activation)      (None, 128, 12, 12)  0           batch_normalization_38[0][0]
__________________________________________________________________________________________________
120 average_pooling2d_4 (AveragePoo (None, 768, 12, 12)  0           mixed3[0][0]
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 192, 12, 12)  147456      mixed3[0][0]
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 192, 12, 12)  172032      activation_33[0][0]
__________________________________________________________________________________________________
conv2d_39 (Conv2D)              (None, 192, 12, 12)  172032      activation_38[0][0]
__________________________________________________________________________________________________
conv2d_40 (Conv2D)              (None, 192, 12, 12)  147456      average_pooling2d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_31 (BatchNo (None, 192, 12, 12)  576         conv2d_31[0][0]
__________________________________________________________________________________________________
batch_normalization_34 (BatchNo (None, 192, 12, 12)  576         conv2d_34[0][0]
__________________________________________________________________________________________________
batch_normalization_39 (BatchNo (None, 192, 12, 12)  576         conv2d_39[0][0]
__________________________________________________________________________________________________
batch_normalization_40 (BatchNo (None, 192, 12, 12)  576         conv2d_40[0][0]
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 192, 12, 12)  0           batch_normalization_31[0][0]
__________________________________________________________________________________________________
130 activation_34 (Activation)      (None, 192, 12, 12)  0           batch_normalization_34[0][0]
__________________________________________________________________________________________________
activation_39 (Activation)      (None, 192, 12, 12)  0           batch_normalization_39[0][0]
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 192, 12, 12)  0           batch_normalization_40[0][0]
__________________________________________________________________________________________________
mixed4 (Concatenate)            (None, 768, 12, 12)  0           activation_31[0][0]
                                                                 activation_34[0][0]
                                                                 activation_39[0][0]
                                                                 activation_40[0][0]
__________________________________________________________________________________________________
conv2d_45 (Conv2D)              (None, 160, 12, 12)  122880      mixed4[0][0]
__________________________________________________________________________________________________
batch_normalization_45 (BatchNo (None, 160, 12, 12)  480         conv2d_45[0][0]
__________________________________________________________________________________________________
activation_45 (Activation)      (None, 160, 12, 12)  0           batch_normalization_45[0][0]
__________________________________________________________________________________________________
conv2d_46 (Conv2D)              (None, 160, 12, 12)  179200      activation_45[0][0]
__________________________________________________________________________________________________
batch_normalization_46 (BatchNo (None, 160, 12, 12)  480         conv2d_46[0][0]
__________________________________________________________________________________________________
activation_46 (Activation)      (None, 160, 12, 12)  0           batch_normalization_46[0][0]
__________________________________________________________________________________________________
140 conv2d_42 (Conv2D)              (None, 160, 12, 12)  122880      mixed4[0][0]
__________________________________________________________________________________________________
conv2d_47 (Conv2D)              (None, 160, 12, 12)  179200      activation_46[0][0]
__________________________________________________________________________________________________
batch_normalization_42 (BatchNo (None, 160, 12, 12)  480         conv2d_42[0][0]
__________________________________________________________________________________________________
batch_normalization_47 (BatchNo (None, 160, 12, 12)  480         conv2d_47[0][0]
__________________________________________________________________________________________________
activation_42 (Activation)      (None, 160, 12, 12)  0           batch_normalization_42[0][0]
__________________________________________________________________________________________________
activation_47 (Activation)      (None, 160, 12, 12)  0           batch_normalization_47[0][0]
__________________________________________________________________________________________________
conv2d_43 (Conv2D)              (None, 160, 12, 12)  179200      activation_42[0][0]
__________________________________________________________________________________________________
conv2d_48 (Conv2D)              (None, 160, 12, 12)  179200      activation_47[0][0]
__________________________________________________________________________________________________
batch_normalization_43 (BatchNo (None, 160, 12, 12)  480         conv2d_43[0][0]
__________________________________________________________________________________________________
batch_normalization_48 (BatchNo (None, 160, 12, 12)  480         conv2d_48[0][0]
__________________________________________________________________________________________________
150 activation_43 (Activation)      (None, 160, 12, 12)  0           batch_normalization_43[0][0]
__________________________________________________________________________________________________
activation_48 (Activation)      (None, 160, 12, 12)  0           batch_normalization_48[0][0]
__________________________________________________________________________________________________
average_pooling2d_5 (AveragePoo (None, 768, 12, 12)  0           mixed4[0][0]
__________________________________________________________________________________________________
conv2d_41 (Conv2D)              (None, 192, 12, 12)  147456      mixed4[0][0]
__________________________________________________________________________________________________
conv2d_44 (Conv2D)              (None, 192, 12, 12)  215040      activation_43[0][0]
__________________________________________________________________________________________________
conv2d_49 (Conv2D)              (None, 192, 12, 12)  215040      activation_48[0][0]
__________________________________________________________________________________________________
conv2d_50 (Conv2D)              (None, 192, 12, 12)  147456      average_pooling2d_5[0][0]
__________________________________________________________________________________________________
batch_normalization_41 (BatchNo (None, 192, 12, 12)  576         conv2d_41[0][0]
__________________________________________________________________________________________________
batch_normalization_44 (BatchNo (None, 192, 12, 12)  576         conv2d_44[0][0]
__________________________________________________________________________________________________
batch_normalization_49 (BatchNo (None, 192, 12, 12)  576         conv2d_49[0][0]
__________________________________________________________________________________________________
160 batch_normalization_50 (BatchNo (None, 192, 12, 12)  576         conv2d_50[0][0]
__________________________________________________________________________________________________
activation_41 (Activation)      (None, 192, 12, 12)  0           batch_normalization_41[0][0]
__________________________________________________________________________________________________
activation_44 (Activation)      (None, 192, 12, 12)  0           batch_normalization_44[0][0]
__________________________________________________________________________________________________
activation_49 (Activation)      (None, 192, 12, 12)  0           batch_normalization_49[0][0]
__________________________________________________________________________________________________
activation_50 (Activation)      (None, 192, 12, 12)  0           batch_normalization_50[0][0]
__________________________________________________________________________________________________
mixed5 (Concatenate)            (None, 768, 12, 12)  0           activation_41[0][0]
                                                                 activation_44[0][0]
                                                                 activation_49[0][0]
                                                                 activation_50[0][0]
__________________________________________________________________________________________________
conv2d_55 (Conv2D)              (None, 160, 12, 12)  122880      mixed5[0][0]
__________________________________________________________________________________________________
batch_normalization_55 (BatchNo (None, 160, 12, 12)  480         conv2d_55[0][0]
__________________________________________________________________________________________________
activation_55 (Activation)      (None, 160, 12, 12)  0           batch_normalization_55[0][0]
__________________________________________________________________________________________________
conv2d_56 (Conv2D)              (None, 160, 12, 12)  179200      activation_55[0][0]
__________________________________________________________________________________________________
170 batch_normalization_56 (BatchNo (None, 160, 12, 12)  480         conv2d_56[0][0]
__________________________________________________________________________________________________
activation_56 (Activation)      (None, 160, 12, 12)  0           batch_normalization_56[0][0]
__________________________________________________________________________________________________
conv2d_52 (Conv2D)              (None, 160, 12, 12)  122880      mixed5[0][0]
__________________________________________________________________________________________________
conv2d_57 (Conv2D)              (None, 160, 12, 12)  179200      activation_56[0][0]
__________________________________________________________________________________________________
batch_normalization_52 (BatchNo (None, 160, 12, 12)  480         conv2d_52[0][0]
__________________________________________________________________________________________________
batch_normalization_57 (BatchNo (None, 160, 12, 12)  480         conv2d_57[0][0]
__________________________________________________________________________________________________
activation_52 (Activation)      (None, 160, 12, 12)  0           batch_normalization_52[0][0]
__________________________________________________________________________________________________
activation_57 (Activation)      (None, 160, 12, 12)  0           batch_normalization_57[0][0]
__________________________________________________________________________________________________
conv2d_53 (Conv2D)              (None, 160, 12, 12)  179200      activation_52[0][0]
__________________________________________________________________________________________________
conv2d_58 (Conv2D)              (None, 160, 12, 12)  179200      activation_57[0][0]
__________________________________________________________________________________________________
180 batch_normalization_53 (BatchNo (None, 160, 12, 12)  480         conv2d_53[0][0]
__________________________________________________________________________________________________
batch_normalization_58 (BatchNo (None, 160, 12, 12)  480         conv2d_58[0][0]
__________________________________________________________________________________________________
activation_53 (Activation)      (None, 160, 12, 12)  0           batch_normalization_53[0][0]
__________________________________________________________________________________________________
activation_58 (Activation)      (None, 160, 12, 12)  0           batch_normalization_58[0][0]
__________________________________________________________________________________________________
average_pooling2d_6 (AveragePoo (None, 768, 12, 12)  0           mixed5[0][0]
__________________________________________________________________________________________________
conv2d_51 (Conv2D)              (None, 192, 12, 12)  147456      mixed5[0][0]
__________________________________________________________________________________________________
conv2d_54 (Conv2D)              (None, 192, 12, 12)  215040      activation_53[0][0]
__________________________________________________________________________________________________
conv2d_59 (Conv2D)              (None, 192, 12, 12)  215040      activation_58[0][0]
__________________________________________________________________________________________________
conv2d_60 (Conv2D)              (None, 192, 12, 12)  147456      average_pooling2d_6[0][0]
__________________________________________________________________________________________________
batch_normalization_51 (BatchNo (None, 192, 12, 12)  576         conv2d_51[0][0]
__________________________________________________________________________________________________
190 batch_normalization_54 (BatchNo (None, 192, 12, 12)  576         conv2d_54[0][0]
__________________________________________________________________________________________________
batch_normalization_59 (BatchNo (None, 192, 12, 12)  576         conv2d_59[0][0]
__________________________________________________________________________________________________
batch_normalization_60 (BatchNo (None, 192, 12, 12)  576         conv2d_60[0][0]
__________________________________________________________________________________________________
activation_51 (Activation)      (None, 192, 12, 12)  0           batch_normalization_51[0][0]
__________________________________________________________________________________________________
activation_54 (Activation)      (None, 192, 12, 12)  0           batch_normalization_54[0][0]
__________________________________________________________________________________________________
activation_59 (Activation)      (None, 192, 12, 12)  0           batch_normalization_59[0][0]
__________________________________________________________________________________________________
activation_60 (Activation)      (None, 192, 12, 12)  0           batch_normalization_60[0][0]
__________________________________________________________________________________________________
mixed6 (Concatenate)            (None, 768, 12, 12)  0           activation_51[0][0]
                                                                 activation_54[0][0]
                                                                 activation_59[0][0]
                                                                 activation_60[0][0]
__________________________________________________________________________________________________
conv2d_65 (Conv2D)              (None, 192, 12, 12)  147456      mixed6[0][0]
__________________________________________________________________________________________________
batch_normalization_65 (BatchNo (None, 192, 12, 12)  576         conv2d_65[0][0]
__________________________________________________________________________________________________
200 activation_65 (Activation)      (None, 192, 12, 12)  0           batch_normalization_65[0][0]
__________________________________________________________________________________________________
conv2d_66 (Conv2D)              (None, 192, 12, 12)  258048      activation_65[0][0]
__________________________________________________________________________________________________
batch_normalization_66 (BatchNo (None, 192, 12, 12)  576         conv2d_66[0][0]
__________________________________________________________________________________________________
activation_66 (Activation)      (None, 192, 12, 12)  0           batch_normalization_66[0][0]
__________________________________________________________________________________________________
conv2d_62 (Conv2D)              (None, 192, 12, 12)  147456      mixed6[0][0]
__________________________________________________________________________________________________
conv2d_67 (Conv2D)              (None, 192, 12, 12)  258048      activation_66[0][0]
__________________________________________________________________________________________________
batch_normalization_62 (BatchNo (None, 192, 12, 12)  576         conv2d_62[0][0]
__________________________________________________________________________________________________
batch_normalization_67 (BatchNo (None, 192, 12, 12)  576         conv2d_67[0][0]
__________________________________________________________________________________________________
activation_62 (Activation)      (None, 192, 12, 12)  0           batch_normalization_62[0][0]
__________________________________________________________________________________________________
activation_67 (Activation)      (None, 192, 12, 12)  0           batch_normalization_67[0][0]
__________________________________________________________________________________________________
210 conv2d_63 (Conv2D)              (None, 192, 12, 12)  258048      activation_62[0][0]
__________________________________________________________________________________________________
conv2d_68 (Conv2D)              (None, 192, 12, 12)  258048      activation_67[0][0]
__________________________________________________________________________________________________
batch_normalization_63 (BatchNo (None, 192, 12, 12)  576         conv2d_63[0][0]
__________________________________________________________________________________________________
batch_normalization_68 (BatchNo (None, 192, 12, 12)  576         conv2d_68[0][0]
__________________________________________________________________________________________________
activation_63 (Activation)      (None, 192, 12, 12)  0           batch_normalization_63[0][0]
__________________________________________________________________________________________________
activation_68 (Activation)      (None, 192, 12, 12)  0           batch_normalization_68[0][0]
__________________________________________________________________________________________________
average_pooling2d_7 (AveragePoo (None, 768, 12, 12)  0           mixed6[0][0]
__________________________________________________________________________________________________
conv2d_61 (Conv2D)              (None, 192, 12, 12)  147456      mixed6[0][0]
__________________________________________________________________________________________________
conv2d_64 (Conv2D)              (None, 192, 12, 12)  258048      activation_63[0][0]
__________________________________________________________________________________________________
conv2d_69 (Conv2D)              (None, 192, 12, 12)  258048      activation_68[0][0]
__________________________________________________________________________________________________
220 conv2d_70 (Conv2D)              (None, 192, 12, 12)  147456      average_pooling2d_7[0][0]
__________________________________________________________________________________________________
batch_normalization_61 (BatchNo (None, 192, 12, 12)  576         conv2d_61[0][0]
__________________________________________________________________________________________________
batch_normalization_64 (BatchNo (None, 192, 12, 12)  576         conv2d_64[0][0]
__________________________________________________________________________________________________
batch_normalization_69 (BatchNo (None, 192, 12, 12)  576         conv2d_69[0][0]
__________________________________________________________________________________________________
batch_normalization_70 (BatchNo (None, 192, 12, 12)  576         conv2d_70[0][0]
__________________________________________________________________________________________________
activation_61 (Activation)      (None, 192, 12, 12)  0           batch_normalization_61[0][0]
__________________________________________________________________________________________________
activation_64 (Activation)      (None, 192, 12, 12)  0           batch_normalization_64[0][0]
__________________________________________________________________________________________________
activation_69 (Activation)      (None, 192, 12, 12)  0           batch_normalization_69[0][0]
__________________________________________________________________________________________________
activation_70 (Activation)      (None, 192, 12, 12)  0           batch_normalization_70[0][0]
__________________________________________________________________________________________________
mixed7 (Concatenate)            (None, 768, 12, 12)  0           activation_61[0][0]
                                                                 activation_64[0][0]
                                                                 activation_69[0][0]
                                                                 activation_70[0][0]
__________________________________________________________________________________________________
230 conv2d_73 (Conv2D)              (None, 192, 12, 12)  147456      mixed7[0][0]
__________________________________________________________________________________________________
batch_normalization_73 (BatchNo (None, 192, 12, 12)  576         conv2d_73[0][0]
__________________________________________________________________________________________________
activation_73 (Activation)      (None, 192, 12, 12)  0           batch_normalization_73[0][0]
__________________________________________________________________________________________________
conv2d_74 (Conv2D)              (None, 192, 12, 12)  258048      activation_73[0][0]
__________________________________________________________________________________________________
batch_normalization_74 (BatchNo (None, 192, 12, 12)  576         conv2d_74[0][0]
__________________________________________________________________________________________________
activation_74 (Activation)      (None, 192, 12, 12)  0           batch_normalization_74[0][0]
__________________________________________________________________________________________________
conv2d_71 (Conv2D)              (None, 192, 12, 12)  147456      mixed7[0][0]
__________________________________________________________________________________________________
conv2d_75 (Conv2D)              (None, 192, 12, 12)  258048      activation_74[0][0]
__________________________________________________________________________________________________
batch_normalization_71 (BatchNo (None, 192, 12, 12)  576         conv2d_71[0][0]
__________________________________________________________________________________________________
batch_normalization_75 (BatchNo (None, 192, 12, 12)  576         conv2d_75[0][0]
__________________________________________________________________________________________________
240 activation_71 (Activation)      (None, 192, 12, 12)  0           batch_normalization_71[0][0]
__________________________________________________________________________________________________
activation_75 (Activation)      (None, 192, 12, 12)  0           batch_normalization_75[0][0]
__________________________________________________________________________________________________
conv2d_72 (Conv2D)              (None, 320, 5, 5)    552960      activation_71[0][0]
__________________________________________________________________________________________________
conv2d_76 (Conv2D)              (None, 192, 5, 5)    331776      activation_75[0][0]
__________________________________________________________________________________________________
batch_normalization_72 (BatchNo (None, 320, 5, 5)    960         conv2d_72[0][0]
__________________________________________________________________________________________________
batch_normalization_76 (BatchNo (None, 192, 5, 5)    576         conv2d_76[0][0]
__________________________________________________________________________________________________
activation_72 (Activation)      (None, 320, 5, 5)    0           batch_normalization_72[0][0]
__________________________________________________________________________________________________
activation_76 (Activation)      (None, 192, 5, 5)    0           batch_normalization_76[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 768, 5, 5)    0           mixed7[0][0]
__________________________________________________________________________________________________
mixed8 (Concatenate)            (None, 1280, 5, 5)   0           activation_72[0][0]
                                                                 activation_76[0][0]
                                                                 max_pooling2d_4[0][0]
__________________________________________________________________________________________________
250 conv2d_81 (Conv2D)              (None, 448, 5, 5)    573440      mixed8[0][0]
__________________________________________________________________________________________________
batch_normalization_81 (BatchNo (None, 448, 5, 5)    1344        conv2d_81[0][0]
__________________________________________________________________________________________________
activation_81 (Activation)      (None, 448, 5, 5)    0           batch_normalization_81[0][0]
__________________________________________________________________________________________________
conv2d_78 (Conv2D)              (None, 384, 5, 5)    491520      mixed8[0][0]
__________________________________________________________________________________________________
conv2d_82 (Conv2D)              (None, 384, 5, 5)    1548288     activation_81[0][0]
__________________________________________________________________________________________________
batch_normalization_78 (BatchNo (None, 384, 5, 5)    1152        conv2d_78[0][0]
__________________________________________________________________________________________________
batch_normalization_82 (BatchNo (None, 384, 5, 5)    1152        conv2d_82[0][0]
__________________________________________________________________________________________________
activation_78 (Activation)      (None, 384, 5, 5)    0           batch_normalization_78[0][0]
__________________________________________________________________________________________________
activation_82 (Activation)      (None, 384, 5, 5)    0           batch_normalization_82[0][0]
__________________________________________________________________________________________________
conv2d_79 (Conv2D)              (None, 384, 5, 5)    442368      activation_78[0][0]
__________________________________________________________________________________________________
260 conv2d_80 (Conv2D)              (None, 384, 5, 5)    442368      activation_78[0][0]
__________________________________________________________________________________________________
conv2d_83 (Conv2D)              (None, 384, 5, 5)    442368      activation_82[0][0]
__________________________________________________________________________________________________
conv2d_84 (Conv2D)              (None, 384, 5, 5)    442368      activation_82[0][0]
__________________________________________________________________________________________________
average_pooling2d_8 (AveragePoo (None, 1280, 5, 5)   0           mixed8[0][0]
__________________________________________________________________________________________________
conv2d_77 (Conv2D)              (None, 320, 5, 5)    409600      mixed8[0][0]
__________________________________________________________________________________________________
batch_normalization_79 (BatchNo (None, 384, 5, 5)    1152        conv2d_79[0][0]
__________________________________________________________________________________________________
batch_normalization_80 (BatchNo (None, 384, 5, 5)    1152        conv2d_80[0][0]
__________________________________________________________________________________________________
batch_normalization_83 (BatchNo (None, 384, 5, 5)    1152        conv2d_83[0][0]
__________________________________________________________________________________________________
batch_normalization_84 (BatchNo (None, 384, 5, 5)    1152        conv2d_84[0][0]
__________________________________________________________________________________________________
conv2d_85 (Conv2D)              (None, 192, 5, 5)    245760      average_pooling2d_8[0][0]
__________________________________________________________________________________________________
270 batch_normalization_77 (BatchNo (None, 320, 5, 5)    960         conv2d_77[0][0]
__________________________________________________________________________________________________
activation_79 (Activation)      (None, 384, 5, 5)    0           batch_normalization_79[0][0]
__________________________________________________________________________________________________
activation_80 (Activation)      (None, 384, 5, 5)    0           batch_normalization_80[0][0]
__________________________________________________________________________________________________
activation_83 (Activation)      (None, 384, 5, 5)    0           batch_normalization_83[0][0]
__________________________________________________________________________________________________
activation_84 (Activation)      (None, 384, 5, 5)    0           batch_normalization_84[0][0]
__________________________________________________________________________________________________
batch_normalization_85 (BatchNo (None, 192, 5, 5)    576         conv2d_85[0][0]
__________________________________________________________________________________________________
activation_77 (Activation)      (None, 320, 5, 5)    0           batch_normalization_77[0][0]
__________________________________________________________________________________________________
mixed9_0 (Concatenate)          (None, 768, 5, 5)    0           activation_79[0][0]
                                                                 activation_80[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 768, 5, 5)    0           activation_83[0][0]
                                                                 activation_84[0][0]
__________________________________________________________________________________________________
activation_85 (Activation)      (None, 192, 5, 5)    0           batch_normalization_85[0][0]
__________________________________________________________________________________________________
280 mixed9 (Concatenate)            (None, 2048, 5, 5)   0           activation_77[0][0]
                                                                 mixed9_0[0][0]
                                                                 concatenate_1[0][0]
                                                                 activation_85[0][0]
__________________________________________________________________________________________________
conv2d_90 (Conv2D)              (None, 448, 5, 5)    917504      mixed9[0][0]
__________________________________________________________________________________________________
batch_normalization_90 (BatchNo (None, 448, 5, 5)    1344        conv2d_90[0][0]
__________________________________________________________________________________________________
activation_90 (Activation)      (None, 448, 5, 5)    0           batch_normalization_90[0][0]
__________________________________________________________________________________________________
conv2d_87 (Conv2D)              (None, 384, 5, 5)    786432      mixed9[0][0]
__________________________________________________________________________________________________
conv2d_91 (Conv2D)              (None, 384, 5, 5)    1548288     activation_90[0][0]
__________________________________________________________________________________________________
batch_normalization_87 (BatchNo (None, 384, 5, 5)    1152        conv2d_87[0][0]
__________________________________________________________________________________________________
batch_normalization_91 (BatchNo (None, 384, 5, 5)    1152        conv2d_91[0][0]
__________________________________________________________________________________________________
activation_87 (Activation)      (None, 384, 5, 5)    0           batch_normalization_87[0][0]
__________________________________________________________________________________________________
activation_91 (Activation)      (None, 384, 5, 5)    0           batch_normalization_91[0][0]
__________________________________________________________________________________________________
290 conv2d_88 (Conv2D)              (None, 384, 5, 5)    442368      activation_87[0][0]
__________________________________________________________________________________________________
conv2d_89 (Conv2D)              (None, 384, 5, 5)    442368      activation_87[0][0]
__________________________________________________________________________________________________
conv2d_92 (Conv2D)              (None, 384, 5, 5)    442368      activation_91[0][0]
__________________________________________________________________________________________________
conv2d_93 (Conv2D)              (None, 384, 5, 5)    442368      activation_91[0][0]
__________________________________________________________________________________________________
average_pooling2d_9 (AveragePoo (None, 2048, 5, 5)   0           mixed9[0][0]
__________________________________________________________________________________________________
conv2d_86 (Conv2D)              (None, 320, 5, 5)    655360      mixed9[0][0]
__________________________________________________________________________________________________
batch_normalization_88 (BatchNo (None, 384, 5, 5)    1152        conv2d_88[0][0]
__________________________________________________________________________________________________
batch_normalization_89 (BatchNo (None, 384, 5, 5)    1152        conv2d_89[0][0]
__________________________________________________________________________________________________
batch_normalization_92 (BatchNo (None, 384, 5, 5)    1152        conv2d_92[0][0]
__________________________________________________________________________________________________
batch_normalization_93 (BatchNo (None, 384, 5, 5)    1152        conv2d_93[0][0]
__________________________________________________________________________________________________
300 conv2d_94 (Conv2D)              (None, 192, 5, 5)    393216      average_pooling2d_9[0][0]
__________________________________________________________________________________________________
batch_normalization_86 (BatchNo (None, 320, 5, 5)    960         conv2d_86[0][0]
__________________________________________________________________________________________________
activation_88 (Activation)      (None, 384, 5, 5)    0           batch_normalization_88[0][0]
__________________________________________________________________________________________________
activation_89 (Activation)      (None, 384, 5, 5)    0           batch_normalization_89[0][0]
__________________________________________________________________________________________________
activation_92 (Activation)      (None, 384, 5, 5)    0           batch_normalization_92[0][0]
__________________________________________________________________________________________________
activation_93 (Activation)      (None, 384, 5, 5)    0           batch_normalization_93[0][0]
__________________________________________________________________________________________________
batch_normalization_94 (BatchNo (None, 192, 5, 5)    576         conv2d_94[0][0]
__________________________________________________________________________________________________
activation_86 (Activation)      (None, 320, 5, 5)    0           batch_normalization_86[0][0]
__________________________________________________________________________________________________
mixed9_1 (Concatenate)          (None, 768, 5, 5)    0           activation_88[0][0]
                                                                 activation_89[0][0]
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 768, 5, 5)    0           activation_92[0][0]
                                                                 activation_93[0][0]
__________________________________________________________________________________________________
310 activation_94 (Activation)      (None, 192, 5, 5)    0           batch_normalization_94[0][0]
__________________________________________________________________________________________________
mixed10 (Concatenate)           (None, 2048, 5, 5)   0           activation_86[0][0]
                                                                 mixed9_1[0][0]
                                                                 concatenate_2[0][0]
                                                                 activation_94[0][0]
__________________________________________________________________________________________________
sequential_1 (Sequential)       (None, 10)           13110026    mixed10[0][0]
==================================================================================================
Total params: 34,912,810
Trainable params: 32,819,818
Non-trainable params: 2,092,992
__________________________________________________________________________________________________
"""
