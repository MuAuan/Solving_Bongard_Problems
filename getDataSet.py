# -*- coding: utf-8 -*-
"""
このコードは一部を除いて、MATHGRAM　by　k3nt0 (id:ket-30)さんの
以下のサイトのものを利用しています。
http://www.mathgram.xyz/entry/chainer/bake/part5
"""
from __future__ import print_function
from collections import defaultdict

from PIL import Image
from six.moves import range
import keras.backend as K

from keras.utils.generic_utils import Progbar
import numpy as np
import keras

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from keras.preprocessing import image
import sys
#import cv2
import os


np.random.seed(1337)

K.set_image_data_format('channels_first')

#その１　------データセット作成------

#フォルダは整数で名前が付いています。
def getDataSet(img_rows,img_cols):
    #リストの作成
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(18,20):
        path = "./data0/BG_"
        path1 = "./data0/BG_"
        s=0.2
        if i == 18:
            #othersは600枚用意します。テスト用には60枚
            cutNum = 6 #99 #69
            cutNum2 = 5   #int(cutNum*s) #57
        else:
            print(i)
            #主要キャラたちは480枚ずつ。テスト用には40枚
            cutNum = 6  #100 #60 #600
            cutNum2 = 5  #int(cutNum*s) #48 #540
        """
        elif i == 1:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 102 #700
            cutNum2 = int(cutNum*s) #630
        elif i==2:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 117 #700
            cutNum2 = int(cutNum*s) #630
 
        elif i==3:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 102 #700
            cutNum2 = int(cutNum*s) #630
 
        elif i==4:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 100 #700
            cutNum2 = int(cutNum*s) #630
        elif i==5:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 109
            cutNum2 = int(cutNum*s)
 
        elif i==6:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 105
            cutNum2 = int(cutNum*s)
 
        elif i==7:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 106
            cutNum2 = int(cutNum*s)
 
        elif i==8:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 98
            cutNum2 = int(cutNum*s)
 
        elif i==9:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 107
            cutNum2 = int(cutNum*s)

        elif i==10:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 35 #600
            cutNum2 = int(cutNum*s) #540

        elif i==11:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 19 #600
            cutNum2 = int(cutNum*s) #540

        elif i==12:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 31 #600
            cutNum2 = int(cutNum*s) #540

        elif i==13:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 43 #600
            cutNum2 = int(cutNum*s) #540

        elif i==14:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 32 #600
            cutNum2 = int(cutNum*s) #540

        elif i==15:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 25 #600
            cutNum2 = int(cutNum*s) #540

        elif i==16:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 41 #600
            cutNum2 = int(cutNum*s) #540

        elif i==17:
            #乃木坂メンバーは700枚ずつ。テスト用には70枚
            cutNum = 43 #600
            cutNum2 = int(cutNum*s) #540
        
        
        else:
            print(i)
            #主要キャラたちは480枚ずつ。テスト用には40枚
            cutNum = 20 #60 #600
            cutNum2 = int(cutNum*s) #48 #540
        """
        #Test data を別のDir中のファイルから読み込む場合    
        if s==0:
            imgList1 = os.listdir(path1+str(i))
            print(imgList1)
            imgNum1 = len(imgList1)
            for j in range(imgNum1):
                #imgSrc = cv2.imread(path+str(i)+"/"+imgList[j])
                #target_size=(img_rows,img_cols)のサイズのimage
                img = image.load_img(path1+str(i)+"/"+imgList1[j], target_size=(img_rows,img_cols))
                imgSrc = image.img_to_array(img)
                        
                if imgSrc is None:continue
                X_train.append(imgSrc)
                y_train.append(i)

        #sの値の割合でTrainデータとTestデータをファイルから読み込む         
        imgList = os.listdir(path+str(i))
        print(imgList)
        imgNum = len(imgList)
        for j in range(cutNum):
            #imgSrc = cv2.imread(path+str(i)+"/"+imgList[j])
            #target_size=(img_rows,img_cols)のサイズのimage
            img = image.load_img(path+str(i)+"/"+imgList[j], target_size=(img_rows,img_cols))
            imgSrc = image.img_to_array(img)
                        
            if imgSrc is None:continue
            if j < cutNum2:
                X_train.append(imgSrc)
                y_train.append(i)
            else:
                X_test.append(imgSrc)
                y_test.append(i)
    print(len(X_train),len(y_train),len(X_test),len(y_test))

    return X_train,y_train,X_test,y_test
