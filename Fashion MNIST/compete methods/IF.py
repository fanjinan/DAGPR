# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:54:07 2019

@author: FJA
"""

print(__doc__)
from tensorflow import keras
import numpy as np
from sklearn.ensemble import IsolationForest
from keras.datasets import mnist
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def read_mnist():
  fashion_mnist = keras.datasets.fashion_mnist
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  # image_size = x_train.shape[1]
  # original_dim = image_size * image_size
  x_train = np.reshape(x_train, [-1, 28,28,1])
  x_test = np.reshape(x_test, [-1, 28,28,1])
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255
  
  # test_data = (x_test, y_test)
  print(x_train.shape, 'reshaped x_train shape')
  print(x_test.shape, 'reshaped x_test shape')
  
  data = []
  for i in range(10):
      data.append(x_train[y_train == i])
      print('class{}'.format(i), data[i].shape)
  data = np.array(data)
  return (data, x_test, y_test)

def performance(label,label_pred,num): 
  FP,TP,TN,FN = 0,0,0,0
  for i in range(np.size(label)):
    if label_pred[i]==-1:
      if label[i] == num:
        FP += 1
      else:
        TP += 1
    else:
      if label[i] == num:
        TN += 1
      else:
        FN += 1
  TPR = TP/(TP+FN)
  TNR = FP/(FP+TN)
  F1  = 2*TP/(2*TP+FP+FN)
  print(F1)
  return TPR,TNR,F1

if __name__=='__main__':
  inlier_num = 9
  outlier_ratio = 0.05
  nomal = inlier_num
  data, x_test, y_test = read_mnist()
  anormal = list(range(10))
  anormal.remove(nomal)
  aa = data[nomal]
  o_num = (aa.shape[0]/(1-outlier_ratio)-aa.shape[0])/9  
  #cut = np.shape(aa)[0]
  label = nomal*np.ones((np.shape(aa)[0],1))
  for i in anormal:
    _ = data[i]
    index = np.random.choice(np.shape(_)[0],np.int(o_num))
    aa = np.vstack((aa,_[index]))
    label = np.vstack((label,i*np.ones((np.int(o_num),1))))
  data = aa
  data = np.reshape(data,(-1,784))

  clf = IsolationForest( max_samples=10,
                        n_estimators=10, contamination=outlier_ratio)
  clf.fit(data)
  label_pred = clf.predict(data)
  TPR,TNR,F1 = performance(label,label_pred,nomal)
  score = -clf.decision_function(data)
  fpr, tpr, thresholds = roc_curve(np.reshape(label,[np.shape(data)[0],1]), score, pos_label=inlier_num)
  print('auc=')
  print(1-auc(fpr, tpr))
  
  