# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:24:16 2019

@author: FJA
"""

from sklearn import svm
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def STL10_read():
  m1 = loadmat("train")
  X1 = np.array(m1['X'])
  Y1 = np.array(m1['y'])
  X1 = np.reshape(X1,[-1,96,96,3],order='F')
  for i in range(np.shape(Y1)[0]):
    if Y1[i,0] == 10:
      Y1[i,0] = 0
  
  m2 = loadmat("test")
  X2 = np.array(m2['X'])
  Y2 = np.array(m2['y'])
  X2 = np.reshape(X2,[-1,96,96,3],order='F')
  for i in range(np.shape(Y2)[0]):
    if Y2[i,0] == 10:
      Y2[i,0] = 0
  
  x_train = np.vstack((X1,X2))
  y_train = np.vstack((Y1,Y2))
  x_train = x_train.astype('float32') / 255
  y_train = np.squeeze(y_train)
  data = []
  label= []
  for i in range(10):
    data.append(x_train[y_train==i])
#    print(np.shape(data[i])[0])
  for i in range(10):
    label.append(i*np.ones(1300))
#  data = np.array(data)
#  data = np.reshape(data,[-1,32,32,3])
  label= np.array(label)
  label= np.reshape(label,[13000,-1])
  return data,label

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
  #print(F1)
  return TPR,TNR,F1

if __name__ == '__main__':
  inlier_num = 9
  outlier_ratio = 0.05
  nomal = inlier_num
  data,label = STL10_read()
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
  
  data = np.reshape(data,(-1,96*96*3))
#  a=[]
#  f=[]
#  for mu in [0.01,0.1]:
#    for gama in [2**(-10),2**(-9),2**(-8),2**(-7),2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),2**(-1)]:
#      clf = svm.OneClassSVM(nu=mu, kernel="rbf", gamma=gama)
#      clf.fit(data)
#      label_pred = clf.predict(data)
#      TPR,TNR,F1 = performance(label,label_pred,nomal)
#      f.append(F1)
#      score = -clf.decision_function(data)
#      fpr, tpr, thresholds = roc_curve(np.reshape(label,[np.shape(data)[0],1]), score, pos_label=inlier_num)
#      a.append(1-auc(fpr, tpr))
#  index = np.argmax(a)
#  print(index)
#  print(a[index])
#  print(f[index])
  
  
  
  mu=0.1
  gama=2**(-9)
  clf = svm.OneClassSVM(nu=mu, kernel="rbf", gamma=gama)
  clf.fit(data)
  score = -clf.decision_function(data)
  fpr, tpr, thresholds = roc_curve(np.reshape(label,[np.shape(data)[0],1]), score, pos_label=inlier_num)
  print(1-auc(fpr, tpr))
  