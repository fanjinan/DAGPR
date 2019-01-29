print(__doc__)

import numpy as np
from sklearn.ensemble import IsolationForest
from keras.datasets import cifar10
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

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
def cifar10_read():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train = np.reshape(x_train, [-1, 32,32,3])
  x_test = np.reshape(x_test, [-1, 32,32,3])
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255
  a = np.vstack((x_train,x_test))
  b = np.vstack((y_train,y_test))
  b = np.squeeze(b)
  data = []
  label= []
  for i in range(10):
    data.append(a[b==i])
  for i in range(10):
    label.append(i*np.ones(6000))
#  data = np.array(data)
#  data = np.reshape(data,[-1,32,32,3])
  label= np.array(label)
  label= np.reshape(label,[60000,-1])
  return data,label

if __name__=='__main__':
  inlier_num = 9
  outlier_ratio = 0.2
  nomal = inlier_num
  data,label = cifar10_read()
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
  data = np.reshape(data,(-1,3072))

  clf = IsolationForest(behaviour='new', max_samples=10000,
                        n_estimators=500, contamination=outlier_ratio)
  clf.fit(data)
#  label_pred = clf.predict(data)
#  TPR,TNR,F1 = performance(label,label_pred,nomal)
  score = -clf.decision_function(data)
  fpr, tpr, thresholds = roc_curve(np.reshape(label,[np.shape(data)[0],1]), score, pos_label=inlier_num)
  print('auc=')
  print(1-auc(fpr, tpr))
  
