# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 08:49:42 2019

@author: FJA
"""

from tensorflow import keras
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow
import copy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF,DotProduct, WhiteKernel,ExpSineSquared,RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from keras.datasets import mnist

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W,pad):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding=pad)

def deconv2d(x, W,pad,kernel_size,strides=(2,2)):
  return tf.layers.conv2d_transpose(x, W, strides=strides, padding=pad,kernel_size=kernel_size)

def creat_net(x):
  encode_layer1_w = weight_variable([5, 5, 1, 32])
  encode_layer1_b = bias_variable([32])
  encode_layer2_w = weight_variable([5, 5, 32, 64])
  encode_layer2_b = bias_variable([64])
  encode_layer3_w = weight_variable([3, 3, 64, 128])
  encode_layer3_b = bias_variable([128])
  
  encode_out_w1 = tf.Variable(tf.truncated_normal([1152,10], stddev=0.1))
  encode_out_b1 = tf.Variable(tf.truncated_normal([10],stddev=0.1))
  decode_in_w1 = tf.Variable(tf.truncated_normal([10,1152],stddev=0.1))
  decode_in_b1 = tf.Variable(tf.truncated_normal([1152],stddev=0.1))
  
  decode_layer1_w = weight_variable([3, 3, 64,128 ])
  decode_layer1_b = bias_variable([64])
  decode_layer2_w = weight_variable([5, 5, 32,64 ])
  decode_layer2_b = bias_variable([32])
  decode_layer3_w = weight_variable([5, 5, 1,32 ])
  decode_layer3_b = bias_variable([1])

  encode_layer1_out = tf.nn.relu(conv2d(x, encode_layer1_w,'SAME') + encode_layer1_b)
  encode_layer2_out = tf.nn.relu(conv2d(encode_layer1_out, encode_layer2_w,'SAME') + encode_layer2_b)
  encode_layer3_out = tf.nn.relu(conv2d(encode_layer2_out, encode_layer3_w,'VALID') + encode_layer3_b)
#  encode_out = tf.nn.leaky_relu(tf.add(tf.matmul(encode_layer3_out,encode_out_w),encode_out_b))
  a = tf.reshape(encode_layer3_out,[-1,3*3*128])
  encode_out_nol2 = tf.add(tf.matmul(a,encode_out_w1),encode_out_b1)
  
  xx = tf.square(encode_out_nol2)
  test1 = tf.sqrt(tf.reduce_sum(xx, 1, keepdims=True))
  c = tf.constant(1,dtype=tf.float32)
  encode_out = tf.multiply(c,tf.divide(encode_out_nol2,test1))
  
#  print(xx)
#  print(test1)
  decode_in = tf.nn.relu(tf.add(tf.matmul(encode_out,decode_in_w1),decode_in_b1))
  b = tf.reshape(decode_in,[-1,3,3,128])
  
  decode_layer1_out = tf.nn.relu(deconv2d(b, 64,'VALID',(3,3)) + decode_layer1_b)
  decode_layer2_out = tf.nn.relu(deconv2d(decode_layer1_out, 32,'SAME',(5,5)) + decode_layer2_b)
  decode_layer3_out = tf.nn.relu(deconv2d(decode_layer2_out, 1,'SAME',(5,5)) + decode_layer3_b)
  decode_out = decode_layer3_out
  return decode_out,encode_out

def gradient_visualization(loss,autoencoder_input):
  with tf.name_scope('gradients'):
    a = tf.reduce_sum(tf.abs(tf.gradients(loss,autoencoder_input)))
    tf.summary.scalar('gradient',a)

def matchlabel(x):
  Acc = []
  for j in range(10):
    acc = 0
    for i in range(len(x)):
      if x[i]-j == 0:
        acc += 1
    Acc.append(acc)
  return np.argmax(Acc),np.max(Acc)

def PLOT(data,reconstruct_data):
  n = 100
  rand_index = np.random.choice(data.shape[0],size=n)
  rand_x = data[rand_index]
  rand_y = reconstruct_data[rand_index]
  plt.figure(figsize=(20, 4))
  for i in range(n):
    ax = plt.subplot(10, 10, i+1)
    plt.imshow(rand_x[i].reshape(28,28))
    plt.gray()
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.figure(figsize=(20, 4))
  for i in range(n):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow(rand_y[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def cluster_match(label,sess,encode_output,kmeans):
  ori_label = label
  label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_0 = [],[],[],[],[],[],[],[],[],[]
  for i in range(1426):
    if ori_label[i] == 0:
      label_0.append(i)
    elif ori_label[i] == 1:
      label_1.append(i)
    elif ori_label[i] == 2:
      label_2.append(i)
    elif ori_label[i] == 3:
      label_3.append(i)
    elif ori_label[i] == 4:
      label_4.append(i)
    elif ori_label[i] == 5:
      label_5.append(i)
    elif ori_label[i] == 6:
      label_6.append(i)
    elif ori_label[i] == 7:
      label_7.append(i)
    elif ori_label[i] == 8:
      label_8.append(i)
    elif ori_label[i] == 9:
      label_9.append(i)  
  compress_0 = sess.run(encode_output, feed_dict={autoencoder_input:data[label_0]})
  compress_1 = sess.run(encode_output, feed_dict={autoencoder_input:data[label_1]})
  compress_2 = sess.run(encode_output, feed_dict={autoencoder_input:data[label_2]})
  compress_3 = sess.run(encode_output, feed_dict={autoencoder_input:data[label_3]})
  compress_4 = sess.run(encode_output, feed_dict={autoencoder_input:data[label_4]})
  compress_5 = sess.run(encode_output, feed_dict={autoencoder_input:data[label_5]})
  compress_6 = sess.run(encode_output, feed_dict={autoencoder_input:data[label_6]})
  compress_7 = sess.run(encode_output, feed_dict={autoencoder_input:data[label_7]})
  compress_8 = sess.run(encode_output, feed_dict={autoencoder_input:data[label_8]})
  compress_9 = sess.run(encode_output, feed_dict={autoencoder_input:data[label_9]})  
  pre_0 = kmeans.predict(compress_0)
  pre_1 = kmeans.predict(compress_1)
  pre_2 = kmeans.predict(compress_2)
  pre_3 = kmeans.predict(compress_3)
  pre_4 = kmeans.predict(compress_4)
  pre_5 = kmeans.predict(compress_5)
  pre_6 = kmeans.predict(compress_6)
  pre_7 = kmeans.predict(compress_7)
  pre_8 = kmeans.predict(compress_8)
  pre_9 = kmeans.predict(compress_9)
  number,acc = [],[]
  a,b = matchlabel(pre_0)
  number.append(a)
  acc.append(b)
  a,b = matchlabel(pre_1)
  number.append(a)
  acc.append(b)
  a,b = matchlabel(pre_2)
  number.append(a)
  acc.append(b)
  a,b = matchlabel(pre_3)
  number.append(a)
  acc.append(b)
  a,b = matchlabel(pre_4)
  number.append(a)
  acc.append(b)
  a,b = matchlabel(pre_5)
  number.append(a)
  acc.append(b)
  a,b = matchlabel(pre_6)
  number.append(a)
  acc.append(b)
  a,b = matchlabel(pre_7)
  number.append(a)
  acc.append(b)
  a,b = matchlabel(pre_8)
  number.append(a)
  acc.append(b)
  a,b = matchlabel(pre_9)
  number.append(a)
  acc.append(b)
  print(number)
  print(acc)
  print(np.sum(acc)/1426)

def data_filter(d,l,q,outlier_ratio):
  l0 = []
  D = []
  L = []
  ano_num = []
  inl_num = 0
  for i in range(d.shape[0]):
    a = l[i,:]
    b = np.argmax(a)
    l0.append(b)
  inlier_num = q 
  for j in range(10):
    if j == inlier_num:
      ii = 0
      for i in range(d.shape[0]):
        if l0[i] == inlier_num:
          ii += 1
      inl_num=ii
  anomaly_proportion = (inl_num/(1-outlier_ratio)-inl_num)/9    
  for j in range(10):
    if j == inlier_num:
      ii = 0
      for i in range(d.shape[0]):
        if l0[i] == inlier_num:
          D.append(d[i,:])
          L.append(l0[i])
          ii += 1
      ano_num.append(ii)
    else:
      _,__ = [],[]
      for i in range(d.shape[0]):
        if l0[i] == j:
          _.append(d[i,:])
          __.append(l0[i])
      for k in range(int(anomaly_proportion)):
        D.append(_[k])
        L.append(__[k])
      ano_num.append(int(anomaly_proportion))
  D = np.reshape(D,(-1,784))
  L = np.reshape(L,(-1,1))
  _rand = np.arange(D.shape[0])
  np.random.shuffle(_rand)
  D2 = D[_rand]
  L2 = L[_rand]
  return D2,L2,D,L,ano_num

def test(_test_num,_test_data):
  inlier_num = np.argmax(_test_num)
  num = 0
  print('inlier number:')
  print(inlier_num)
  for i in range(10):
    if i==inlier_num:
      reconstruct_loss = sess.run(loss, feed_dict={autoencoder_input:_test_data[num:num + _test_num[i],:]})
      print(reconstruct_loss/_test_num[i])
    else:
      reconstruct_loss = sess.run(loss, feed_dict={autoencoder_input:_test_data[num:num + _test_num[i],:]})
      print(reconstruct_loss/_test_num[i])
    num += _test_num[i]

def GPC(X,Y,Z):
  m = gpflow.models.SVGP(
      X, Y, kern=gpflow.kernels.Matern32(1)+gpflow.kernels.White(1,variance=0.1),
      likelihood=gpflow.likelihoods.MultiClass(2),
      Z=X[::5].copy(), num_latent=2, whiten=True, q_diag=True)
  m.kern.kernels[1].variance.trainable = False
  m.feature.trainable = False
  m.as_pandas_table()
  print(m)
  opt = gpflow.train.ScipyOptimizer()
  opt.minimize(m)
  mu, var = m.predict_f(Z)
  p, _ = m.predict_y(Z)
  return mu,var,p

def rechoose(p,GP_test):
  th1 = 0.98
  th2 = 0.5
  _p_1 = np.sort(p[:,0])
  _p_2 = np.sort(p[:,1])
  _p1_index = []
  for i in range(np.size(_p_1)):
    if _p_1[i] >= th1:
      _p1_index.append(i)
      
  _p2_index = []    
  for i in range(np.size(_p_2)):
    if _p_2[i]>th2:
      _p2_index.append(i)
  d1 = GP_test[_p1_index]
  d2 = GP_test[_p2_index]
  d  = np.float64(np.vstack((d1,d2)))
  l1 = np.zeros((np.shape(d1)[0],1))
  l2 = np.ones((np.shape(d2)[0],1))
  l  = np.float64(np.vstack((l1,l2)))
  return d,l

def rechoose2(D,m,v,l):
  d = D[np.argsort(m,axis=None)]
  rl= l[np.argsort(m,axis=None)]
  d1= d[0:np.int(0.2*np.shape(D)[0]),:]
  l1= np.zeros([np.shape(d1)[0],1])
  
  d2= d[np.shape(D)[0]-np.int(0.055*np.shape(D)[0]):,:]
  l2= np.ones([np.shape(d2)[0],1])
  
  _1 = np.arange(np.shape(d1)[0])
  _2 = np.arange(np.shape(d2)[0])
  np.random.shuffle(_1)
  np.random.shuffle(_2)
  _1 = _1[0:np.int(0.8*np.size(_1))]
  _2 = _2[0:np.int(0.9*np.size(_2))]
  
  d1 = d1[_1]
  l1 = l1[_1]
  d2 = d2[_2]
  l2 = l2[_2]
  return d,d1,d2,l1,l2,rl

def rechoose4(D,m,v,l):
#  D,m,v,l = GP_test,mu2,var2,label
  a = np.reshape(np.arange(np.shape(D)[0]),[np.shape(D)[0],1])
  m_index_1 = np.argsort(m,axis=None)
  a2 = a[m_index_1]
  sore1 = np.zeros(np.shape(D)[0])
  for i in range(np.shape(D)[0]):
    sore1[a2[i]] = i
  
  a = np.reshape(np.arange(np.shape(D)[0]),[np.shape(D)[0],1])
  m_index_1 = np.argsort(v,axis=None)
  a2 = a[m_index_1]
  sore2 = np.zeros(np.shape(D)[0])
  for i in range(np.shape(D)[0]):
    sore2[a2[i]] = i
  
  a = np.reshape(np.arange(np.shape(D)[0]),[np.shape(D)[0],1])
  m_index_1 = np.argsort(-m,axis=None)
  a2 = a[m_index_1]
  sore3 = np.zeros(np.shape(D)[0])
  for i in range(np.shape(D)[0]):
    sore3[a2[i]] = i
  
  sore = sore1+sore2
  _1 = np.argsort(sore)
  D_new_1 = D[_1]
  D_new_1 = D_new_1[0:1266,:]
  rl1 = l[_1]
  rl1 = rl1[0:1266,:]
  l1 = np.zeros([1266,1])
  
  sore = sore3+sore2
  _1 = np.argsort(sore)
  D_new_2 = D[_1]
  D_new_2 = D_new_2[0:355,:]
  rl2 = l[_1]
  rl2 = rl2[0:355,:]
  l2 = np.ones([355,1])
  return D_new_1,D_new_2,l1,l2

def rechoose3(D,m,v,l):
  d = D[np.argsort(mu,axis=None)]
  rl= l[np.argsort(mu,axis=None)]
  d1= d[0:np.int(0.3*np.shape(D)[0]),:]
  l1= np.zeros([np.shape(d1)[0],1])
  
  _ = []
  for i in range(np.shape(d)[0]):
    if m[i,0]>0.9:
      _.append(i)
  d2= d[_]
  l2= np.ones([np.shape(d2)[0],1])
  
  _1 = np.arange(np.shape(d1)[0])
  _2 = np.arange(np.shape(d2)[0])
  np.random.shuffle(_1)
  np.random.shuffle(_2)
  _1 = _1[0:np.int(0.8*np.size(_1))]
  _2 = _2
  
  d1 = d1[_1]
  l1 = l1[_1]
  d2 = d2[_2]
  l2 = l2[_2]
  return d,d1,d2,l1,l2,rl

def GPC2(X,Y,Z):
  Y = np.squeeze(Y)
  kernels = 1.0*RBF(length_scale=1.0)
  clf = GaussianProcessClassifier(kernel = kernels,warm_start=True).fit(X,Y)
  p = clf.predict(Z)
  return p

def GPR(X,Y,Z):
  k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
  k2 = 2.0**2 * RBF(length_scale=100.0) \
      * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                       periodicity_bounds="fixed")  # seasonal component
  # medium term irregularities
  k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
  k4 = 0.1**2 * RBF(length_scale=0.1) \
      + WhiteKernel(noise_level=0.1**2,
                    noise_level_bounds=(1e-3, np.inf))  # noise terms
  kernel = k4
  
  gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                                normalize_y=True)
  gp.fit(X, Y)
  
  mu,var = gp.predict(Z, return_std=True)
  return mu,var

def GPR2(X,Y,Xtest):
  likelihood= gpflow.likelihoods.Gaussian()
  kern1=gpflow.kernels.RBF(input_dim=10,lengthscales=0.3)
  kern2=gpflow.kernels.Linear(input_dim=10)
  kern3=gpflow.kernels.Constant(input_dim=10)
  m = gpflow.models.VGP(X, Y,kern=kern1+kern2+kern3,likelihood=likelihood)
  #m = gpflow.models.VGP(X, Y,kern=kern1+kern3,likelihood=likelihood)
  m.kern.kernels[0].lengthscales.trainable = False
  m.likelihood.variance.trainable = False
  m.kern.kernels[0].variance.trainable = False
  m.kern.kernels[1].variance.trainable = False
  m.kern.kernels[2].variance.trainable = False
#  print(m)
  gpflow.train.ScipyOptimizer().minimize(m)
  mu, var = m.predict_y(Xtest)
#  plt.plot(Xtest, mu+2*np.sqrt(var), '--', lw=2)
#  plt.plot(Xtest, mu-2*np.sqrt(var), 'kx', lw=2)
#  plt.plot(Xtest, mu, 'kx', mew=2)
  return mu,var


def result_of_method(a,num,th):
  rl = a[:,0]
  pl = a[:,1]
  FP,TP,TN,FN = 0,0,0,0
  for i in range(np.size(rl)):
    if pl[i]>=th:
      if rl[i] == num:
        FP += 1
      else:
        TP += 1
    else:
      if rl[i] == num:
        TN += 1
      else:
        FN += 1
#  print(FP,TP,TN,FN)
  TPR = TP/(TP+FN)
  TNR = FP/(FP+TN)
  F1  = 2*TP/(2*TP+FP+FN)
  return TPR,TNR,F1

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

def plot_roc(fpr, tpr, anomal):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.title('normal vs number {}'.format(anomal))
    plt.legend(loc="best")


if __name__ == '__main__':
  inlier_num = 5
  outlier_ratio = 0.1 #anomaly rate
  
#  data,label,test_data,test_label = mnist_data_import()
#  data = np.nan_to_num(data)
#  data,label,_data,_label,_num = data_filter(data,label,inlier_num,outlier_ratio)
#  _data = np.reshape(_data,[-1,28,28,1])
#  data = np.reshape(data,[-1,28,28,1])
#  test_data = np.nan_to_num(test_data)
#  test_data,test_label,_test_data,_test_label,_test_num = data_filter(test_data,test_label,inlier_num,outlier_ratio)
#  test_data = np.reshape(test_data,[-1,28,28,1])
#  _test_data = np.reshape(_test_data,[-1,28,28,1])
#  

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
  
  gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.26)
  config=tf.ConfigProto(gpu_options=gpu_options)
  sess = tf.Session(config=config)
  autoencoder_input = tf.placeholder(tf.float32,[None,28,28,1])
  autoencoder_output,encode_output = creat_net(autoencoder_input)
  with tf.name_scope('loss'):
    loss = tf.nn.l2_loss(autoencoder_output-autoencoder_input)
    tf.summary.scalar('loss',loss)
  gradient_visualization(loss,autoencoder_input)
  train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
  init = tf.global_variables_initializer()
  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter("logs/", sess.graph)  
  sess.run(init)
#  print(sess.run(encode_out_nol2, feed_dict={autoencoder_input:batch_xs}))
#  print(sess.run(test1, feed_dict={autoencoder_input:batch_xs}))
#  print(sess.run(encode_output, feed_dict={autoencoder_input:batch_xs}))
#  training
  tr_a = copy.deepcopy(data)
  print('start training:')
  for episode in range(400):
    dex = np.arange(np.shape(tr_a)[0])
    np.random.shuffle(dex)
    tr_a =tr_a[dex]
    for i in range(np.int(np.shape(tr_a)[0]/256)):
      #batch_size = 128
      sess.run(train_step, feed_dict={autoencoder_input:tr_a[i*256:i*256+256]})
    #visualization
    rs=sess.run(merged,feed_dict={autoencoder_input:tr_a[i*256:i*256+256]})
    writer.add_summary(rs, episode)
    print('epsiod {}'.format(episode))
#    rs2=sess.run(loss,feed_dict={autoencoder_input:tr_a[i*256:i*256+256]})
#    print('error: {}'.format(rs2))
  print('training end')
#  test(_test_num,_test_data)
#  print(_test_num)
  compress_data = sess.run(encode_output, feed_dict={autoencoder_input:data})
  
  
  
  recon_loss = []
  for i in range(np.shape(data)[0]):
    recon_loss.append(sess.run(loss,feed_dict={autoencoder_input:data[i:i+1,:,:,:]}))
  recon_loss_index = np.argsort(recon_loss)
  
  combine_data = np.hstack((compress_data,np.reshape(recon_loss,(-1,1))))
  center = np.sum(combine_data,axis=0)/np.shape(compress_data)[0]
  distance = np.sum(np.square(combine_data-center),axis=1)
  dis_index = np.argsort(distance)
  
#  np.savetxt('compress_data.txt',compress_data)
#  np.savetxt('label.txt',label)
#  label = np.loadtxt('label.txt')
#  compress_data = np.loadtxt('compress_data.txt')
  
  index = np.random.choice(2000,800)
  GP_init_data1 = compress_data[dis_index[index]]
  GP_init_data2 = compress_data[dis_index[np.size(dis_index)-400:np.size(dis_index)],:]
  GP_init_data = np.float64(np.vstack((GP_init_data1,GP_init_data2)))
  GP_init_label1 = np.zeros([800,1])
  GP_init_label2 = np.ones([400,1])
  GP_init_label = np.float64(np.vstack((GP_init_label1,GP_init_label2)))
  
  label1 = label[dis_index[index]]
  label2 = label[dis_index[np.size(dis_index)-400:np.size(dis_index)]]
  
  #\D1\E9֤\C5\C5\D4\DAǰ\C3\E6\B5\C4\CAǷ\F1\B6\BC\D5\FDȷ
#  label2 = label[recon_loss_index]  
#  GP_init_data1 = compress_data[recon_loss_index[0:np.int(np.size(recon_loss)*0.2)]]
#  GP_init_data2 = compress_data[recon_loss_index[np.size(recon_loss)-np.int(np.size(recon_loss)*0.05):np.size(recon_loss)]]
#  GP_init_data = np.float64(np.vstack((GP_init_data1,GP_init_data2)))
#  GP_init_label1 = np.zeros([np.int(np.size(recon_loss)*0.2),1])
#  GP_init_label2 = np.ones([np.int(np.size(recon_loss)*0.05),1])
#  GP_init_label = np.float64(np.vstack((GP_init_label1,GP_init_label2)))
#  
#  real_label1 = label[recon_loss_index[0:np.int(np.size(recon_loss)*0.2)]]
#  real_label2 = label[recon_loss_index[np.size(recon_loss)-np.int(np.size(recon_loss)*0.05):np.size(recon_loss)]]
#  real_label = np.float64(np.vstack((real_label1,real_label2)))
  
  GP_test = sess.run(encode_output, feed_dict={autoencoder_input:data})
  mu,var = GPR2(GP_init_data,GP_init_label,compress_data)
  
#  np.savetxt('GP_test.txt',GP_test)
#  np.savetxt('mu.txt',mu)
#  np.savetxt('var.txt',var)
#  GP_test = np.loadtxt('GP_test.txt')
#  mu = np.loadtxt('mu.txt')
#  var = np.loadtxt('var.txt')
  
  a = np.hstack((np.reshape(label,[np.shape(data)[0],1]),100*np.reshape(mu,[np.shape(data)[0],1]),np.reshape(var,[np.shape(data)[0],1])))
  f1 = []
  tprm,tnrm = [],[]
  for j in np.linspace(0,99,100):
    TPR,TNR,F1 = result_of_method(a,inlier_num,j)
    f1.append(F1)
    tprm.append(TPR)
    tnrm.append(TNR)
  _ = np.argmax(f1)
  print(_)
  print(f1[_],tprm[_],tnrm[_])
  #d,l = rechoose(p,GP_test[:,0])
  
  fpr, tpr, thresholds = roc_curve(np.reshape(label,[np.shape(data)[0],1]), mu, pos_label=inlier_num)
  print(1-auc(fpr, tpr))
  
  for k in range(20):
    d,d1,d2,l1,l2,rl = rechoose2(compress_data,mu,var,label)
    mu,var = GPR2(np.float64(np.vstack((d1,d2))),np.float64(np.vstack((l1,l2))),compress_data)
    a = np.hstack((np.reshape(label,[np.shape(data)[0],1]),100*np.reshape(mu,[np.shape(data)[0],1]),np.reshape(var,[np.shape(data)[0],1])))
    f1 = []
    tprm,tnrm = [],[]
    for j in np.linspace(0,99,100):
      TPR,TNR,F1 = result_of_method(a,inlier_num,j)
      f1.append(F1)
      tprm.append(TPR)
      tnrm.append(TNR)
    _ = np.argmax(f1)
    print(_)
    print(f1[_],tprm[_],tnrm[_])
    fpr, tpr, thresholds = roc_curve(np.reshape(label,[np.shape(data)[0],1]), mu, pos_label=inlier_num)
    print(1-auc(fpr, tpr))
  
  
  print('AUC of DCAE:')
  recon_loss = []
  for i in range(np.shape(data)[0]):
    recon_loss.append(sess.run(loss,feed_dict={autoencoder_input:data[i:i+1,:,:,:]}))
  recon_loss_index = np.argsort(recon_loss)
  
  score = np.reshape(recon_loss,[np.shape(data)[0],1])
  fpr, tpr, thresholds = roc_curve(np.reshape(label,[np.shape(data)[0],1]), score, pos_label=inlier_num)
  print('auc=')
  print(1-auc(fpr, tpr))
  
  
