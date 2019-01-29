# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:14:09 2018

@author: FJA
"""

from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data as input_data
import gpflow
import copy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF,DotProduct, WhiteKernel,ExpSineSquared,RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from keras.datasets import mnist
from keras.datasets import cifar10

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
  encode_layer1_w = weight_variable([5, 5, 3, 32])
  encode_layer1_b = bias_variable([32])
  encode_layer2_w = weight_variable([5, 5, 32, 64])
  encode_layer2_b = bias_variable([64])
  encode_layer3_w = weight_variable([3, 3, 64, 128])
  encode_layer3_b = bias_variable([128])
  
  encode_out_w1 = tf.Variable(tf.truncated_normal([2048,10], stddev=0.1))
  encode_out_b1 = tf.Variable(tf.truncated_normal([10],stddev=0.1))
  decode_in_w1 = tf.Variable(tf.truncated_normal([10,2048],stddev=0.1))
  decode_in_b1 = tf.Variable(tf.truncated_normal([2048],stddev=0.1))
  
  decode_layer1_w = weight_variable([3, 3, 64,128 ])
  decode_layer1_b = bias_variable([64])
  decode_layer2_w = weight_variable([5, 5, 32,64 ])
  decode_layer2_b = bias_variable([32])
  decode_layer3_w = weight_variable([5, 5, 3,32 ])
  decode_layer3_b = bias_variable([1])

  encode_layer1_out = tf.nn.relu(conv2d(x, encode_layer1_w,'SAME') + encode_layer1_b)
  encode_layer2_out = tf.nn.relu(conv2d(encode_layer1_out, encode_layer2_w,'SAME') + encode_layer2_b)
  encode_layer3_out = tf.nn.relu(conv2d(encode_layer2_out, encode_layer3_w,'SAME') + encode_layer3_b)
#  encode_out = tf.nn.leaky_relu(tf.add(tf.matmul(encode_layer3_out,encode_out_w),encode_out_b))
  a = tf.reshape(encode_layer3_out,[-1,4*4*128])
  encode_out_nol2 = tf.add(tf.matmul(a,encode_out_w1),encode_out_b1)
  
  xx = tf.square(encode_out_nol2)
  test1 = tf.sqrt(tf.reduce_sum(xx, 1, keepdims=True))
  c = tf.constant(1,dtype=tf.float32)
  encode_out = tf.multiply(c,tf.divide(encode_out_nol2,test1))
  
#  print(xx)
#  print(test1)
  decode_in = tf.nn.relu(tf.add(tf.matmul(encode_out,decode_in_w1),decode_in_b1))
  b = tf.reshape(decode_in,[-1,4,4,128])
  
  decode_layer1_out = tf.nn.relu(deconv2d(b, 64,'SAME',(3,3)) + decode_layer1_b)
  decode_layer2_out = tf.nn.relu(deconv2d(decode_layer1_out, 32,'SAME',(5,5)) + decode_layer2_b)
  decode_layer3_out = tf.nn.relu(deconv2d(decode_layer2_out, 3,'SAME',(5,5)) + decode_layer3_b)
  decode_out = tf.nn.relu(decode_layer3_out)
  return decode_out,encode_out

def gradient_visualization(loss,autoencoder_input):
  with tf.name_scope('gradients'):
    a = tf.reduce_sum(tf.abs(tf.gradients(loss,autoencoder_input)))
    tf.summary.scalar('gradient',a)

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

def plot_roc(fpr, tpr, anomal):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.title('normal vs number {}'.format(anomal))
    plt.legend(loc="best")


if __name__ == '__main__':
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
  gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
  config=tf.ConfigProto(gpu_options=gpu_options)
  sess = tf.Session()
  autoencoder_input = tf.placeholder(tf.float32,[None,32,32,3])
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
  tr_a = copy.deepcopy(data)
  print('start training:')
  for episode in range(1000):
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
#    print(sess.run(loss, feed_dict={autoencoder_input:data}))
  print('training end')

  recon_loss = []
  for i in range(np.shape(data)[0]):
    recon_loss.append(sess.run(loss,feed_dict={autoencoder_input:data[i:i+1,:,:,:]}))
  recon_loss_index = np.argsort(recon_loss)
  
  score = np.reshape(recon_loss,[np.shape(data)[0],1])
  fpr, tpr, thresholds = roc_curve(np.reshape(label,[np.shape(data)[0],1]), score, pos_label=inlier_num)
  print('auc=')
  print(1-auc(fpr, tpr))
