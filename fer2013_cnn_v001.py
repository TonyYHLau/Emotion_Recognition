# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 02:08:11 2017

@author: Yuen Hoi, LAU
"""

import pandas as pd
import numpy as np

#load data
fer2013 = pd.read_csv("fer2013.csv")
train_x = fer2013[fer2013['Usage']=='Training']['pixels']
train_y = fer2013[fer2013['Usage']=='Training']['emotion']

pubtest_x = fer2013[fer2013['Usage']=='PublicTest']['pixels']
pubtest_y = fer2013[fer2013['Usage']=='PublicTest']['emotion']

pritest_x = fer2013[fer2013['Usage']=='PrivateTest']['pixels']
pritest_y = fer2013[fer2013['Usage']=='PrivateTest']['emotion']

#onehot encoder

list_train_x=[]
list_pubtest_x=[]
list_pritest_x=[]
for i in range(len(train_x)):
    a = train_x[i].split(" ")
    b = list(np.array(a).astype(float))    
    list_train_x.append(b)

for i in range(len(pubtest_x)):    
    c = pubtest_x[len(train_x) + i].split(" ")
    d = list(np.array(c).astype(float))    
    list_pubtest_x.append(d)

for i in range(len(pritest_x)):
    e = pritest_x[len(train_x)+ len(pubtest_x)+i].split(" ")
    f = list(np.array(e).astype(float))    
    list_pritest_x.append(f)

X_train = np.array(list_train_x)
X_pubtest = np.array(list_pubtest_x)
X_pritest = np.array(list_pritest_x)

list_train_y=[0]*7
all_list_train_y=[]
for i in range(len(train_y)):
    
    if train_y[i] == 0:
        list_train_y[0]=1
    elif train_y[i] == 1:
        list_train_y[1]=1
    elif train_y[i] == 2:
        list_train_y[2]=1
    elif train_y[i] == 3:
        list_train_y[3]=1
    elif train_y[i] == 4:
        list_train_y[4]=1
    elif train_y[i] == 5:
        list_train_y[5]=1
    elif train_y[i] == 6:
        list_train_y[6]=1
    all_list_train_y.append(list_train_y)
    list_train_y=[0]*7

list_pubtest_y=[0]*7
all_list_pubtest_y=[]
for i in range(len(pubtest_y)):
    
    if pubtest_y[len(train_x) +i] == 0:
        list_pubtest_y[0]=1
    elif pubtest_y[len(train_x) +i] == 1:
        list_pubtest_y[1]=1
    elif pubtest_y[len(train_x) +i] == 2:
        list_pubtest_y[2]=1
    elif pubtest_y[len(train_x) +i] == 3:
        list_pubtest_y[3]=1
    elif pubtest_y[len(train_x) +i] == 4:
        list_pubtest_y[4]=1
    elif pubtest_y[len(train_x) +i] == 5:
        list_pubtest_y[5]=1
    elif pubtest_y[len(train_x) +i] == 6:
        list_pubtest_y[6]=1
    all_list_pubtest_y.append(list_pubtest_y)
    list_pubtest_y=[0]*7

list_pritest_y=[0]*7
all_list_pritest_y=[]
for i in range(len(pritest_y)):
    
    if pritest_y[len(train_x)+len(pubtest_y) +i] == 0:
        list_pritest_y[0]=1
    elif pritest_y[len(train_x)+len(pubtest_y) +i] == 1:
        list_pritest_y[1]=1
    elif pritest_y[len(train_x)+len(pubtest_y) +i] == 2:
        list_pritest_y[2]=1
    elif pritest_y[len(train_x) +len(pubtest_y)+i] == 3:
        list_pritest_y[3]=1
    elif pritest_y[len(train_x)+len(pubtest_y) +i] == 4:
        list_pritest_y[4]=1
    elif pritest_y[len(train_x) +len(pubtest_y)+i] == 5:
        list_pritest_y[5]=1
    elif pritest_y[len(train_x) +len(pubtest_y)+i] == 6:
        list_pritest_y[6]=1
    all_list_pritest_y.append(list_pritest_y)
    list_pritest_y=[0]*7

    
y_train = np.array(all_list_train_y)
y_pubtest = np.array(all_list_pubtest_y)
y_pritest = np.array(all_list_pritest_y)

del  b, c, d, e, f, all_list_train_y, all_list_pubtest_y, all_list_pritest_y, list_pritest_x, list_pritest_y, list_pubtest_x, list_pubtest_y, list_train_x, list_train_y

from sklearn.preprocessing import StandardScaler 
import tensorflow as tf
ntrain , rowtrain = np.shape(X_train)

#Scale the data
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_pubtest=ss.transform(X_pubtest)
n_classes = 7
batch_size =1511

x = tf.placeholder('float', [None, 2304])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([7,7,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([7,7,32,64])),
               'W_fc':tf.Variable(tf.random_normal([12*12*64,1000])),
               'out':tf.Variable(tf.random_normal([1000, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1000])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 48, 48, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 12*12*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for j in range(int(ntrain/batch_size)):
                epoch_x, epoch_y = X_train[batch_size*j:batch_size*(j+1),:],y_train[batch_size*j:batch_size*(j+1),:]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval({x:X_pubtest, y:y_pubtest}))

#Execution        
train_neural_network(x)
