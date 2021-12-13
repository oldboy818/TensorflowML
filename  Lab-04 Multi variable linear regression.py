# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 모두를 위한 딥러닝 시즌2
# ##  Lab-04 Multi variable linear regression

import tensorflow as tf
import numpy as np

# ### Multi variable

# +
# data and label
x1 = [73, 93, 89, 96, 73]
x2 = [80, 88, 91, 98, 66]
x3 = [75, 93, 90, 100, 70]
Y  = [152, 185, 180, 196, 142]

W1 = tf.Variable(tf.random.normal([1]))
W2 = tf.Variable(tf.random.normal([1]))
W3 = tf.Variable(tf.random.normal([1]))
b  = tf.Variable(tf.random.normal([1]))

learning_rate = 0.000001

# Gradient descent
for i in range(1000+1):
    
    # tf.GradientTape() tor record the gradient of the cost func
    with tf.GradientTape() as tape:
        hypothesis = W1*x1 + W2*x2 + W3*x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # calculates the gradients of the cost
    W1_grad, W2_grad, W3_grad, b_grad = tape.gradient(cost, [W1, W2, W3, b])
    
    # update w1, w2, w3 and b
    W1.assign_sub(learning_rate * W1_grad)
    W2.assign_sub(learning_rate * W2_grad)
    W3.assign_sub(learning_rate * W3_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:12.4f}".format(i, cost.numpy()))
# -
# ### Multi Variable With Matrix


# +
# Data by matrix
data = np.array([
    # x1, x2, x3, y
    [ 72, 80, 75, 152],
    [ 93, 88, 93, 185],
    [ 90, 91, 90, 180],
    [ 96, 98, 100, 196],
    [ 73, 66, 70, 142]
], dtype=np.float32)

# Data slice
X = data[:,:-1]
Y = data[:,[-1]]

# W,b inintialize
W = tf.Variable(tf.random.normal([3,1])) # W의 row는 X의 column 수와 같다.
b = tf.Variable(tf.random.normal([1]))   # Y는 5*1이니, 그에 맞게 W, b의 사이즈를 결정.

# hypothesis, prediction function
def predict(X):
    return tf.matmul(X,W) + b

learning_rate = 0.000001

n_epochs = 2000
for i in range(n_epochs+1):
    # record the gradient of the cost func
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X)-Y)))
    
    # calculates the gradients of the cost
    W_grad, b_grad = tape.gradient(cost, [W, b])
    
    # update W and b
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5} | {:12.4f}".format(i, cost.numpy()))
# -

# Multi Variable 연산에서 Matrix를 사용에 따른 차이는 Weight, hypothesis, parameters update에 있어 간단한 한 줄로 정리가 가능하다.


