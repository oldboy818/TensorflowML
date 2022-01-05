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
#     display_name: Python [conda env:py37tf20]
#     language: python
#     name: conda-env-py37tf20-py
# ---

# # 모두를 위한 딥러닝 시즌2
# ##  Lab-06-1 Softmax Classifier

# ### Sigmoid
# $$
# H_L(x) = WX일 때\\
# Z = H_L(X)이고\ g(Z)이다.\\
# 이때\ g(Z)=\frac{1}{1+e^{-z}} : Sigmoid func 혹은 Logistic func이라 한다.\\
# H_R(X) = g(H_L(X))이다.
# $$
#
# logistic classification 문제에서 두 가지 클래스를 Sigmoid func을 사용하면 해결할 수 있다.\
# 그럼 Multinomial classification의 경우는??
#
# $$
# \begin{bmatrix}
# W_{A1} & W_{A2} & W_{A3} \\
# W_{B1} & W_{B2} & W_{B3} \\
# W_{C1} & W_{C2} & W_{C3}
# \end{bmatrix}
# \begin{bmatrix}
# {x_1} \\
# {x_2} \\
# {x_3}
# \end{bmatrix}
# =
# \begin{bmatrix}
# W_{A1}x_1+W_{A2}x_2+W_{A3}x_3 \\
# W_{B1}x_1+W_{B2}x_2+W_{B3}x_3 \\
# W_{C1}x_1+W_{C2}x_2+W_{C3}x_3
# \end{bmatrix}
# =
# \begin{bmatrix}
# \bar{y_A} \\
# \bar{y_B} \\
# \bar{y_C}
# \end{bmatrix}$$
#
# 위와 같이 독립된 Classification이 행렬의 곱으로 구할 수 있다.\
# 이를 통해 출력되는 값은 0부터 1사이의 값이 아니므로 다시 어떤 함수에 입력하여 원하는 범위 0~1으로 출력되도록 해야한다.\
# 이때 사용되는 함수가 Softmax function이다.\
#
# ### Softmax function
# $$
# S(y_i)=\frac{e^{y_i}}{\sum_{j}e^{y_i}}$$
#
# **Softmax func은 Hypothesis가 되고**, 그 출력값은 0~1 범위 내의 값이고, 모든 출력값의 합은 1이다. 즉 각각을 확률로 볼 수 있다는 말이다.
#
# 그렇다면 **Cost function**은 어떻게 될까?
#
# $$
# CROSS-ENTROPY: D(S, L) = -\sum_{i}L_{i}log(S_i)$$
#
# 이때, S는 Softmax func의 출력값 $S(y)= bar{y}$이고 L은 결과값 Y이다.
#
# softmax func을 hypothesis로 받아 구하는 cost function을 **Cross-entropy**라고 한다.
#
# 왜 이 함수가 Cost func에 적합한지에 대해 알아보자.
# $$
# -\sum_{i}L_{i}log(\bar{y_i})=\sum_{i}L_{i}(-log(\bar{y_i}))$$
# 마이너스가 안으로 들어가 log가 x축 대칭되었다. $\bar{y_i}$가 0으로 가면 $\infty$, 1로 가면 0이 출력된다.
#
# ### Logistic cost VS cross entropy
# $$
# C:\ (H(x),y)\ =\ ylog(H(x))-(1-y)log(1-H(x))$$
# VS
# $$
# D(S,L)\ =\ -\sum_{i}L_{i}log(S_i)$$
#
# 여러 개의 Training Set이 있을 때, cost func은 다음과 같다.
# $$
# Loss\ =\ \frac{1}{N}\sum_{i}D(S(Wx_{i}+b),\ L_i)$$

import tensorflow as tf
import numpy as np

# +
# 8*4 Matrix
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

# ONE-HOT ENCODING 8*3 Matrix
# 결과값 Y = L_i
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# convert into numpy and float format
x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

# num classes
nb_classes = 3 # y_data의 column 크기가 3이므로

print(x_data.shape)
print(y_data.shape)

# +
# Weight 'W' and bias 'b'

# W * x + b = y 행렬연산을 위해 Weight의 shape은 (4, 3)이어야 하고, bias는 (3, 0)
W = tf.Variable(tf.random.normal([4, nb_classes]), name = 'weight')
b = tf.Variable(tf.random.normal([nb_classes]), name = 'bias')

variables = [W, b]

print(W, b)


# +
# Softmax function = Hypothesis

def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X,W) + b) # 행렬의 덧셈은 shape이 달라도 연산이 되는군.

print(hypothesis(x_data))


# +
# Cost function = Cross-Entropy

def cost_func(X, Y):
    S = hypothesis(X) # softmax func을 이용한 hypothesis
    cost = -tf.reduce_sum(Y * tf.math.log(S), axis=1)
    cost_mean = tf.reduce_mean(cost)
    
    return cost_mean

print(cost_func(x_data, y_data))


# -

# * tf.reduce_sum\
# tensor의 dimension을 탐색하며 개체들의 총합을 계산
#

# +
# Gradient descent : mininze Cost func to find opimizative W, b

def grad_func(X,Y):
    with tf.GradientTape() as tape:
        cost = cost_func(X, Y)
        grad = tape.gradient(cost, variables) # variables = [W, b]
                                              # W와 b에 대한 기울기 값을 반환
            
        return grad


# +
# Optimization and Launch graph

def fit(X, Y, epochs = 2000, verbose = 100):
    
    #경사하강법으로 손실 함수를 최소화하는 모델 파라미터를 찾기 위한 class
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
    
    for i in range(epochs):
        optimizer.apply_gradients(zip(grad_func(X, Y), variables))
        
        if (i==0) | ((i+1)%verbose==0):
            print('Loss at epoch %d: %f' %(i+1, cost_func(X, Y).numpy()))
            
fit(x_data, y_data)

# +
# Test

a = hypothesis(x_data)
print(a) #softmax 함수를 통과시킨 x_data

#argmax 가장큰 값의index를 찾아줌

print(tf.argmax(a,1)) #가설을 통한 예측값
print(tf.argmax(y_data,1)) #실제 값
# -


