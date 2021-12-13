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
# ## Lab02_Simple Regression LAB

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ### H(x) = Wx+b

# Data
x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]

# W, b inintialize
W = tf.Variable(2.0)
b = tf.Variable(0.5)

#hypothesis = W * x + b
hypothesis = W * x_data + b

# ### $cost(W,b) = \frac{1}{m}*\sum_{i=1}^{m}((H(x^{i})-y^{i}))^2$

# cost는 error 제곱의 평균값
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# * tf.reduce_mean()\
# reduce 는 차원(Rank)가 줄어든다는 의미를 갖는다.
# 예로, v = [1,2,3,4]의 Rank는 $R^4$지만, tf.reduce_mean(v)=2.5로 $R^0$이다.
# * square()\
# 제곱승을 해준다.

# 실제 데이터와 초기값으로 주어진 Hyphothesis
hypothesis.numpy()
plt.plot(x_data, hypothesis.numpy(), 'r-')
plt.plot(x_data, y_data, 'o')
plt.ylim(0, 8)
plt.show()

# ## Gradient descent
#
# 경사하강법은 error 제곱의 평균값인 cost를 minimize하는 여러 방법 중 하나이다.\
# 즉, $minimize_{W,b} cost(W,b)$

# Learning_rate initialize
learning_rate = 0.01

# +
for i in range(100):

# Gradient descent
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    
    W_grad, b_grad = tape.gradient(cost, [W,b])

    W.assign_sub(learning_rate*W_grad)
    b.assign_sub(learning_rate*b_grad)
    
    if i % 10 ==0: # 10번 반복될 때마다 값을 출력
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))
# -

# * GradientTape()\
#     with 구문과 사용되는데, with 안 변수의 정보를 tape에 저장한다.<p>
# * tape.gradient(함수, 변수)\
#     tape의 gradient method로 함수의 변수에 대한 경사도값(=미분값)을 구한다.\
# 그 값은 순서대로 반환한다.<p>
# * A.assign_sub(B)\
#     A = A-b\
#     A -= B

# Gradient descent로 minimize한 결과.
plt.plot(x_data, y_data, 'o')
plt.plot(x_data, hypothesis.numpy(), 'r-')
plt.ylim(0, 8)


