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
# ##  Lab-06-1 Softmax Classifier

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
nb_classes = 3
# -

# Softmax func은 여러 클래스를 예측할 때 유용하게 쓰인다.\
# 구조는 주어진 입력 X와 Weight W의 곱을 통해 Scores Y를 출력한다
# $$
# XW = Y$$
# Score를 softmax 함수에 입력하면 출력값이 확률로 나온다. 따라서 출력값을 총합은 반드시 1이어야 한다.
#


