# -*- coding: utf-8 -*-
"""
Tarea 1 RRNN
Created on Fri Aug 30 23:59:42 2019

This code is specific for Iris dataset (7 characteristics)
This and aux_functions are not generic, only works for Iris
Neuron, Activation_functions and Network modules are generic

@author: Cristobal
"""

import numpy as np
import random

from activation_functions import *
from neuron import *
from network import *
from aux_functions import *

"""Random fix"""
seed = 128
random.seed(seed)
np.random.seed(seed)

"""Load dataset"""
fo = np.loadtxt("seeds_dataset.txt")

"""Normalization"""
m1,m2,m3,m4,m5,m6,m7,m8 = fo.max(axis=0)
fo = fo / np.array([[m1,m2,m3,m4,m5,m6,m7, 1]])

"""Shuffle Examples"""
np.random.shuffle(fo)

"""Separate input and class data + 1 hot encoding"""
input_set = fo[::,:-1]
pre_class_set = fo[::,-1:]
class_set = hot_enc(pre_class_set)

"""Dataset partition"""
set_n = 170 #train size out of 210

train_set = input_set[:set_n][::]
train_set_class = class_set[:set_n][::]

test_set = input_set[set_n:][::]
test_set_class = class_set[set_n:][::]

"""Set up Neural Network"""
a_1 = [Tanh] * 5
NN = Red(7, 3, 5, [15, 10, 12, 8, 3], act_fun=a_1)

"""Training + Error and hit_rate curves generation"""
print("Training...")
hit_rate = []
error = []
for i in range(1000):  
  #Por cada epoch
  er = (NN.train(train_set, train_set_class))
  output,cach = NN.feed(test_set)
  predicted = reverse_hot_enc(output)
  expected = reverse_hot_enc(test_set_class)
  hits = 0
  total = 0
  for p,e in zip(predicted, expected):
    if p==e:
      hits += 1
    total += 1
  hit_rate.append(hits / total * 100)
  error.append(er)

"""Plotting"""
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(111)
plt.plot(error)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.show()

plt.subplot(111)
plt.plot(hit_rate)
plt.ylabel('% Acierto')
plt.xlabel('Epoch')
plt.show()
  
"""Final Results"""
output,cach = NN.feed(test_set)
predicted = reverse_hot_enc(output)
expected = reverse_hot_enc(test_set_class)

"""Confusion Matrix"""
conf_mat, tot_pred, tot_cl = confusion_matrix(predicted, expected)  
nice_print (conf_mat, tot_pred, tot_cl)


#aux = hot_enc(np.reshape(np.argmax(output, axis=1) + 1, (-1, 1)))

