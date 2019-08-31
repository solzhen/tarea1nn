# -*- coding: utf-8 -*-
"""
Activation functions
Created on Fri Aug 30 23:55:11 2019

@author: Cristobal
"""

import numpy as np

class ActivationFunction:
  def apply(x):
    '''
    Applies function to input X
    '''
    pass
  def derivative(z):
    '''
    Calculates derivative in terms of the function itself as input z.
    Assume z = apply(x)
    '''
    pass

class Step(ActivationFunction):
  def apply(x):
    return np.where(x >= 0, 1, 0)
  def derivative(z):
    return 0
  
class Sigmoid(ActivationFunction):
  def apply(x):
    return 1 / (1 + np.exp(-x))
  def derivative(z):
    return z * (1 - z)
  
class Tanh(ActivationFunction):
  def apply(x):
    p = np.exp(x)
    n = np.exp(-x)
    return (p - n) / (p + n)
  def derivative(z):
    return 1 - np.power(z, 2)
  
class Relu(ActivationFunction):
  def apply(x):
    return np.maximum.reduce([x, np.zeros(x.shape)])
  def derivative(z):
    return -np.maximum.reduce([-z, -np.ones(z.shape)])