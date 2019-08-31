# -*- coding: utf-8 -*-
"""
Red
Created on Fri Aug 30 23:58:17 2019

@author: Cristobal
"""
import numpy as np
from activation_functions import *

class Red:    
  def __init__(self, n_inputs, n_outputs, n_layers, n_neurons_layer, 
               weights=None, biases=None, act_fun=None, learn_r=0.01):
    
    if not isinstance(n_neurons_layer, list):
        raise TypeError("n_neurons_layer must be a list!")
    if len(n_neurons_layer) != n_layers:
      raise Exception("n_layers must be equal to length of n_neurons_layer")
    if n_neurons_layer[-1] != n_outputs:
      raise Exception("length of last layer must be n_outputs")
    if weights:
      if not isinstance(weights, list):
        raise TypeError("weights must be a list!")
      if any(not isinstance(w, np.ndarray) for w in weights):
        raise TypeError("weights elements must be np.ndarray instances!")
      if len(weights) != n_layers:
        raise Exception("n_layers must be equal to length of weights")
      for w,n in zip(weights, n_neurons_layer):
        prev = n_inputs
        r,c = w.shape
        if r != prev or c != n:
          raise Exception("Size mismatch on weights")
    if biases:
      if not isinstance(biases, list):
        raise TypeError("biases must be a list!")
      if any(not isinstance(w, np.ndarray) for w in biases):
        raise TypeError("biases elements must be np.ndarray instances!")
      if len(biases) != n_layers:
        raise Exception("n_layers must be equal to length of biases")
    if act_fun:
      if not isinstance(act_fun, list):
        raise TypeError("act_fun must be a list!")
      if any(not w in [Sigmoid, Tanh, Relu] for w in act_fun):
        raise TypeError("act_fun elements must be valid function names!")
      if len(act_fun) != n_layers:
        raise Exception("n_layers must be equal to length of act_fun")
        
    self.n_inputs = n_inputs
    self.n_outputs = n_outputs
    self.n_layers = n_layers
    self.n_neurons_layer = n_neurons_layer
    self.learn_r = learn_r
    self.grad = []
    self.cache = []
    
    if not weights:
      self.weights = []
      p = n_inputs
      for n_neurons in n_neurons_layer:
        self.weights.append( np.random.randn(p, n_neurons) )
        p = n_neurons
    else:
      self.weights = weights
      
    if not biases:
      self.biases = []
      for n_neurons in n_neurons_layer:
        self.biases.append( np.zeros(n_neurons) )
    else:
      self.biases = biases
      
    if not act_fun:
      self.act_fun = []
      for n_neurons in n_neurons_layer:
        self.act_fun.append( Relu )
    else:
      self.act_fun = act_fun  
      
  def feed(self, x):
    if not isinstance(x, np.ndarray):
      raise TypeError("Input dataset must be a np.ndarray instance!")
    try:
      d1 = x.shape[0]
      d2 = x.shape[1]
    except IndexError:
      raise IndexError("Input database must be two dimensional (even if \
it's only one input")
    self.cache = []
    for layer_weights, layer_biases, layer_act_fun in zip(
        self.weights, self.biases, self.act_fun):
      u = x @ layer_weights + layer_biases
      x = layer_act_fun.apply(u)
      self.cache.append(x)
    return x, self.cache
  
  def error_propagation(self, init_error):
    error = init_error
    self.grad = []
    for i in range(self.n_layers - 1, -1, -1):
      trans_der = self.act_fun[i].derivative(self.cache[i])
      delta = error * trans_der
      self.grad.append(delta)
      error = delta @ self.weights[i].T
    self.grad = list(reversed(self.grad))
    return self.grad
      
  def update_parameters(self, inputN):
    for i in range(self.n_layers):
      for j in range(inputN.shape[0]):        
        inpu = np.reshape(inputN[j], (1, -1))
        delt =  np.reshape(self.grad[i][j], (1, -1))
        aux = np.repeat(delt, inputN.shape[1], 0)
        product = aux * inpu.T        
        self.weights[i] += product * self.learn_r
        self.biases[i] += self.grad[i][j] * self.learn_r
      inputN = self.cache[i]
    return self.weights, self.biases
      
  def train(self, x, y):  
    z,c = self.feed(x)
    mse = ((y - z)**2).mean(axis=0)
    error = (y - z) / x.shape[0] # actually the derivative of the error, 
     #without sum, that¡s done inside function below
    self.error_propagation(error)
    self.update_parameters(x)
    return mse