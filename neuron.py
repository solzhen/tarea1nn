# -*- coding: utf-8 -*-
"""
Neurona
Created on Fri Aug 30 23:57:02 2019

@author: Cristobal
"""
import numpy as np
from activation_functions import *

class Neurona:
  def __init__(self, pesos, sesgo, funcion):
    #print (type(pesos))
    if not isinstance(pesos, np.ndarray):
      print ("error1")
      return
    if not isinstance(sesgo,numbers.Number):
      print ("error2")
      return
    self.pesos = pesos
    self.sesgo = sesgo
    self.funcion = funcion
    self.memory = None
    
  def feed(self, entrada):
    return self.funcion.apply(np.dot(entrada, self.pesos) + self.sesgo)
  
  def train(self, entrada, esperado, learn_rate):
    real = self.feed(entrada)
    error = esperado - real
    trans_der = self.funcion.derivative(real)
    delta = error * trans_der
    self.pesos += entrada * learn_rate
    self.sesgo += delta * learn_rate