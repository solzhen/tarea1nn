# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:48:40 2019

Not generic.

@author: Cristobal
"""

"""Aux Functions"""
import numpy as np

def hot_enc(s):
  return np.eye(3)[s.T.astype(int) - 1][0]

def reverse_hot_enc(arr):
  return np.argmax(arr, axis=1) + 1

def nice_print(m,t,c):
  '''Don't let Alexandre see this disgusting thing!'''
  print ("             | Class_1  | Class_2  | Class_3")
  print ("Predicted_1  | ", end="")
  print (m[0][0], " "*(8 - len(str(m[0][0]))), end = "| ")
  print (m[0][1], " "*(8 - len(str(m[0][1]))), end = "| ")
  print (m[0][2], " "*(8 - len(str(m[0][2]))), end = "| ")
  print ("TotalPredicted_1:", t[0])
  print ("Predicted_2  | ", end="")
  print (m[1][0], " "*(8 - len(str(m[1][0]))), end = "| ")
  print (m[1][1], " "*(8 - len(str(m[1][1]))), end = "| ")
  print (m[1][2], " "*(8 - len(str(m[1][2]))), end = "| ")
  print ("TotalPredicted_2:", t[1])
  print ("Predicted_3  | ", end="")
  print (m[2][0], " "*(8 - len(str(m[2][0]))), end = "| ")
  print (m[2][1], " "*(8 - len(str(m[2][1]))), end = "| ")
  print (m[2][2], " "*(8 - len(str(m[2][2]))), end = "| ")
  print ("TotalPredicted_3:", t[2])
  print ("        Total: ", end="")
  print (c[0]," "*(10 - len(str(c[0]))), end="")
  print (c[1]," "*(10 - len(str(c[1]))), end="")
  print (c[2])  
  
def confusion_matrix(p, r):
  cm = [[0,0,0],[0,0,0],[0,0,0]]
  for h,j in zip(p, r):
    cm[h-1][j-1] += 1
  tc = [0,0,0]
  for e in r:
    tc[e - 1] += 1 
  tp = [sum(row) for row in cm]
  return cm, tp, tc  