import numpy as np
import numbers

class Step:
    def apply(z):
        return np.where(z >= 0, 1, 0)
class Sigmoid:
    def apply(z):
        return 1 / (1 + np.exp(-z))
    def derivative(z):
        return Sigmoid.apply(z) * (1 - Sigmoid.apply(z))
class Tanh:
    def apply(z):
        p = np.exp(z)
        n = np.exp(-z)
        return (p - n) / (p + n)
    def derivative(z):
        return 1 - Tanh.apply(z) * Tanh.apply(z)
class Relu:
    def apply(z):
        return np.where(z >= 0, z, 0)
    def derivative(z):
        return np.where(z >= 0, 1, 0)

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
        self.memory = np.dot(entrada, self.pesos) + self.sesgo
        return self.funcion.apply(self.memory)
##    def train(self, x, y):
##        z = self.memory
##        de_dz = (y - z)
##        dz_du = funcion.derivative(u)
##        de_dw = de_dz * dz_du * x
##        de_db = de_dz * dz_du
##        pesos += de_dw
##        sesgo += de_db
##        return
    def back(self, x, error):
        deriv = self.funcion.derivative(self.memory)
        self.delta = error * deriv
        return self.delta, self.pesos
    def update(self, inputN, r):
        self.pesos += np.dot(self.delta, inputN) * r
        self.sesgo += self.delta * r
        return self.memory
class Red:    
    def __init__(self, n_x, n_y, n_l, l_n, ll_w=None, ll_b=None, ll_a=None, r=0.01):
        self.n_x = n_x #n of entries
        self.n_y = n_y #n of output values
        self.n_l = n_l #n of layers
        self.l_n = l_n #list of number of neurons per layer
        self.ll_w = ll_w if ll_w else []#list weight
        self.ll_b = ll_b if ll_b else []#list bias
        self.ll_a = ll_a if ll_a else []#list act.func.
        self.r = r #learning rate
        self.nn = []
        prev = 0
        p = n_x
        for i in range(len(l_n)):
            self.nn.append([]) 
            for j in range(prev, l_n[i] + prev):                
                if not ll_w:
                    self.ll_w.append( np.random.randn(p) )
                    #print(p)
                if not ll_b:
                    self.ll_b.append( 0 )
                if not ll_a:
                    self.ll_a.append( Sigmoid )
                self.nn[i].append(Neurona(self.ll_w[j], self.ll_b[j], self.ll_a[j]))
            prev += l_n[i]
            p = l_n[i]
    def feed(self, x):
        for l in self.nn:
            h = []
            for n in l:
                m = n.feed(x)
                h.append(m)
            x = np.array(h)
        return x
    def train(self, x, y):
        z = self.feed(x)
        error = y - z
        for l in reversed(self.nn):
            delta = []
            for n, i in zip(l, range(len(l))):
                delta.append(n.back(x, error[i]))
            suma = 0
            for a,b in delta:
                suma += a * b
            error = suma
        inputN = x
        for l in self.nn:
            h = []
            for n in l:
                h.append(n.update(inputN, self.r))
            inputN = np.array(h)
        
import os
fo = open(os.getcwd().replace("\\","/") + "/seeds_dataset.txt", "r")

fldata = []
for line in [a.split("\t") for a in fo.read().split("\n")]:
    fl = [float(e) for e in line if e]
    if int(fl[-1]) == 1:
        fl[-1] = [1,0,0]
    elif int(fl[-1]) == 2:
        fl[-1] = [0,1,0]
    else:
        fl[-1] = [0,0,1]
    fldata.append(fl)
import random
random.seed(27)
random.shuffle(fldata)
line = fldata[0][:-1]
train_data = fldata[1:155]
test_data = fldata[155:]

NN = Red(7, 3, 3, [7, 3], r = 0.1)

for i in range(1000):
    for e in train_data:
        NN.train(np.array(e[:-1]), np.array(e[-1]))
    z = NN.feed(np.array(train_data[20][:-1]))
    print (z, end=' vs ')
    print (train_data[20][-1])
    
