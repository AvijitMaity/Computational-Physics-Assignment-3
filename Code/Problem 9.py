'''
Python code to compute the convolution of the box function
====================================================================
Author: Avijit Maity
====================================================================
'''

import numpy as np
import matplotlib.pyplot as plt

def f(x):#Definition of the function given
    if -1<x<1:
        value=1
    else:
        value=0
    return value

def dft(y):# Definition of discrete fourier transform
    N = len(y)
    Fourier_arr = []
    for p in range(N):
        F = 0
        for q in range(N):
            F += y[q] * np.exp(- 2j *np.pi * p * q / N)
        Fourier_arr.append(F / (np.sqrt(N)))
    return Fourier_arr

def idft(y):# Definition of discrete fourier transform
    N = len(y)
    IFourier_arr = []
    for p in range(N):
        F = 0
        for q in range(N):
            F += y[q] * np.exp(+ 2j *np.pi * p * q/ N)
        IFourier_arr.append(F / (np.sqrt(N)))
    return IFourier_arr


x_min = -5.0 #minimum value of x
x_max = 5.0#maximum value of x
n = 256 #number of smple points
dx = (x_max-x_min)/(n-1)# resolution
sampled_arr = np.zeros(n)
x = np.zeros(n)

for i in range(n):
        sampled_arr[i] = f(x_min+i*dx)
        x[i] = x_min+i*dx

nft = dft(sampled_arr) # here  we will do discrete fourier transform of our function
k = 2*np.pi*np.fft.fftfreq(n, d=dx) # Computing frequencies
sorted_k=k.argsort()[::1] # sorting of the k points
Double=np.asarray(nft)*np.asarray(nft)
conv_x=dx*np.sqrt(n)*np.asarray(idft(Double)) # this is the definition of the convolution
conv_x=conv_x[sorted_k]

plt.title("DFT CONVOLUTION",size = 15)
plt.plot(x,np.real(conv_x),'g',label='Convolution of the Box function to it-self')
plt.plot(x,sampled_arr,'r', label='The box function')
plt.xlabel('x',fontsize=14)
plt.ylabel('f(x)',fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
