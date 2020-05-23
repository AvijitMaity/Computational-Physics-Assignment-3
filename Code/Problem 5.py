'''
DFT vs FFT
====================================================================
Author: Avijit Maity
====================================================================
'''

import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
import time as t


def dft(q, y): # Definition of the DFT for a particular q
    N = len( y)
    c = 0
    for n in range(N):
         c +=  y[n] * np.exp(- 2j *np.pi * q * n / N)
    return c/(np.sqrt(N))

x=np.linspace(4,100,97)

p=0
time_DFT=np.zeros(len(x))
time_numpy=np.zeros(len(x))

for n in range(4,101):
    sampled_data=np.arange(n) # sample data
    dft_direct=np.zeros(n, dtype=np.complex)
    start_time1=t.time()
    for i in range(n):
        dft_direct[i]=dft(i, sampled_data)
    time_DFT[p]=t.time()-start_time1    # time required for not using numpy
    start_time2=t.time()
    DFT_numpy=np.fft.fft(sampled_data)
    time_numpy[p]=t.time()-start_time2  # time required for using numpy
    p=p+1



plt.plot(x,time_DFT, '.', marker='d', color='blue', label='Time taken without using numpy.fft.fft')
plt.plot(x,time_numpy, '.', marker='d', color='green', label='Time taken using numpy.fft.fft' )
plt.title('Comparison of time taken by DFT and FFT',size = 18, color= "r")
plt.xlabel("Number of points n")
plt.ylabel("Time taken(in s)")
plt.legend()
plt.grid()
plt.show()