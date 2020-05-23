'''
Python code to compute the Fourier transform of a constant function.
====================================================================
Author: Avijit Maity
====================================================================
'''
from numpy import *
import  matplotlib.pyplot as plt


xmin = -1.0e5
xmax = 1.0e5

n= 512 # Number of Sampling points
dx = (xmax-xmin)/(n-1)
x = linspace(xmin,xmax,n)
f = linspace(2.0,2.0,len(x))


nft = fft.fft(f, norm='ortho') # computing DFT
karr = fft.fftfreq(n, d=dx) # Computing frequencies
karr = 2*pi*karr
factor = exp(-1j * karr * xmin)
aft = dx * sqrt(n/(2.0*pi)) * factor * nft

# Now we will reorder arrays for plotting using fftshift
karr= fft.fftshift(karr)
aft=fft.fftshift(aft)

plt.suptitle('Fourier tramsform of constant function f(x)=2', size = 18, color = 'g')
# Here we will plot our original function
plt.subplot(1, 2, 1)
plt.plot(x,f, label ='f(x)=2')
plt.xlabel('x',size = 13)
plt.ylabel('f(x)',size = 13)
plt.legend(loc=4)
plt.grid()
plt.title("Configaration space",size = 14)

#Here we will plot Fourier transform of the constant function
plt.subplot(1, 2, 2)
plt.xlim(-3,+3)
plt.plot(karr,aft.real,label='Numerical', color = 'r')
plt.xlabel('frequency(k)',size = 13)
plt.ylabel("FT(f(x)= 2)",size = 13)
plt.legend(loc=4)
plt.grid()
plt.title("Fourier space",size = 14)


plt.show()



