'''
Python code to compute the Fourier transform of the sinc function
====================================================================
Author: Avijit Maity
====================================================================
'''
from numpy import *
import  matplotlib.pyplot as plt

def f(x): # defining given function
    if (x.any() == 0.0):
        return 1
    else:
        return sin(x)/x

def analytical(karr): # It will give analytical fourier transform of the sinc(x) function
	return 0.5*sqrt(pi/2.0)*(sign(karr+1)-sign(karr-1))

xmin = -50
xmax = 50

n= 512 # Number of Sampling points
dx = (xmax-xmin)/(n-1)
x = linspace(xmin,xmax,n)


nft = fft.fft(f(x), norm='ortho') # computing DFT
karr = fft.fftfreq(n, d=dx) # Computing frequencies
karr = 2*pi*karr
factor = exp(-1j * karr * xmin)
aft = dx * sqrt(n/(2.0*pi)) * factor * nft

# Now we will reorder arrays for plotting using fftshift
karr= fft.fftshift(karr)
aft=fft.fftshift(aft)

plt.suptitle('Fourier tramsform of Sin(c)', size = 18, color = 'r')
# Here we will plot our original function
plt.subplot(1, 2, 1)
plt.plot(x,f(x), label ='Sinx / x')
plt.xlabel('x',size = 13)
plt.ylabel('f(x)',size = 13)
plt.legend(loc=4)
plt.grid()
plt.title("Configaration space",size = 14)

#Here we will plot Fourier transform of the sinc function
plt.subplot(1, 2, 2)
plt.xlim(-3,+3)
plt.plot(karr,aft.real,label='Numerical')
plt.plot(karr,analytical(karr),label='Analytical')
plt.xlabel('frequency(k)',size = 13)
plt.ylabel("FT(sinc(x))",size = 13)
plt.legend(loc=4)
plt.grid()
plt.title("Fourier space",size = 14)


plt.show()



