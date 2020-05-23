import numpy as np
import matplotlib.pyplot as plt

#for plotting the given function
def f(x): # defining given function
    if (x.any() == 0.0):
        return 1
    else:
        return np.sin(x)/x

xmin = -50
xmax = 50

n= 512 # Number of Sampling points
dx = (xmax-xmin)/(n-1)
x = np.linspace(xmin,xmax,n)


def analytical(k):       #defining the function for the analytical Fourier transform
    return 0.5*np.sqrt(np.pi/2)*(np.sign(k+1)-np.sign(k-1))


#plotting the numerical Fourier transform
k=np.loadtxt("Problem 2 data.txt",usecols=[0],dtype="float")
y1=np.loadtxt("Problem 2 data.txt",usecols=[1],dtype="float")
y2=np.loadtxt("Problem 2 data.txt",usecols=[2],dtype="float")
aft=abs(np.sqrt(y1**2+y2**2))

m=k.argsort()
aft=aft[m]
k=np.sort(k)

plt.suptitle('Fourier transform of Sin(c) using FFTW', size = 18, color = 'r')
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
plt.xlim(-4,4)
plt.plot(k,aft.real,label='Numerical')
plt.plot(k,analytical(k),label='Analytical')
plt.xlabel('frequency(k)',size = 13)
plt.ylabel("FT(sinc(x))",size = 13)
plt.legend(loc=4)
plt.grid()
plt.title("Fourier space",size = 14)


plt.show()
