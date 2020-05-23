import numpy as np
import matplotlib.pyplot as plt

#for plotting the given function
def f(x): # defining given function
    return np.exp(-(x**2))

xmin = -50
xmax = 50

n= 5000 # Number of Sampling points
dx = (xmax-xmin)/(n-1)
x = np.linspace(xmin,xmax,n)

#defining the function for the analytical Fourier transform
def analytical(k):       #defining the function for the analytical Fourier transform
    return np.exp(-0.25 * k ** 2.0) / np.sqrt(2.0)

#plotting the numerical Fourier transform
k=np.loadtxt("Problem 4 data.txt",usecols=[0],dtype="float")
y1=np.loadtxt("Problem 4 data.txt",usecols=[1],dtype="float")
y2=np.loadtxt("Problem 4 data.txt",usecols=[2],dtype="float")
aft=abs(np.sqrt(y1**2+y2**2))

m=k.argsort()
aft=aft[m]
k=np.sort(k)


plt.suptitle('Fourier transform of exp(-(x**2)) ', size = 18, color = 'r')
# Here we will plot our original function
plt.subplot(1, 2, 1)
plt.plot(x,f(x), label ='exp(-(x**2))')
plt.xlabel('x',size = 13)
plt.ylabel('f(x)',size = 13)
plt.xlim(-5,5)
plt.legend(loc=4)
plt.grid()
plt.title("Configaration space",size = 14)

#Here we will plot Fourier transform of the given function
plt.subplot(1, 2, 2)
plt.xlim(-3,+3)
plt.plot(k,aft.real,'o',label='Numerical')
plt.plot(k,analytical(k),label='Analytical')
plt.xlabel('frequency(k)',size = 13)
plt.ylabel("FT(exp(-(x**2)))",size = 13)
plt.xlim(-15,15)
plt.legend()
plt.grid()
plt.title("Fourier space",size = 14)


plt.show()
