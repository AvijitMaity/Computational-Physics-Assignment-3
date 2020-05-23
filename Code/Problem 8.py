'''
Python code to compute the 2D Fourier transform of the given function
====================================================================
Author: Avijit Maity
====================================================================
'''
from numpy import *
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fun(x,y): # defining given function
   return exp(-(x**2.0+y**2.0))

def analytical(kx,ky): # It will give analytical fourier transform of the sinc(x) function
	return 0.5*exp(-(kx**2.0+ky**2.0)*0.25)

# minimum and maximum value
xmin = -50
xmax = 50
ymin = -50
ymax = 50


nx= 512 # Number of Sampling points
dx = (xmax-xmin)/(nx-1)
x = linspace(xmin,xmax,nx)

ny= 512 # Number of Sampling points
dy = (xmax-xmin)/(ny-1)
y = linspace(xmin,xmax,ny)

X,Y = meshgrid(x,y)
f = fun(X,Y)

nft = fft.fftshift(fft.fft2(f, norm='ortho')) # computing DFT

kx = fft.fftshift(fft.fftfreq(nx, d=dx)) # Computing frequencies
ky = fft.fftshift(fft.fftfreq(ny, d=dy)) # Computing frequencies

kx = 2*pi*kx
ky = 2*pi*ky
K_x,K_y = meshgrid(kx,ky)

factor_x = exp(-1j * K_x * xmin)
factor_y = exp(-1j * K_y * ymin)


aft = dx *dy* sqrt((nx*ny))/(2.0*pi) * factor_x*factor_y * nft


# Here we will plot the FFT of the function computed using numpy.fft.fft2
fig=plt.figure(figsize=plt.figaspect(0.4))
ax1=fig.add_subplot(1,2,1,projection='3d')
ax1.contour3D(K_x,K_y,abs(aft),100)             #plotting the FFT of the function computed using numpy.fft.fft2
ax1.set_title("Numerical Fourier Transform of 2D Gaussian")
ax1.set_xlabel("kx")
ax1.set_ylabel("ky")
ax1.set_zlabel("Fourier Transform")

# Here we will plot the anaytical result
ax1=fig.add_subplot(1,2,2,projection='3d')
ax1.contour3D(K_x,K_y,analytical(K_x,K_y),100)        #plotting the FFT of the function analytically
ax1.set_title("Analytical Fourier Transform of 2D Gaussian")
ax1.set_xlabel("kx")
ax1.set_ylabel("ky")
ax1.set_zlabel("Fourier Transform")


plt.show()
