'''
Python code to compute the power spectrum
====================================================================
Author: Avijit Maity
====================================================================
'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps

#computing DFT of the given data using np.fft
data = np.loadtxt('noise.txt',float)
N = len(data)
x = np.linspace(1,N, N)
dft = np.fft.fft(data, norm='ortho')
k= np.fft.fftfreq(N)
k = 2 * np.pi * k
k = np.fft.fftshift(k) # Now we will reorder arrays for plotting using fftshift
dft = np.fft.fftshift(dft)

#computing power spectrum
Power_spec = np.zeros(len(k))
for i in range(len(k)):
    Power_spec[i] = abs(dft[i]) ** 2 / len(k)


# performing Accuracy test using Scipy
sci_pow_spec = sps.periodogram(data,scaling = 'spectrum',return_onesided = False)

#making 10 equal bin of k amd binning the data of power spectra(Power_noise)
k_span = int((max(k) - min(k)))
n_bin = 10
width = k_span / n_bin
lower_bound = min(k)
upper_bound = max(k)

k1 = np.linspace(lower_bound, upper_bound, n_bin + 1)
binned_pow_spec = np.zeros(n_bin)
k_bin = np.zeros(n_bin)

for i in range(n_bin):
    count = 0
    for j in range(len(k)):
        if k1[i] <= k[j] < k1[i + 1]:
            binned_pow_spec[i] += Power_spec[j]
            count += 1
    binned_pow_spec[i] = binned_pow_spec[i] / count
    k_bin[i] = k1[i] + (k1[i + 1] - k1[i]) / 2


# Here we will plot our experimental data
plt.subplot(2, 2, 1)
plt.scatter(x,data)
plt.xlabel("Data Label",size = 13)
plt.ylabel("Experimental Data",size = 13)
plt.grid()
plt.tight_layout()
plt.title("Plotting of Data",size = 14)

#Here we will plot DFT of the given data
plt.subplot(2, 2, 2)
plt.plot(k,dft,color = 'brown',label = 'DFT')
plt.xlabel('frequency(k)',size = 13)
plt.ylabel("DFT of Data",size = 13)
plt.grid()
plt.tight_layout()
plt.title("DFT of Data",size = 14)


#Here we will plot the power spectrum
plt.subplot(2, 2, 3)
plt.plot(k,Power_spec,color = 'red',label = 'DFT Computed')
plt.scatter(sci_pow_spec[0],sci_pow_spec[1],color = 'green',label = 'Scipy Computed')
plt.xlabel('frequency(k)',size = 13)
plt.ylabel("Periodogram",size = 13)
plt.legend()
plt.grid()
plt.tight_layout()
plt.title("Normal periodogram",size = 14)

# Here we will plot binned periodogram
plt.subplot(2, 2, 4)
plt.bar(k_bin,binned_pow_spec,width, color = "yellow")
plt.xlabel("frequency bins",size = 13)
plt.ylabel("Binned power spectra",size = 13)
plt.legend()
plt.grid()
plt.tight_layout()
plt.title("Binned Periodogram",size = 14)

plt.show()


