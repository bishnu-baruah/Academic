import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as int
from scipy import signal

x = np.linspace(-3e-3,3e-3,2000)
lmda = 500e-9
k = 2*np.pi/lmda
z = 2
A = (1j*k)/(2*z)

def rect(z):
    return np.where(np.abs(z) <= 1e-3, 1,0)

#print(rect(x))

def exp(i):
    return (np.exp(A*i**2))    

rl = (signal.convolve(rect(x), exp(x), mode="same")).real
im = (signal.convolve(rect(x), exp(x), mode="same")).imag
Inty = (rl**2+im**2)

fig, ax = plt.subplots()
ax. plot(x*1e3, Inty)
ax.axhline(y=0)
plt.show()