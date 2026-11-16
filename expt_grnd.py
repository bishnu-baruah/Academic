import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as int
from scipy import signal

# # fig = plt.figure(figsize=(10,5))
# # ax = fig.add_subplot(121)
# # ax1 = fig.add_subplot(122)

# # x = np.linspace(-1,1,100)
# # y = np.sin(x*np.pi)

# # ax.plot(x,y)
# # ax1.plot(x,y)

# # plt.show()

# # ## plt.subplots create both figure and axes simultaneously
# # fig, axe = plt.subplots(2, 2, figsize=(10, 8))

# # axe[0, 0].plot([1, 2, 3], [1, 4, 9])
# # axe[0, 1].plot([1, 2, 3], [1, 8, 27])
# # axe[1, 0].plot([1, 2, 3], [1, 2, 3])
# # axe[1, 1].plot([1, 2, 3], [3, 2, 1])

# # plt.tight_layout()
# # plt.show()

x = np.linspace(-4e-3,4e-3,200)
lmda = 500e-9
k = 2*np.pi/lmda
z = 2
A = (1j*k)/(2*z)

def rect(z):
    return np.where(np.abs(z) <= 1e-3, 1,0)

#print(rect(x))

def exp(x):
    l = list()
    for i in x:
        l.append(np.exp(A*i**2))    
    return l

rl = (signal.convolve(rect(x), exp(x), mode="same")).real
im = (signal.convolve(rect(x), exp(x), mode="same")).imag
Inty = (rl**2+im**2)

fig, ax = plt.subplots()
ax. plot(x*1e3, Inty)
ax.axhline(y=0)
plt.show()

