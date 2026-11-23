# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import integrate as int
# from scipy import signal

# # # fig = plt.figure(figsize=(10,5))
# # # ax = fig.add_subplot(121)
# # # ax1 = fig.add_subplot(122)

# # # x = np.linspace(-1,1,100)
# # # y = np.sin(x*np.pi)

# # # ax.plot(x,y)
# # # ax1.plot(x,y)

# # # plt.show()

# # # ## plt.subplots create both figure and axes simultaneously
# # # fig, axe = plt.subplots(2, 2, figsize=(10, 8))

# # # axe[0, 0].plot([1, 2, 3], [1, 4, 9])
# # # axe[0, 1].plot([1, 2, 3], [1, 8, 27])
# # # axe[1, 0].plot([1, 2, 3], [1, 2, 3])
# # # axe[1, 1].plot([1, 2, 3], [3, 2, 1])

# # # plt.tight_layout()
# # # plt.show()

# lmda = 500e-9
# k = 2*np.pi/lmda
# z = 0.1
# A = (1j*k)/(2*z)

# r = np.linspace(-3e-3,3e-3,1000)
# x,y = np.meshgrid(r,r)

# def rect(a,b):
#     i = np.abs(a) <= 1e-3
#     j = np.abs(b) <= 1e-3
#     return np.where( i & j, 1,0)

# def exp(a,b):
#     i = a**2+b**2
#     return (np.exp(A*i)) 

# rl = ((signal.fftconvolve(rect(x,y), exp(x,y), mode="same"))/(1j*lmda*z)).real
# im = ((signal.fftconvolve(rect(x,y), exp(x,y), mode="same"))/(1j*lmda*z)).imag
# Inty = (rl**2+im**2)
# #print(Inty.shape)

# fig, ax = plt.subplots(figsize=(7,5))
# plt.imshow(Inty, extent=[-3*1e3,3*1e3,-3*1e3,3*1e3], cmap="inferno")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

lmda = 500e-9
k = 2*np.pi/lmda
z = 0.1
a = 1e-3                       # half-width of the square aperture

L = 3e-3
n = 601                        # odd, so r = 0 is on the grid
r = np.linspace(-L, L, n)
dx = r[1] - r[0]
x, y = np.meshgrid(r, r)

aperture = ((np.abs(x) <= a) & (np.abs(y) <= a)).astype(float)
kernel = np.exp(1j * k * (x**2 + y**2) / (2*z))

U = signal.fftconvolve(aperture, kernel, mode="same") * dx**2 / (1j*lmda*z)
I = np.abs(U)**2

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(I / I.max(),
               extent=[-L*1e3, L*1e3, -L*1e3, L*1e3],
               origin="lower", cmap="inferno")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
fig.colorbar(im, label="Normalized intensity")
plt.show() 