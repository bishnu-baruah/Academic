import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import j1, jn_zeros

n = 401
r = np.linspace(-2e-3,2e-3,n)
x,y = np.meshgrid(r,r)

radius = 1e-3
lmda = 500e-9
k = 2*np.pi/lmda
z = 2.0 #1m
A = np.pi*radius**2
a = (2*radius*r)/(lmda*z)

R = np.sqrt(x**2+y**2)
nu = (k * radius * R) / z

def I3d(R):
    u = (k * radius * R) / z
    ratio = np.divide(2*j1(u), u, out=np.ones_like(u), where=(u != 0))
    #amp = (A/(lmda*z))**2
    return ratio**2

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(x*1e3, y*1e3, I3d(R), cmap="coolwarm",
#                       linewidth=0, rcount=0.25*n, ccount=0.25*n, antialiased=False)
plt.imshow(I3d(R), extent=[-2,2,2,-2], cmap="hot")
plt.show()