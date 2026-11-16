import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import scipy
from scipy.special import j1, jn_zeros

r = np.linspace(-2e-3,2e-3,2001)
x,y = np.meshgrid(r,r)

radius = 1e-3
lmda = 500e-9
k = 2*np.pi/lmda
z = 0.5 #1m
A = np.pi*radius**2
a = (2*radius*r)/(lmda*z)


def I(r):
    u = (k * radius * r) / z
    ratio = np.divide(2*j1(u), u, out=np.ones_like(u), where=(u != 0))
    #amp = (A/(lmda*z))**2
    return ratio**2


zero = jn_zeros(1,2)
r0 = (zero[0]*z)/(k*radius)
r1 = (zero[1]*z)/(k*radius)


# # le = int(len(r)/2) + 1
fig, ax = plt.subplots()
# ax.scatter(r*1e3, jn_zeros(1,1))
ax.scatter(r0*1e3, I(r0), color="red")
ax.scatter(r1*1e3, I(r1), color="blue")
ax.scatter(-r0*1e3,I(-r0), color="red")
ax.scatter(-r1*1e3,I(-r1), color="blue")
ax.plot(r*1e3, I(r))
ax.set_ylim(bottom=0)
plt.show()

##========================= 3d version =====================

R = np.sqrt(x**2+y**2)

def I3d(R):
    u = (k * radius * R) / z
    ratio = np.divide(2*j1(u), u, out=np.ones_like(u), where=(u != 0))
    #amp = (A/(lmda*z))**2
    return ratio**2

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(x, y, I3d(R), cmap="coolwarm",
#                        linewidth=0, antialiased=False)
# plt.show()

###========================= image version ======================
fig, ax = plt.subplots(figsize=(7,5))
plt.imshow(I3d(R), extent=[-2*1e3,2*1e3,2*1e3,-2*1e3], cmap="inferno", norm=PowerNorm(gamma=0.6, vmin=0, vmax=1),origin="lower")
plt.show()

