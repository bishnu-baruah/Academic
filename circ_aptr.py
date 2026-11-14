import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import j1

r = np.linspace(-2e-3,2e-3,201)
x,y = np.meshgrid(r,r)

radius = 10e-3
lmda = 500e-9
k = 2*np.pi/lmda
z = 1.0

fig,ax = plt.subplots()
a = k*r*radius/z
a_ = 2*r*radius/(lmda*z)
b = (2*j1(a)/a)**2
ax.plot(a_,b)
plt.show()
