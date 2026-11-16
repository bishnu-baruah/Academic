import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as int

x = np.linspace(-1e-3,1e-3,100)
lmda = 500e-9
k = 2*np.pi/lmda
z = 2
A = (1j*k)/(2*z)

f = lambda x,e: np.exp(A*(e-x)**2)
re = []
for i in x:
    re.append((int.quad(f, -1e-3,1e-3, args=(i,), complex_func=True)[0])**2)
#print(int.quad(f, -1,1, args=(1,)))
#print(re)


fig, ax = plt.subplots()
ax.plot(x*1e3,re)
plt.show()