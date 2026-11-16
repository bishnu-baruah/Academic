import numpy as np
from scipy import integrate as int

x = np.linspace(-2e-3,2e-3,10)
lmda = 500e-9
k = 2*np.pi/lmda
z = 1
A = (1j*k)/(2*z)

f = lambda x,e: np.exp(A*(e-x)**2)
re = []
for i in x:
    re.append((int.quad(f, -1,1, args=(i,))[0])**2)
#print(int.quad(f, -1,1, args=(1,)))
print(re)