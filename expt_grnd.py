import matplotlib.pyplot as plt
import numpy as np 
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)

x = np.linspace(-1,1,100)
y = np.sin(x*np.pi)

ax.plot(x,y)
ax1.plot(x,y)

plt.show()

## plt.subplots create both figure and axes simultaneously
fig, axe = plt.subplots(2, 2, figsize=(10, 8))

axe[0, 0].plot([1, 2, 3], [1, 4, 9])
axe[0, 1].plot([1, 2, 3], [1, 8, 27])
axe[1, 0].plot([1, 2, 3], [1, 2, 3])
axe[1, 1].plot([1, 2, 3], [3, 2, 1])

plt.tight_layout()
plt.show()