import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(-10,10,100)
x,y = np.meshgrid(r,r)

x0,y0 = 0,0
radius = 5

circle = (x - x0)**2+(y - y0)**2 <= radius**2