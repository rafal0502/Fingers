import numpy as np
import differint.differint as df
from matplotlib import pyplot as plt
import math
from scipy.ndimage import gaussian_filter
import random


def f(x):
   return x**0.5

v = 0.8

DF = df.RL(v, f)
print(DF)


DF = df.RL(v, f, 0, 1, 4)
print(DF)

#
# gauss_filter = np.matrix([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
# print(gauss_filter)


def gaussian(x, alpha, r):
   return 1. / (math.sqrt(alpha ** math.pi)) * np.exp(-alpha * np.power((x - r), 2.))


x = np.linspace(1, 4, 9)
plt.plot(x, gaussian(x, 1, 0))
plt.show()


gaussian_matrix = gaussian(x, 1, 0).reshape(3,3)
print(gaussian(x, 1, 0))


DF = df.RL(0.5, gaussian(x, 1, 0))
print(DF)
plt.plot(x, DF)
plt.show()



DF = df.RLpoint(0.5, f)
print(DF)


differ_matrix = df.RLpoint(0.5, f)
print(differ_matrix)


def gauss_2d(mu, sigma):
   x = random.gauss(mu, sigma)
   y = random.gauss(mu, sigma)
   return (x, y)


print(gaussian_filter(gauss_filter, sigma=1))
print(gauss_2d(273,1))


def gaussian_kernel(size, size_y=None):
   size = int(size)
   if not size_y:
      size_y = size
   else:
      size_y = int(size_y)
   x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
   g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size_y)))
   return g / g.sum()

print(gaussian_kernel(5))


from scipy.ndimage import gaussian_filter
a = np.ones(9).reshape((3, 3))
kernel = np.matrix([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
print(a)
x = gaussian_filter(kernel, sigma=-2)
print(x)

DF = df.RL(0.5, kernel, 1, 5, 3)
print(DF)


def f(x):
   return x


DF = df.RLpoint(0.5, f)
print(DF)
