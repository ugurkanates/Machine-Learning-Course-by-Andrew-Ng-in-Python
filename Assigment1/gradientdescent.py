import numpy as np 
import matplotlib.pyplot as plt

arraym = np.loadtxt("ex1data1.txt",delimiter = ",")
arrayZero = arraym[:,0]
arrayOne = arraym[:,1]

#plt.plot(arrayZero, arrayOne, 'x', color='black')
#plt.show()

alpha = 0.01