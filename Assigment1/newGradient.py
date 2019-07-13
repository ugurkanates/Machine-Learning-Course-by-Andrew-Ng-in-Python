import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt', header = None) #read from dataset
X = data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m = len(y) # number of training example
data.head() # view first few rows of the data

plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
#plt.show()

X = X[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 15000
alpha = 0.001
ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term
#print(X)

def computeCostMulti(X, y, theta):
    temp = np.dot(X, theta) -y
    return np.sum(np.power(temp, 2)) / (2*m)
J = computeCostMulti(X, y, theta)
print(J)

def gradientDescent(X, y, theta, alpha, iterations):
    for i in range(iterations):
        temp = np.dot(X, theta) - y #this code is or*asm ! wow fuckin python lol
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
        print("i'th train:",i,computeCostMulti(X,y,theta))
    return theta
theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
#plt.show()

print("What would be for let say 5000 city people profit? \n")
myfunc = np.ones([2,1])
#myfunc[0][0] = 5000
#a = np.sum(np.dot(theta,myfunc))
print(theta[0]+theta[1]*5000)