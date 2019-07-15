import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt    # more on this later

data = pd.read_csv('ex2data1.txt', header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]
#print(data.head())

mask = y == 1
adm = plt.scatter(X[mask][0].values, X[mask][1].values)
not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
#plt.show()

#Sigmoid Function
def sigmoid(x):
  return 1/(1+np.exp(-x))

#Cost Function
#Let’s implement the cost function for the Logistic Regression.
def costFunction(theta, X, y):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta))) 
        + np.multiply((1-y), np.log(1 - sigmoid(X @ theta))))
    return J
#Note that we have used the sigmoid function in the costFunction above.
#There are multiple ways to code cost function. Whats more important is the underlying mathematical ideas and our ability to translate them into code.

def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))
(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]
theta = np.zeros((n+1,1)) # intializing theta with all zeros
J = costFunction(theta, X, y)
print(J)

temp = opt.fmin_tnc(func = costFunction, 
                    x0 = theta.flatten(),fprime = gradient, 
                    args = (X, y.flatten()))
theta_optimized = temp[0]
print(theta_optimized)
#temp = opt.fmin_tnc
J = costFunction(theta_optimized[:,np.newaxis], X, y)
print(J)