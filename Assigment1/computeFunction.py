import numpy as np 
import matplotlib.pyplot as plt

arraym = np.loadtxt("ex1data1.txt",delimiter = ",")
arrayZero = arraym[:,0]
arrayOne = arraym[:,1]
def hFunc(inpute):
    param = [0,0]
    res  = param[0] * param[1]*inpute
    #print("res",res,"param0",param[0],"param1*inp",param[1]*inpute,"\n")
    return res
def compute(m,arrayZero,arrayOne):
    value = 0
    for i in range (m):
       value = value + (hFunc(arrayZero[i])-arrayOne[i])**2
       #print("value = ",value,"hfunici",arrayZero[i],"hfuncsonuc",)
    return (1/(2*m))*value
print(compute(arrayZero.size,arrayZero,arrayOne))   