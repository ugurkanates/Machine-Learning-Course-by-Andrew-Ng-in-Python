import numpy as np 
import matplotlib.pyplot as plt

steps = 1500 # how many steps it wants to
alfa = 0.001
#param = [-1,2] # Q0 and Q1 x (q1)
param = np.zeros([2,1])


arraym = np.loadtxt("ex1data1.txt",delimiter = ",")
arrayZero = arraym[:,0]
arrayZero = arrayZero[:,np.newaxis]
arrayOne = arraym[:,1]
arrayOne = arrayOne[:,np.newaxis]
#inputVector is npArray which contains  VALUE = X features array (we only have one in this case)
#We currently only have one input vector so pretend only one feature.
#param vector is o0 o1 o2 .. 
def hFunc(inputVector,param):
    res = 0
    size = inputVector.size +1
    for i in range(size): # this way no modification to inputVector like passing 1 more element as 1
        if i == 0: 
            res = param[i] * 1 # inputvector first element is always 1
        else: 
            res  = res+ param[i] * inputVector.item(i-1)
        #print("res",res,"param0",param[0],"param1*inp",param[1]*inpute,"\n")

    return res
def cost(m,arrayZero,arrayOne,param):
    value = 0
    for i in range (m):
       testInputVector = np.array(arrayZero[i])
       value = value + (hFunc(testInputVector,param)-arrayOne[i])**2
       #print("value = ",value,"hfunici",arrayZero[i],"hfuncsonuc",)
    return (1/(2*m))*value
#print(cost(arrayZero.size,arrayZero,arrayOne))   
def gradient(param,inputVector):
    paramOld = np.copy(param) # q0,q1,q2.. will be updated so copied.
    # hoxo -yi calculate here for first time it will be same for all parameter updates
    hValue = np.zeros(inputVector.size) # dataset eleman sayisi 1'den m'e kadar . size m, dataset satiri kdr
    for i in range(hValue.size):
        testInputVector = np.array(arrayZero[i])
        hValue[i] = hFunc(testInputVector,paramOld) - arrayOne[i] # 0 index olanlar actually dataset sirasinda 1 elemanlar
    #simultainus update icin baska bir sy vardi tensorflow kullananbilir tf start vs
    for i in range(len(param)):
        value = 0
       
        if i == 0: # x0 = 1 always
            value = value + hValue.item(i)*1 # * x0
        else:
            value = value + hValue.item(i)*inputVector.item(i-1) # * x1 x2 x3 ...  parametre sayisi kadar

        param[i] = paramOld[i] - (alfa/inputVector.size)*value
print("First cost : ",cost(arrayZero.size,arrayZero,arrayOne,param),"\n")  
for i in range(steps):
    gradient(param,arrayZero)
    print(i," cost : ",cost(arrayZero.size,arrayZero,arrayOne,param),"\n")
print("param 0", param[0],"--param 1",param[1])
plt.plot(arrayZero, arrayOne, '.', color='black')


#plt.scatter(param[0],param[1])
#plt.plot(param[0],param[1],'-r')
#plt.show()
plt.scatter(arrayZero[:,1], arrayZero)
plt.plot(arrayZero, np.dot(arrayZero, param))


