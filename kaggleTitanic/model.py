import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
import seaborn as sns


data = pd.read_csv("train.csv")
#data.info()
#print(data.head(n=5))

sns.countplot(x='Survived', data=data)
#plt.show()
print(data.shape)