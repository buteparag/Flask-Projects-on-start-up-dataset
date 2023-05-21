import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

df=pd.read_csv("Startups.csv")
x=df.iloc[:,:-2]
y=df.iloc[:,-1]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=0)
def mymodel(model):
    model.fit(xtrain,ytrain)
    return model
def makeprediction():
    knn=KNeighborsRegressor(n_neighbors=2)
    model=mymodel(knn)
    return model
