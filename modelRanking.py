import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import pickle

data = pd.read_csv("2017RankingEngg.csv")
data0 = pd.read_csv("2016RankingEngg.csv")
data2 = pd.read_csv("2018RankingEngg.csv")
data3 = pd.read_csv("2019RankingEngg1.csv")
data4 = pd.read_csv("2020RankingEngg.csv")
data5 = pd.read_csv("2021_Engg.csv")


c0 = np.array([data0.iloc[0]])
for i in range(99):
    x = np.array(data0.iloc[i+1])
    c0 = np.concatenate((c0,[x]),axis=0)

c1 = np.array([data.iloc[0]])
for i in range(data.shape[0]-1):
    x = np.array(data.iloc[i+1])
    c1 = np.concatenate((c1,[x]),axis=0)
    
c2 = np.array([data2.iloc[0]])
for i in range(99):
    x = np.array(data2.iloc[i+1])
    c2 = np.concatenate((c2,[x]),axis=0)
    
c3 = np.array([data3.iloc[0]])
for i in range(168):
    x = np.array(data3.iloc[i+1])
    c3 = np.concatenate((c3,[x]),axis=0)

c4 = np.array([data4.iloc[0]])
for i in range(data4.shape[0]-1):
    x = np.array(data4.iloc[i+1])
    c4 = np.concatenate((c4,[x]),axis=0)

c5 = np.array([data5.iloc[0]])
for i in range(data5.shape[0]-1):
    x = np.array(data5.iloc[i+1])
    c5 = np.concatenate((c5,[x]),axis=0)

averaged_data = np.zeros((169,5))
for i in range(100):
    for j in range(5):
        averaged_data[i,j] = (c0[i,j]*100/max(c0[:,j])+c1[i,j]*100/max(c1[:,j])+c2[i,j]*100/max(c2[:,j])+c3[i,j]*100/max(c3[:,j])+c4[i,j]*100/max(c4[:,j]))/5.0

for i in range(100,169):
    for j in range(5):
        averaged_data[i,j] = (c3[i,j]*100/max(c3[:,j]) + c4[i,j]*100/max(c4[:,j]))/2.0
        
d = np.array([[1]])
for i in range(168):
    d = np.concatenate((d,[[i+1]]),axis=0)

sc = np.zeros((169,1))
for i in range(169):
    sc[i,0] = 0.3*averaged_data[i,0]+0.3*averaged_data[i,1]+0.2*averaged_data[i,2]+0.1*averaged_data[i,3]+0.1*averaged_data[i,4]
    
test = np.zeros((200,1))
c_test = c5*100/[max(c4[:,0]),max(c4[:,1]),max(c4[:,2]),max(c4[:,3]),max(c4[:,4])]
for i in range(200):
    test[i,0] = 0.3*c_test[i,0]+0.3*c_test[i,1]+0.2*c_test[i,2]+0.1*c_test[i,3]+0.1*c_test[i,4]

poly = PolynomialFeatures(degree = 4)
regr = linear_model.LinearRegression()
x = poly.fit_transform(sc)
regr.fit(x, d)

file_name = "models/rank_model.pkl"
file_name1="models/polynomial_transform.pkl"

pickle.dump(regr,open(file_name,"wb"))
pickle.dump(poly,open(file_name1,"wb"))