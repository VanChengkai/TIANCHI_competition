#coding:utf-8
import pandas as pd
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

#read_table 会去掉制表符
test_data = pd.read_csv('zhengqi_test.csv')

#print(data1)

train_data = pd.read_csv('zhengqi_train.csv')

#print(train_data)

#输入参数读入
x_train = train_data.iloc[0:,:38]
#target读入
y_train = train_data.iloc[0:,38]
#print(x)
#print(y)

x_test = test_data.iloc[0:,:38]
#target读入

#划分训练集
#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


#标准化
from sklearn.preprocessing import StandardScaler
ss_x,ss_y = StandardScaler(),StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train[:].values.reshape([-1,1])).reshape(-1)
#y_test = ss_y.transform(y_test[:].values.reshape([-1,1])).reshape(-1)
print(y_train.shape)

from sklearn.svm import SVR

l_svr = SVR(kernel='poly')
l_svr.fit(x_train,y_train)
#print("MSE_lsvr:",metrics.mean_squared_error(l_svr.predict(x_test),y_test))
y_test = l_svr.predict(x_test)

np.savetxt("ans.txt",y_test,fmt="%.3f")


'''
n_svr = SVR(kernel='poly')
n_svr.fit(x_train,y_train)
print("MSE_nsvr:",metrics.mean_squared_error(n_svr.predict(x_test),y_test))

r_svr = SVR(kernel='rbf')
r_svr.fit(x_train,y_train)
print("MSE_rsvr:",metrics.mean_squared_error(r_svr.predict(x_test),y_test))

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(weights="uniform")
knn.fit(x_train,y_train)
print("MSE_knn:",metrics.mean_squared_error(knn.predict(x_test),y_test))

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
print("MSE_tree:",metrics.mean_squared_error(dt.predict(x_test),y_test))


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)
print("MSE_forest:",metrics.mean_squared_error(rfr.predict(x_test),y_test))

from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()
etr.fit(x_train,y_train)
print("MSE_ExtraTreesRegressor:",metrics.mean_squared_error(etr.predict(x_test),y_test))


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(x_train,y_train)
print("MSE_GradientBoostingRegressor:",metrics.mean_squared_error(etr.predict(x_test),y_test))
'''