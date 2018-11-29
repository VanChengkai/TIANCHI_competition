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
x = train_data.iloc[0:,:38]
#target读入
y = train_data.iloc[0:,38]
#print(x)
#print(y)



#划分训练集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#print(len(x_train))
#print(len(x_test))

#数据归一化
x = x - x.min()
x = x / x.max()


#cls = LogisticRegression()
cls = LinearRegression()
cls.fit(x_train,y_train.astype('int'))

cls2 = PolynomialFeatures(degree=3)
cls2.fit(x_train,y_train.astype('int'))

#print("Coefficients:%s, intercept %s"%(cls.coef_,cls.intercept_))

#print("Residual sum of squares: %.2f"% np.mean((cls.predict(x_test) - y_test) ** 2))
print("MSE:",metrics.mean_squared_error(cls.predict(x_test),y_test))
print("MSE2:",metrics.mean_squared_error(cls.predict(cls2),y_test))
#print('Score: %.2f' % cls.score(x_test, y_test))
