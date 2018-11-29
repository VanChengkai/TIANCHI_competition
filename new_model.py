from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import sklearn.neural_network as nn

#导入数据
zhengqi_train = pd.read_table('./zhengqi_train.txt',encoding='utf-8')
zhengqi_test = pd.read_table('./zhengqi_test.txt',encoding='utf-8')

#数据分割
X = np.array(zhengqi_train.drop('target', axis = 1))
y = np.array(zhengqi_train.target)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print(len(X_train))
print(len(X_test))


#进行PCA降维
#pca = PCA(n_components=0.99)
#pca.fit(X)
#X_pca = pca.transform(X)
#X1_pca = pca.transform(zhengqi_test)

#X_train,X_test,y_trian,y_test = train_test_split(X_pca,y,test_size=0.3,random_state=0)

#线性回归拟合
clfL = LinearRegression()
#model = nn.MLPClassifier(activation='relu',solver='adam',alpha=0.0001,learning_rate='adaptive',learning_rate_init=0.001,max_iter=1000)
#clfL = SVR(kernel="poly")

clfL.fit(X_train,y_train)
#model.fit(X_train,y_train)

y_true,y_pred = y_test,clfL.predict(X_test)
print(metrics.mean_squared_error(y_true,y_pred))

# 预测输出
ans_Liner = clfL.predict(zhengqi_test)
print(ans_Liner.shape)
df = pd.DataFrame(ans_Liner)
df.to_csv('ans.txt',index=False,header=False)