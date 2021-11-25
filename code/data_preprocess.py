import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## 导入数据集
dataset = pd.read_csv('../datasets/Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values

## 处理丢失数据
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

## 解析分类数据
labelencoder_x = LabelEncoder()
X[ : , 0] = labelencoder_x.fit(X[ : , 0])

## 创建虚拟变量
onehotencoder = OneHotEncoder(categories='auto')
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

## 拆分数据集为训练集合和测试集合
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

## 特征向量化
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)