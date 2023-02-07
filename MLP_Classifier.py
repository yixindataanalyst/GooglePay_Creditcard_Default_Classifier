
from this import d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, classification_report
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split




data_train = pd.read_csv('C:/Users/79298/Desktop/Google实习/Week 1/data.csv')
data_train.describe()
train = data_train.copy()
train



#数据类型
numerical_fea = list(train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(train.columns)))



#过滤数值型类别特征
def get_numerical_serial_fea(data,feas):
    numerical_serial_fea = []
    numerical_noserial_fea = []
    for fea in feas:
        temp = data[fea].nunique()
        if temp <= 10:
            numerical_noserial_fea.append(fea)
            continue
        numerical_serial_fea.append(fea)
    return numerical_serial_fea,numerical_noserial_fea
numerical_serial_fea,numerical_noserial_fea = get_numerical_serial_fea(train,numerical_fea)



numerical_fea = list(train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(train.columns)))
label = 'title'
numerical_fea.remove(label)





train.isnull().sum()




#按照中位数填充数值型特征
train[numerical_fea] = train[numerical_fea].fillna(train[numerical_fea].median())
#按照众数填充类别型特征
train[category_fea] = train[category_fea].fillna(train[category_fea].mode())


train['grade'] = train['grade'].map({'A':10,'B':20,'C':30,'D':40,'E':50,'F':60,'G':70})

train['subGrade'] = train['subGrade'].map({'A1':10,'A2':12,'A3':14,'A4':16,'A5':18,'B1':20,'B2':22,'B3':24,'B4':26,'B5':28,'C1':30,'C2':32,'C3':34,'C4':36,'C5':38,'D1':40,'D2':42,'D3':44,'D4':46,'D5':48,'E1':50,'E2':52,'E3':54,'E4':56,'E5':58,'F1':60,'F2':62,'F3':64,'F4':66,'F5':68,'G1':70,'G2':72,'G3':74,'G4':76,'G5':78})

train['title']= train['title'].replace(np.nan, 0)
train.isnull().sum()

# 类型数在2之上，又不是高维稀疏的,且纯分类特征
for data in [train,train]:
    data = pd.get_dummies(data, columns=['subGrade', 'purpose','grade'], drop_first=True)
data['title']= data['title'].replace(np.nan, 0)
data=data.drop(columns=['earliesCreditLine'],axis = 1)
X, y = data.iloc[:, np.r_[0:6,8:68]].values, data.iloc[:, 7].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)#随机选择25%作为测试集，剩余作为训练集

##逻辑回归 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
print(model.coef_, model.intercept_)

model.predict(X_test)
 ##发现模型的准确率在80%的水平 
print(model.score(X_test, y_test))
 
 ##我们接下去采用skilearn去进行模型的拟合
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe_lr1 = make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
pipe_lr1.fit(X_train,y_train)
y_pred1 = pipe_lr1.predict(X_test)
print("Test Accuracy: %.3f"% pipe_lr1.score(X_test,y_test))
##Test Accuracy: 0.800


##k折交叉验证
from sklearn.model_selection import cross_val_score

scores1 = cross_val_score(estimator=pipe_lr1,X = X_train,y = y_train,cv=10,n_jobs=1)
print("CV accuracy scores:%s" % scores1)
print("CV accuracy:%.3f +/-%.3f"%(np.mean(scores1),np.std(scores1)))

##CV accuracy Score: [0.80056667 0.80108333 0.8009     0.80145    0.80028333 0.80086667
 0.80066667 0.8008     0.80086667 0.8005    ] 

#### 分层k折交叉验证
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10,random_state=1,shuffle=True).split(X_train,y_train)
scores2 = []
for k,(train,test) in enumerate(kfold):
    pipe_lr1.fit(X_train[train],y_train[train])
    score = pipe_lr1.score(X_train[test],y_train[test])
    scores2.append(score)
    print('Fold:%2d,Class dist.:%s,Acc:%.3f'%(k+1,np.bincount(y_train[train]),score))
print('nCV accuracy :%.3f +/-%.3f'%(np.mean(scores2),np.std(scores2)))




### 用学习曲线诊断偏差与方差
pipe_lr3 = make_pipeline(StandardScaler(),LogisticRegression(random_state=1,penalty='l2'))
train_sizes,train_scores,test_scores = learning_curve(estimator=pipe_lr3,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1,10),cv=10,n_jobs=1)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='red',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='red')
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.ylim([0.8,1.02])
plt.show()

# 用验证曲线解决欠拟合和过拟合
from sklearn.model_selection import validation_curve

pipe_lr3 = make_pipeline(StandardScaler(),LogisticRegression(random_state=1,penalty='l2'))
param_range = [0.001,0.01,0.1,1.0,10.0,100.0]
train_scores,test_scores = validation_curve(estimator=pipe_lr3,X=X_train,y=y_train,param_name='logisticregression__C',param_range=param_range,cv=10,n_jobs=1)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(param_range,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(param_range,test_mean,color='red',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='red')
plt.xscale('log')
plt.xlabel("Parameter C")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.ylim([0.8,1.02])
plt.show()


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#需要调优的参数
#请尝试将L1正则和L2正则分开，并配合合适的优化求解算法（solver）
#tuned_parameters={'penalties':['l1','l2'],'C':[0.001,0.01,0.1,1,10,100,1000]}
#参数的搜索范围
penalties=['l1','l2']
Cs=[0.001,0.01,0.1,1,10,100,1000]
#调优的参数集合，搜索网格为2x5，在网格上的交叉点进行搜索
tuned_parameters=dict(penalty=penalties,C=Cs)


lr_penalty=LogisticRegression(solver='liblinear')
grid=GridSearchCV(lr_penalty,tuned_parameters,cv=3,scoring='neg_log_loss',n_jobs=4)
grid.fit(X_train,y_train)
print(-grid.best_score_)
print(grid.best_params_)
####print(-grid.best_score_)
####0.4563801562259693
##### print(grid.best_params_)
####{'C': 1, 'penalty': 'l1'}


pipe_lr1 = make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1,penalty='l1',solver='liblinear'))
pipe_lr1.fit(X_train,y_train)
y_pred1 = pipe_lr1.predict(X_test)
print("Test Accuracy: %.3f"% pipe_lr1.score(X_test,y_test))
##Test Accuracy: 0.800

#######################GBDT 模型
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X_train,y_train)
y_pred = gbm0.predict(X_train)
y_predprob = gbm0.predict_proba(X_train)[:, 1]
print("Test Accuracy: %.3f"% gbm0.score(X_test,y_test)) #Test Accuracy: 0.802
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob))  ###AUC Score (Train): 0.711272

param_test1 = {'n_estimators': list(range(20, 81, 10))}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X_train, y_train)
print(gsearch1.best_params_, gsearch1.best_score_)
####得到最优n_estimator 为60 

param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, min_samples_leaf=20, 
      max_features='sqrt', subsample=0.8, random_state=10), 
   param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch2.fit(X_train, y_train)
gsearch2.best_params_, gsearch2.best_score_
##得到最优决策树深度以及最小节点划分

#######mlp 模型

from sklearn.neural_network import MLPClassifier

import numpy as np
import scipy.io
mlp = MLPClassifier(solver='adam', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50,50), random_state=1,max_iter=11,verbose=10,learning_rate_init=.1)

mlp.fit(X_train, y_train)

print (mlp.score(X_train, y_train))
print (mlp.n_layers_)
print (mlp.n_iter_)
print (mlp.loss_)
##调整超参数
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=100)
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)
print('Best parameters found:\n', clf.best_params_)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
