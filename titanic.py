#!/usr/bin/env python
# coding: utf-8

# In[317]:


# Library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


# In[221]:


# Data

train = pd.read_csv("train_titanic.csv")


# In[222]:


df = pd.read_csv("train_titanic.csv")


# In[223]:


train.shape


# In[224]:


train.head()


# In[225]:


train.describe()


# <b> Missing data

# In[6]:


train.isnull().sum()/train.shape[0]*100


# In[230]:


train['hasCabin'] = train.Cabin.map(lambda x: 1 if str(x)!='nan' else 0)


# In[235]:


train['CabinCat'] = train.Cabin.map(lambda x: x[0] if str(x)!='nan' else 0)


# In[245]:


tmp = train.groupby(['CabinCat'])['Fare'].mean().sort_values().reset_index().reset_index().drop('Fare',axis=1)


# In[248]:


train = train.merge(tmp,on='CabinCat',how='right')


# In[251]:


train = train.rename(columns={"index": "CabienCatNum"})


# <b> Drop Cabin, fill Age with median, and fill Embarked with the most common data

# In[253]:


train = train.drop("Cabin",axis=1)
train = train.fillna(train.mean())


# In[254]:


train.Embarked.value_counts()


# In[255]:


train.Embarked[train.Embarked.isnull()]


# In[256]:


train = train.fillna(train.mode().iloc[0])


# In[257]:


train.Embarked[train.Embarked.isnull()]


# <b> Categorical columns

# In[258]:


train.dtypes


# In[259]:


train.Sex = pd.get_dummies(train.Sex)['female']


# In[261]:


train = pd.concat([train, pd.get_dummies(train.Embarked)],axis=1).drop('Embarked',axis=1)


# In[262]:


train['LastName'] = train.Name.map(lambda x: x.split(',')[0])


# In[263]:


le = preprocessing.LabelEncoder()
lefit = le.fit(train['LastName'])
train['LastNameEncoding'] = lefit.transform(train['LastName'])
train = train.drop(['Name','LastName'],axis=1)


# In[264]:


train.dtypes


# In[265]:


train = train.drop('Ticket',axis=1)


# <b> Imbalanced class

# In[266]:


train.Survived.value_counts().plot(kind="bar")


# In[267]:


notsurvived = train.Survived.value_counts()[0]/train.Survived.value_counts().sum() * 100
survived = train.Survived.value_counts()[1]/train.Survived.value_counts().sum() * 100


# In[268]:


print("{} % did not survived (class 0), i.e {} indiv".format(round(notsurvived,2),train.Survived.value_counts()[0]))
print("{} % did survived (class 0), i.e {} indiv".format(round(survived,2),train.Survived.value_counts()[1]))


# In[269]:


rs = np.random.RandomState(0)
corr = train.corr()
sns.heatmap(corr, cmap='coolwarm_r')
plt.show()


# In[272]:


sns.boxplot(x=train.Survived, y=train.Age,orient = "v",width = 0.2)
#sns.boxplot(train.Sex,orient = "v",width = 0.2) 


# In[315]:


train.Age.hist()


# <b> Undersampling

# In[273]:


undersampling = train[train.Survived==0][:342]


# In[274]:


X = pd.concat([undersampling, train[train.Survived==1]]).reset_index(drop=True)


# In[280]:


X = X.drop('CabinCat',axis=1)


# <b> Split train/test

# In[281]:


X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,X.columns!='Survived'], 
                                                    X.Survived, test_size=0.2, random_state=42)


# In[282]:


models = {
    'RandomForest' : RandomForestClassifier(),
    'AdaBoost' : AdaBoostClassifier(),
    'Knn' : KNeighborsClassifier()
}


# In[283]:


scores = []
for model in models:
    scores.append(cross_validate_score(models[model], X_train, y_train, cv=5,
                   scoring=('accuracy'))['test_score'].mean())


# In[284]:


for i,model in enumerate(models):
    print(f"Accuracy test_score for {model} : {round(scores[i],3)}")


# In[285]:


rdf = RandomForestClassifier().fit(X_train,y_train)
features = X_train.columns
importances = rdf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# <b> With dummies Last Name

# In[286]:


train['LastName'] = df.Name.map(lambda x: x.split(',')[0])


# In[287]:


train_x = pd.concat([train,pd.get_dummies(train.LastName)],axis=1).drop(['LastName'],axis=1)


# In[288]:


undersampling_X = train_x[train_x.Survived==0][:342]


# In[289]:


X_x = pd.concat([undersampling_X, train_x[train_x.Survived==1]]).reset_index(drop=True)


# In[290]:


X_x = X_x.drop(["LastNameEncoding",'CabinCat'],axis=1)


# In[126]:


from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=3).fit_transform(X_x.iloc[:,X_x.columns!='Survived'])


# In[291]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X_x.iloc[:,X_x.columns != 'Survived'], 
                                                        X_x.Survived, test_size=0.2, random_state=42)


# In[292]:


scores1 = []
for model in models:
    scores1.append(cross_validate(models[model], X1_train, y1_train, cv=5,
                   scoring=('accuracy'))['test_score'].mean())


# In[293]:


for i,model in enumerate(models):
    print(f"Accuracy test_score for {model} : {round(scores1[i],3)}")


# In[294]:


rdf = RandomForestClassifier().fit(X1_train,y1_train)
feat_importances = pd.Series(rdf.feature_importances_, index=X1_train.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# <b> GridSearch parameters

# In[319]:


params_rf = {'n_estimators':[10,20,30,40,50,60,70,80,90],
            'criterion': ['gini', 'entropy'],
             'max_depth':[2,3,4,5],
             'min_weight_fraction_leaf':[0.,1.],
             'max_features': ['auto', 'sqrt', 'log2']
            }
rf_grid = GridSearchCV(RandomForestClassifier(), params_rf)
rf_fit = rf_grid.fit(X_train, y_train)


# In[320]:


rf_fit.best_params_ 


# In[321]:


rf_fit.best_score_


# In[322]:


y_pred_rf = rf_fit.predict(X_test)
rf_cf = confusion_matrix(y_test, y_pred_rf)


# In[331]:


fig = plt.plot(figsize=(22,12))
sns.heatmap(rf_cf, annot=True, cmap=plt.cm.copper)
plt.title("Random Forest \n Confusion Matrix", fontsize=14)

plt.show()


# In[193]:


params_knn = {'n_neighbors':[3,5,7,9,11,15,17,19,21,23,25,30,35,40,45],
            'weights': ['uniform', 'distance'],
             'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
            }
knn_grid = GridSearchCV(KNeighborsClassifier(), params_knn)
knn_fit = knn_grid.fit(X_train, y_train)


# In[194]:


knn_fit.best_params_


# In[195]:


knn_fit.best_score_


# In[295]:


knn_fit_ = KNeighborsClassifier().fit(X_train, y_train)


# In[304]:


knn_fit_.score(X_train, y_train)


# In[306]:


k_feat = [3,5,7,9,11,15,17,19,21,23,25,30,35,40,45]
knn_scores = [] 
for k in k_feat:
    knn_fit_ = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    knn_scores.append(knn_fit_.score(X_train, y_train))


# In[309]:


plt.plot(k_feat,knn_scores)


# In[310]:


np.argmax(knn_scores)


# In[314]:


knn_fit_ = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
knn_fit_.score(X_train, y_train)


# # Steps

# In[334]:


from sklearn.model_selection import cross_val_score


# In[345]:


models = {
    'RandomForest' : RandomForestClassifier(),
    'AdaBoost' : AdaBoostClassifier(),
    'Knn' : KNeighborsClassifier()
}


# In[346]:


scores_cv = []
for model in models:
    scores_cv.append(cross_val_score(models[model], X_train, y_train, cv=5,).mean())


# In[347]:


for i,model in enumerate(models):
    print(f"Accuracy test_score for {model} : {round(scores_cv[i],3)}")


# In[391]:


params_rf = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],
            'criterion': ['gini', 'entropy'],
             'max_depth':[2,3,4,5,None],
             'min_weight_fraction_leaf':[0.,0.3,0.5],
             'max_features': ['auto', 'sqrt', 'log2']
            }
rf_grid = GridSearchCV(RandomForestClassifier(), params_rf)
rf_fit = rf_grid.fit(X_train, y_train)


# In[392]:


rf_cv = cross_val_score(rf_fit.best_estimator_,X_train,y_train)


# In[393]:


rf_cv.mean()


# In[394]:


rf_best = rf_fit.best_estimator_.fit(X_train,y_train)


# In[395]:


y_pred_rf = rf_best.predict(X_test)
rf_cf = confusion_matrix(y_test, y_pred_rf)


# In[396]:


fig = plt.plot(figsize=(22,12))
sns.heatmap(rf_cf, annot=True, cmap=plt.cm.copper)
plt.title("Random Forest \n Confusion Matrix", fontsize=14)

plt.show()


# In[397]:


from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc


# In[398]:


print(f"Accuracy score: {round(accuracy_score(y_test, y_pred_rf),3)}")
print(f"Recall score: {round(recall_score(y_test, y_pred_rf),3)}")
print(f"Precision score: {round(precision_score(y_test, y_pred_rf),3)}")


# In[399]:


rf_pred_proba = rf_best.predict_proba (X_test)


# In[400]:


fpr_rf, tpr_rf, _ = roc_curve(y_test,rf_pred_proba[:,1])


# In[401]:


auc_rf = round(auc(fpr_rf, tpr_rf),3)


# In[402]:


plt.plot(fpr_rf, tpr_rf,label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(f'ROC curve, AUC {auc_rf}')
plt.legend(loc='best')
plt.show()


# In[ ]:




