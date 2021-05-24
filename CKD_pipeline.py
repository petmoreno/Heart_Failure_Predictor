# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:16:20 2020

@author: k5000751
"""


#CKD_pipeline.py file aimed at reproduce performance of CKD_script through pipeline
#to improve modularity

## All necessary modules as well as different functions that will be used in this work are explicit here.
#import all neccesary modules
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

#import modules created 
import my_utils
import missing_val_imput
import feature_select
import preprocessing
import adhoc_transf

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Classifier models to use
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

import joblib

#%matplotlib inline 


#importing file into a pandas dataframe# As being unable to extract data from it original source, the csv file is downloaded from
#https://www.kaggle.com/mansoordaku/ckdisease
path_data=r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease\Chronic_Kidney_Disease\kidney_disease.csv'
df=pd.read_csv(path_data)
df.head()
df.describe()
df['classification'].value_counts()

#Set column id as index
df.set_index('id', inplace=True)

# Lets see summary of data
df.describe()

#Looking at describe table we can see that there are some missing features that apparently have numerical values. Let's see the
#type of these features, apart from the proportion of non-null values
my_utils.info_adhoc(df)

#As seen above, there are some strange caracters in pcv feature, therefore we will explore every features' value to homogeneize it.
my_utils.df_values(df)

#############################
##Step 0 Train-Test splitting
#############################
#Before starting to clean data, lets split train set and data set with stratrification on y=classification

train_set,test_set=train_test_split(df, test_size=0.3, random_state=42, stratify=df["classification"])


# from sklearn.model_selection import StratifiedShuffleSplit

# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(df, df["classification"]):
#     strat_train_set = df.loc[train_index]
#     strat_test_set = df.loc[test_index]
    
train_set['classification'].value_counts()
test_set['classification'].value_counts()
    

# strat_train_set['classification'].value_counts()
# strat_test_set['classification'].value_counts()

train_set_copy=train_set.copy()
test_set_copy=test_set.copy()

X_train=train_set_copy.drop('classification',axis=1)
y_train=train_set_copy['classification'].copy()

X_test=test_set_copy.drop('classification',axis=1)
y_test=test_set_copy['classification'].copy()

#############################
##Step 1 Misspelling correction and Encoding target feature
#############################
#Correct any misspelling correction in y_train
def misspellingCorrector(df):
    df.iloc[:] = df.iloc[:].str.replace(r'\t','')
    df.iloc[:] = df.iloc[:].str.replace(r' ','')
    return df

y_train=misspellingCorrector(y_train)

label_enc=LabelEncoder()
y_train=label_enc.fit_transform(y_train)
label_enc.classes_

#############################
##Step 2 Feature Engineering
#############################
#Cross_val_score fails due to features al and su has only few samples of values 5.0. So we have to cast to previous category
#X_train.loc[:,'al'].replace(5,4,inplace=True)
#X_train.loc[:,'su'].replace(5,4,inplace=True)
#############################
##Step 3 Pipeline creation for data preparation
#############################

print('Creating the data preparation Pipeline')

numerical_features=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
category_features= ['sg','al','su','rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
len(category_features)
pipeline_numeric_feat= Pipeline([('mispelling',adhoc_transf.misspellingTransformer()),
                                 ('features_cast',adhoc_transf.Numeric_Cast_Column()),
                                 ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median')),
                                 ('features_select',feature_select.Feature_Selector(strategy='wrapper_RFECV')),
                                 ('scaler', MinMaxScaler())
                        ])

pipeline_category_feat= Pipeline([('mispelling',adhoc_transf.misspellingTransformer()),
                                 ('features_cast',adhoc_transf.Category_Cast_Column()),
                                 ('data_missing',missing_val_imput.Category_Imputer(strategy='most_frequent')),
                                 ('cat_feat_engineering',adhoc_transf.CastDown()),
                                 ('encoding', OrdinalEncoder()),
                                 ('features_select',feature_select.Feature_Selector(strategy='wrapper_RFECV'))
                        ])

dataprep_pipe=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_features),
                                 ('category_pipe',pipeline_category_feat, category_features)
                                ])


#For testing data_prep pipelines individually
#X_train1=pipeline_numeric_feat.fit_transform(X_train[numerical_features],y_train)
#X_train1=pipeline_category_feat.fit_transform(X_train[category_features],y_train)

#X_train1=dataprep_pipe.fit_transform(X_train,y_train)

#############################
##Step 4 Pipeline creation for model
#############################
#Several classifier with Cross validation will be applied
y_test=misspellingCorrector(y_test)

label_enc=LabelEncoder()
y_test=label_enc.fit_transform(y_test)

#Init the clasfifier
sgd_clf=SGDClassifier()
logreg_clf=LogisticRegression()
linsvc_clf=LinearSVC()
svc_clf=SVC()
dectree_clf=DecisionTreeClassifier()
rndforest_clf=RandomForestClassifier()
knn_clf=KNeighborsClassifier()
mlp_clf= MLPClassifier(alpha=1, max_iter=1000)
ada_clf= AdaBoostClassifier()
nb_clf= GaussianNB()
disc_clf=QuadraticDiscriminantAnalysis()

#
print ('Creating the full Pipeline')

estimator=rndforest_clf
full_pipeline=Pipeline([('data_prep',dataprep_pipe),
                        ('model',rndforest_clf)])

full_pipeline.fit(X_train,y_train)

##Apply cross validation with the full_pipeline
cross_val_score(full_pipeline,X_train,y_train, cv=5, scoring='accuracy')


y_pred=full_pipeline.predict(X_test)

print ('Accuracy Score with',estimator,' estimator : ',accuracy_score(y_test, y_pred))
print('F1 Score with',estimator,' estimator : ',f1_score(y_test, y_pred, average='weighted'))
print('Precision Score with',estimator,' estimator : ',precision_score(y_test, y_pred, average='weighted'))
print('Recall Score with',estimator,' estimator : ',recall_score(y_test, y_pred, average='weighted'))
print('ROC_AUC score with',estimator,' estimator ', roc_auc_score(y_test, y_pred))

full_pipeline.get_params().keys()

scoring=['accuracy','f1','precision','recall']
#############################
##Step 5 GridSearchCV to find best params
#############################

##########v0 every possible option in the param grid- computationally unfeasible
param_grid_v0={'model': [SGDClassifier(),LogisticRegression(),LinearSVC(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()],
            'data_prep__numeric_pipe__data_missing__strategy':['median','mean','iterative','knn'],
            'data_prep__numeric_pipe__features_select':['passthrough',feature_select.Feature_Selector()],
            'data_prep__numeric_pipe__features_select__k_out_features': [1,2,3,4,5,6,7,8,9,10,11],
            'data_prep__numeric_pipe__features_select__rfe_estimator':['LogisticRegression','SVR'],
            'data_prep__numeric_pipe__features_select__strategy':['filter_num','filter_mutinf','wrapper_RFECV','wrapper_BackElim','LassoCV','RidgeCV'] ,
            'data_prep__category_pipe__data_missing__strategy': ['most_frequent','constant'],
            'data_prep__categroy_pipe__features_select':['passthrough',feature_select.Feature_Selector()],
            'data_prep__category_pipe__features_select__k_out_features': [1,2,3,4,5,6,7,8,9,10,11,12,13],
            'data_prep__category_pipe__features_select__rfe_estimator':['LogisticRegression','SVR'],
            'data_prep__category_pipe__features_select__strategy': ['filter_cat','filter_mutinf','wrapper_RFECV','wrapper_BackElim','LassoCV','RidgeCV'],
    }


clf0=GridSearchCV(full_pipeline,param_grid_v0, cv=5)
clf0.fit(X_train,y_train)

#Grid_Search to see the optimal amount of k_out_features in filter methods of features select
##########v1 param grid- computationally unfeasible because too many option with k_out_features
param_grid_v1={'model': [SGDClassifier(),LogisticRegression(),LinearSVC(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()],
            'data_prep__numeric_pipe__data_missing__strategy':['median','mean','iterative','knn'],
            #'data_prep__numeric_pipe__features_select':['passthrough',feature_select.Feature_Selector()],
            'data_prep__numeric_pipe__features_select__k_out_features': [2,3,4,5,6,7,8,9,10,11],
            #'data_prep__numeric_pipe__features_select__rfe_estimator':['LogisticRegression','SVR'],
            'data_prep__numeric_pipe__features_select__strategy':['filter_num','filter_mutinf'] ,
            #'data_prep__category_pipe__data_missing__strategy': ['most_frequent'],
            #'data_prep__categroy_pipe__features_select':['passthrough',feature_select.Feature_Selector()],
            'data_prep__category_pipe__features_select__k_out_features': [2,3,4,5,6,7,8,9,10,11,12,13],
            #'data_prep__category_pipe__features_select__rfe_estimator':['LogisticRegression','SVR'],
            'data_prep__category_pipe__features_select__strategy': ['filter_cat','filter_mutinf'],
    }

clf_v1=GridSearchCV(full_pipeline,param_grid_v1, cv=5,n_jobs=-1)
clf_v1.fit(X_train,y_train)

clf_v1.best_params_
clf_v1.best_score_

#We modify the dataprep pipeline to withdraw feature select steps and run Grid Search to see if best score persist
##############v2
param_grid_v2={'model': [SGDClassifier(),LogisticRegression(),LinearSVC(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()],
            'data_prep__numeric_pipe__data_missing__strategy':['median','mean','iterative','knn'],            
            'data_prep__category_pipe__data_missing__strategy': ['most_frequent','constant'],
            'data_prep__numeric_pipe__features_select':['passthrough'],
            'data_prep__category_pipe__features_select':['passthrough'],            
    }

clf_v2=GridSearchCV(full_pipeline,param_grid_v2, cv=5,n_jobs=-1)
clf_v2.fit(X_train,y_train)
clf_v2.best_params_
clf_v2.best_score_

#########v31
#Multi paragrid to test i) best classifier plus data missing strategy, 

param_grid_v31=[{'model': [SGDClassifier(),LogisticRegression(),LinearSVC(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()],
            'data_prep__numeric_pipe__data_missing__strategy':['median','mean','iterative','knn'],
            'data_prep__numeric_pipe__features_select':['passthrough'],
            'data_prep__category_pipe__data_missing__strategy': ['most_frequent','constant'],
            'data_prep__category_pipe__features_select':['passthrough'],
                }]
#load model to save time of fitting
clf_v31= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v31.pkl')

clf_v31=GridSearchCV(full_pipeline,param_grid_v31,scoring=scoring,refit='accuracy',cv=5,n_jobs=-1)
#clf_v31=GridSearchCV(full_pipeline,param_grid_v31,cv=5,n_jobs=-1)
clf_v31.fit(X_train,y_train)
clf_v31.best_estimator_
print('Params of best estimator of clf_v31:', clf_v31.best_params_)
#Results:Params of best estimator of clf_v31: {'data_prep__category_pipe__data_missing__strategy': 'most_frequent',
#'data_prep__category_pipe__features_select': 'passthrough', 'data_prep__numeric_pipe__data_missing__strategy': 'mean', 
#'data_prep__numeric_pipe__features_select': 'passthrough', 'model': RandomForestClassifier()}

print('Score of best estimator of clf_v31:', clf_v31.best_score_)
#Score of best estimator of clf_v31: 0.9964285714285716

print('Index of best estimator of clf_v31:', clf_v31.best_index_)
#Index of best estimator of clf_v31: 1

df_results_v31=pd.DataFrame(clf_v31.cv_results_)
path_df_results_v31=r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v31.csv'
df_results_v31.to_csv(path_df_results_v31,index=False)
df_results_v31.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v31.xlsx',index=False)
clf_v31.refit

preds = clf_v31.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.975

y_pred_31=clf_v31.predict(X_test)
estimator='clf_v31'

print ('Accuracy Score with',estimator,' estimator : ',accuracy_score(y_test, y_pred_31))
print('F1 Score with',estimator,' estimator : ',f1_score(y_test, y_pred_31, average='weighted'))
print('Precision Score with',estimator,' estimator : ',precision_score(y_test, y_pred_31, average='weighted'))
print('Recall Score with',estimator,' estimator : ',recall_score(y_test, y_pred_31, average='weighted'))
print('ROC_AUC score with',estimator,' estimator ', roc_auc_score(y_test, y_pred_31))

#Saving the model
joblib.dump(clf_v31, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v31.pkl', compress=1)

#Explore feature_importances
importances=clf_v31.best_estimator_.named_steps['model'].feature_importances_
indices = np.argsort(importances)[::-1]
X_train_featimp=pd.concat([X_train[numerical_features],X_train[category_features]],axis=1)
# Print the feature ranking
print("Feature ranking of clf_v31:")

for f in range(X_train_featimp.shape[1]):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], X_train_featimp.columns.values[indices[f]] ,importances[indices[f]]))

#########v32
#ii)best feature select strategy, 
param_grid_v32=[{'model': [SGDClassifier(),LogisticRegression(),LinearSVC(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()],                        
            'data_prep__numeric_pipe__features_select__strategy':['filter_num','filter_mutinf','wrapper_RFECV','wrapper_BackElim','LassoCV','RidgeCV'] ,            
            'data_prep__category_pipe__features_select__strategy': ['filter_cat','filter_mutinf','wrapper_RFECV','wrapper_BackElim','LassoCV','RidgeCV']
            }]
#load model to save time of fitting
clf_v32= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v32.pkl')


clf_v32=GridSearchCV(full_pipeline,param_grid_v32,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)

clf_v32.fit(X_train,y_train)
clf_v32.best_estimator_
print('Params of best estimator of clf_v32:', clf_v32.best_params_)
# Params of best estimator of clf_v32: {'data_prep__category_pipe__features_select__strategy': 'filter_cat', 
#                                       'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFECV', 
#                                       'model': RandomForestClassifier()}

print('Score of best estimator of clf_v32:', clf_v32.best_score_)
#Score of best estimator of clf_v32: 0.9928571428571429

print('Index of best estimator of clf_v31:', clf_v32.best_index_)
#Index of best estimator of clf_v32: 17

df_results_v32=pd.DataFrame(clf_v32.cv_results_)
df_results_v32.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v32.csv',index=False)
df_results_v32.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v32.xlsx',index=False)
clf_v32.refit
preds = clf_v32.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.975

y_pred_32=clf_v32.predict(X_test)

#Saving the model
joblib.dump(clf_v32, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v32.pkl', compress=1)

#########v33
#iii)best number of our features in filter methods in feature_select we choose RFE whitout CV for avoiding CV_nesting
param_grid_v33=[{'model': [SGDClassifier(),LogisticRegression(),LinearSVC(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()],
            'data_prep__numeric_pipe__features_select__k_out_features': [1,2,3,4,5,6,7,8,9,10,11],
            'data_prep__numeric_pipe__features_select__strategy':['filter_num','filter_mutinf','wrapper_RFE'] ,
            'data_prep__category_pipe__features_select__k_out_features': [1,2,3,4,5,6,7,8,9,10,11,12,13],
            'data_prep__category_pipe__features_select__strategy': ['filter_cat','filter_mutinf','wrapper_RFE']
            }]
#load model to save time of fitting
clf_v33= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v33.pkl')

clf_v33=GridSearchCV(full_pipeline,param_grid_v33,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)
clf_v33.fit(X_train,y_train)
clf_v33.best_estimator_
print('Params of best estimator of clf_v33:', clf_v33.best_params_)
#Results:Params of best estimator of clf_v33: {'data_prep__category_pipe__features_select__k_out_features': 7,
 # 'data_prep__category_pipe__features_select__strategy': 'filter_cat', 
 # 'data_prep__numeric_pipe__features_select__k_out_features': 1, 
 # 'data_prep__numeric_pipe__features_select__strategy': 'filter_num', 'model': RandomForestClassifier()}

print('Score of best estimator of clf_v33:', clf_v33.best_score_)
#Score of best estimator of clf_v33: 1,0

print('Index of best estimator of clf_v33:', clf_v33.best_index_)
#Index of best estimator of clf_v33: 3569

df_results_v33=pd.DataFrame(clf_v33.cv_results_)
df_results_v33.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v33.csv',index=False)
df_results_v33.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v33.xlsx',index=False)
clf_v33.refit
preds = clf_v33.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.9833333333333333
y_pred_33=clf_v33.predict(X_test)

#Saving the model
joblib.dump(clf_v33, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v33.pkl', compress=1)

#########v4
#iv) overall search with data missing strategy, features select strategy, and number of features in features_select
param_grid_v4={'model': [SGDClassifier(),LogisticRegression(),LinearSVC(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()],
            'data_prep__numeric_pipe__data_missing__strategy':['median','mean','iterative','knn'],
            #'data_prep__numeric_pipe__features_select':['passthrough',feature_select.Feature_Selector()],
            'data_prep__numeric_pipe__features_select__k_out_features': [1,2,3,4,5,6,7],
            #'data_prep__numeric_pipe__features_select__rfe_estimator':['LogisticRegression','SVR'],
            'data_prep__numeric_pipe__features_select__strategy':['filter_num','filter_mutinf','wrapper_RFECV'] ,
            #'data_prep__category_pipe__data_missing__strategy': ['most_frequent'],
            #'data_prep__categroy_pipe__features_select':['passthrough',feature_select.Feature_Selector()]
            'data_prep__category_pipe__features_select__k_out_features': [1,2,3,4,5,6,7],
            #'data_prep__category_pipe__features_select__rfe_estimator':['LogisticRegression','SVR'],
            'data_prep__category_pipe__features_select__strategy': ['filter_cat','filter_mutinf','wrapper_RFECV']
    }
#load model to save time of fitting
clf_v4= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v4.pkl')

clf_v4=GridSearchCV(full_pipeline,param_grid_v4,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)
clf_v4.fit(X_train,y_train)
clf_v4.best_estimator_
print('Params of best estimator of clf_v4:', clf_v4.best_params_)
#Results:Params of best estimator of clf_v4: {'data_prep__category_pipe__features_select__k_out_features': 5, 
# 'data_prep__category_pipe__features_select__strategy': 'wrapper_RFECV', 
# 'data_prep__numeric_pipe__data_missing__strategy': 'mean', 
# 'data_prep__numeric_pipe__features_select__k_out_features': 1, 
# 'data_prep__numeric_pipe__features_select__strategy': 'filter_mutinf', 
# 'model': RandomForestClassifier()}

print('Score of best estimator of clf_v4:', clf_v4.best_score_)
#Score of best estimator of clf_v4: 1.0

print('Index of best estimator of clf_v4:', clf_v4.best_index_)
#Index of best estimator of clf_v4: 7193

df_results_v4=pd.DataFrame(clf_v4.cv_results_)
df_results_v4.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v4.csv',index=False)
df_results_v4.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v4.xlsx',index=False)
clf_v4.refit
preds = clf_v4.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test) #0.9666666666666667

y_pred_4=clf_v4.predict(X_test)

#Saving the model
joblib.dump(clf_v4, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v4.pkl', compress=1)

#########v5
#v) best estimator with default parameters but feature select.
param_grid_v5=[{'model': [SGDClassifier(),LogisticRegression(),LinearSVC(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()],            
            'data_prep__numeric_pipe__features_select':['passthrough'],            
            'data_prep__category_pipe__features_select':['passthrough']
                }]
#load model to save time of fitting
clf_v5= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v5.pkl')

clf_v5=GridSearchCV(full_pipeline,param_grid_v5,scoring=scoring,refit='accuracy',cv=5,n_jobs=-1)
#clf_v31=GridSearchCV(full_pipeline,param_grid_v31,cv=5,n_jobs=-1)
clf_v5.fit(X_train,y_train)
clf_v5.best_estimator_
print('Params of best estimator of clf_v31:', clf_v5.best_params_)
#Results:Params of best estimator of clf_v5:{'data_prep__category_pipe__features_select': 'passthrough', 
# 'data_prep__numeric_pipe__features_select': 'passthrough', 
# 'model': RandomForestClassifier()}

print('Score of best estimator of clf_v31:', clf_v5.best_score_)
#Score of best estimator of clf_v5: 0.9928571428571429

print('Index of best estimator of clf_v31:', clf_v5.best_index_)
#Index of best estimator of clf_v31: 5

df_results_v5=pd.DataFrame(clf_v5.cv_results_)

df_results_v5.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v5.csv',index=False)
df_results_v5.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v5.xlsx',index=False)

clf_v5.refit
preds = clf_v5.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.975

y_pred_5=clf_v5.predict(X_test)

#Saving the model

joblib.dump(clf_v5, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v5.pkl', compress=1)

#RandomForest is the best estimator in clf_v5 so we can see feature_importances since no feature_selection has been performed.
importances=clf_v5.best_estimator_.named_steps['model'].feature_importances_
indices = np.argsort(importances)[::-1]
X_train_featimp=pd.concat([X_train[numerical_features],X_train[category_features]],axis=1)
# Print the feature ranking
print("Feature ranking:")

for f in range(X_train_featimp.shape[1]):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], X_train_featimp.columns.values[indices[f]] ,importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train_featimp.shape[1]), importances[indices],
        color="r")
plt.xticks(range(X_train_featimp.shape[1]), X_train_featimp.columns.values[indices])
plt.xlim([-1, X_train_featimp.shape[1]])
plt.show()


#########v6
#the paramgrid clf_v33 will be repeated with the new classifiers
param_grid_v6=[{'model': [knn_clf,mlp_clf,ada_clf,nb_clf,disc_clf],
            'data_prep__numeric_pipe__features_select__k_out_features': [1,2,3,4,5,6,7,8,9,10,11],
            'data_prep__numeric_pipe__features_select__strategy':['filter_num','filter_mutinf','wrapper_RFE'] ,
            'data_prep__category_pipe__features_select__k_out_features': [1,2,3,4,5,6,7,8,9,10,11,12,13],
            'data_prep__category_pipe__features_select__strategy': ['filter_cat','filter_mutinf','wrapper_RFE']
            }]
#load model to save time of fitting
clf_v6= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v6.pkl')

clf_v6=GridSearchCV(full_pipeline,param_grid_v6,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)
clf_v6.fit(X_train,y_train)
clf_v6.best_estimator_
print('Params of best estimator of clf_v6:', clf_v6.best_params_)
#Results:Params of best estimator of clf_v6: {'data_prep__category_pipe__features_select__k_out_features':5 ,
 # 'data_prep__category_pipe__features_select__strategy': 'filter_cat', 
 # 'data_prep__numeric_pipe__features_select__k_out_features':3 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE', 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v6:', clf_v6.best_score_)
#Score of best estimator of clf_v6: 1.0

print('Index of best estimator of clf_v6:', clf_v6.best_index_)
#Index of best estimator of clf_v6: 2022

df_results_v6=pd.DataFrame(clf_v6.cv_results_)
df_results_v6.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v6.csv',index=False)
df_results_v6.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v6.xlsx',index=False)
clf_v6.refit
preds = clf_v6.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.9666667
y_pred_6=clf_v6.predict(X_test)

#Saving the model
joblib.dump(clf_v6, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v6.pkl', compress=1)



#########v7
#the paramgrid clf_v4 will be repeated with the new classifiers
param_grid_v7={'model': [knn_clf,mlp_clf,ada_clf,nb_clf,disc_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median','mean','iterative','knn'],
            #'data_prep__numeric_pipe__features_select':['passthrough',feature_select.Feature_Selector()],
            'data_prep__numeric_pipe__features_select__k_out_features': [1,2,3,4,5,6,7],
            #'data_prep__numeric_pipe__features_select__rfe_estimator':['LogisticRegression','SVR'],
            'data_prep__numeric_pipe__features_select__strategy':['filter_num','filter_mutinf','wrapper_RFECV'] ,
            #'data_prep__category_pipe__data_missing__strategy': ['most_frequent'],
            #'data_prep__categroy_pipe__features_select':['passthrough',feature_select.Feature_Selector()]
            'data_prep__category_pipe__features_select__k_out_features': [1,2,3,4,5,6,7],
            #'data_prep__category_pipe__features_select__rfe_estimator':['LogisticRegression','SVR'],
            'data_prep__category_pipe__features_select__strategy': ['filter_cat','filter_mutinf','wrapper_RFECV']
    }


#load model to save time of fitting
clf_v7= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v7.pkl')

# clf_v7=GridSearchCV(full_pipeline,param_grid_v7,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)
clf_v7=GridSearchCV(full_pipeline,param_grid_v7,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)
clf_v7.fit(X_train,y_train)
clf_v7.best_estimator_
print('Params of best estimator of clf_v7:', clf_v7.best_params_)
#Results:Params of best estimator of clf_v7: {'data_prep__category_pipe__features_select__k_out_features': 7,
 # 'data_prep__category_pipe__features_select__strategy': 'filter_cat', 
 # 'data_prep__numeric_pipe__data_missing__strategy': 'median',
 # 'data_prep__numeric_pipe__features_select__k_out_features':7 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'filter_num', 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v7:', clf_v7.best_score_)
#Score of best estimator of clf_v7: 1.0

print('Index of best estimator of clf_v7:', clf_v7.best_index_)
#Index of best estimator of clf_v7: 7652

df_results_v7=pd.DataFrame(clf_v7.cv_results_)
clf_v7.refit
preds = clf_v7.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.9916666666666667
y_pred_8=clf_v8.predict(X_test)

#Saving the model
joblib.dump(clf_v7, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v7.pkl', compress=1)

###########v8 The best classifier are RandomForest and Adaboost, lets define a gridsearch without feature select and only rndforest
param_grid_v8={'model': [rndforest_clf,ada_clf],
               'data_prep__numeric_pipe__features_select':['passthrough'],            
               'data_prep__category_pipe__features_select':['passthrough']
            }
clf_v8=GridSearchCV(full_pipeline,param_grid_v8,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)
clf_v8.fit(X_train,y_train)
clf_v8.best_estimator_
print('Params of best estimator of clf_v8:', clf_v8.best_params_)
#Results:Params of best estimator of clf_v8: {8 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v8:', clf_v8.best_score_)
#Score of best estimator of clf_v8: 0.9964285714285716

print('Index of best estimator of clf_v8:', clf_v8.best_index_)
#Index of best estimator of clf_v8: 1

df_results_v8=pd.DataFrame(clf_v8.cv_results_)
df_results_v8.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v8.csv',index=False)
df_results_v8.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v8.xlsx',index=False)
clf_v8.refit
preds = clf_v8.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0.9916666666666667
y_pred_8=clf_v8.predict(X_test)

###########v9 The best classifier are RandomForest and Adaboost, lets define a Grid search with these two classifier and feature select and data missing parameters
param_grid_v9={'model': [rndforest_clf,ada_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median','mean','iterative','knn'],            
            'data_prep__numeric_pipe__features_select__k_out_features': [1,2,3,4,5,6,7,8,9,10],            
            'data_prep__numeric_pipe__features_select__strategy':['filter_num','filter_mutinf','wrapper_RFE'] ,        
            'data_prep__category_pipe__features_select__k_out_features': [1,2,3,4,5,6,7,8,9,10],           
            'data_prep__category_pipe__features_select__strategy': ['filter_cat','filter_mutinf','wrapper_RFE']
    }

#load model to save time of fitting
clf_v9= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v9.pkl')

clf_v9=GridSearchCV(full_pipeline,param_grid_v9,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)
clf_v9.fit(X_train,y_train)
clf_v9.best_estimator_
print('Params of best estimator of clf_v9:', clf_v9.best_params_)
#Results:Params of best estimator of clf_v9: {'data_prep__category_pipe__features_select__k_out_features': 5,
 # 'data_prep__category_pipe__features_select__strategy': 'wrapper_RFE', 
 # 'data_prep__numeric_pipe__data_missing__strategy': 'median',
 # 'data_prep__numeric_pipe__features_select__k_out_features':4 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE', 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v9:', clf_v9.best_score_)
#Score of best estimator of clf_v9: 1.0

print('Index of best estimator of clf_v9:', clf_v9.best_index_)
#Index of best estimator of clf_v9: 3383

df_results_v9=pd.DataFrame(clf_v9.cv_results_)
df_results_v9.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v9.csv',index=False)
df_results_v9.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v9.xlsx',index=False)
clf_v9.refit
preds = clf_v9.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0,983333333

y_pred_9=clf_v9.predict(X_test)

#Saving the model
joblib.dump(clf_v9, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v9.pkl', compress=1)


######v10 to test the best option found in the previous GridSearch
param_grid_v10={'model': [rndforest_clf,ada_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [1,4,7],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num','wrapper_RFE'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [5,7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat','wrapper_RFE']
     }



#load model to save time of fitting
clf_v10= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v10.pkl')

clf_v10=GridSearchCV(full_pipeline,param_grid_v10,scoring=scoring,refit='accuracy', cv=5,n_jobs=-1)
clf_v10.fit(X_train,y_train)
clf_v10.best_estimator_
print('Params of best estimator of clf_v10:', clf_v10.best_params_)
#Results:Params of best estimator of clf_v10: {'data_prep__category_pipe__features_select__k_out_features': 5,
 # 'data_prep__category_pipe__features_select__strategy': 'wrapper_RFE', 
 # 'data_prep__numeric_pipe__data_missing__strategy': 'median',
 # 'data_prep__numeric_pipe__features_select__k_out_features':4 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE', 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v10:', clf_v10.best_score_)
#Score of best estimator of clf_v10: 1.0

print('Index of best estimator of clf_v10:', clf_v10.best_index_)
#Index of best estimator of clf_v10: 19

df_results_v10=pd.DataFrame(clf_v10.cv_results_)
df_results_v10.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v10.csv',index=False)
df_results_v10.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v10.xlsx',index=False)
clf_v10.refit
preds = clf_v10.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0,983333333
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)

y_pred_10=clf_v10.predict(X_test)

#Saving the model
joblib.dump(clf_v10, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v10.pkl', compress=1)

######v11 to test the best option found in the previous GridSearch with all classifiers


param_grid_v11={'model': [logreg_clf,svc_clf,dectree_clf, rndforest_clf, knn_clf, mlp_clf,ada_clf, nb_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [4,7],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num','wrapper_RFE'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [5,7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat','wrapper_RFE']
     }
from sklearn.metrics import make_scorer
scoring2 = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0),
    'precision':make_scorer(precision_score),
    'f1':make_scorer(f1_score),
    'roc_auc':make_scorer(roc_auc_score)
}


#load model to save time of fitting
clf_v11= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v11.pkl')

clf_v11=GridSearchCV(full_pipeline,param_grid_v11,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v11.fit(X_train,y_train)
clf_v11.best_estimator_
print('Params of best estimator of clf_v10:', clf_v11.best_params_)
#Results:Params of best estimator of clf_v10: {'data_prep__category_pipe__features_select__k_out_features': 5,
 # 'data_prep__category_pipe__features_select__strategy': 'wrapper_RFE', 
 # 'data_prep__numeric_pipe__data_missing__strategy': 'median',
 # 'data_prep__numeric_pipe__features_select__k_out_features':4 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE', 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v11:', clf_v11.best_score_)
#Score of best estimator of clf_v11: 1.0

print('Index of best estimator of clf_v11:', clf_v11.best_index_)
#Index of best estimator of clf_v11: 19

df_results_v11=pd.DataFrame(clf_v11.cv_results_)
df_results_v11.to_csv(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v11.csv',index=False)
df_results_v11.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v11.xlsx',index=False)
clf_v11.refit
preds = clf_v11.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0,983333333
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)

y_pred_11=clf_v11.predict(X_test)

#Saving the model
joblib.dump(clf_v11, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\clf_v11.pkl', compress=1)



param_grid_v12={'model': [ada_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [7],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [5],           
             'data_prep__category_pipe__features_select__strategy': ['wrapper_RFE']
     }

clf_v12=GridSearchCV(full_pipeline,param_grid_v12,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v12.fit(X_train,y_train)
clf_v12.best_estimator_
print('Params of best estimator of clf_v10:', clf_v12.best_params_)
#Results:Params of best estimator of clf_v10: {'data_prep__category_pipe__features_select__k_out_features': 7,
 # 'data_prep__category_pipe__features_select__strategy': 'wrapper_RFE', 
 # 'data_prep__numeric_pipe__data_missing__strategy': 'median',
 # 'data_prep__numeric_pipe__features_select__k_out_features':4 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE', 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v11:', clf_v12.best_score_)
#Score of best estimator of clf_v11: 1

print('Index of best estimator of clf_v11:', clf_v12.best_index_)
#Index of best estimator of clf_v11: 19

clf_v12.refit
preds = clf_v12.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0,983333333
y_pred_12 = clf_v12.predict(X_test)
df_results_v12=pd.DataFrame(clf_v12.cv_results_)
df_results_v12.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_results_v12.xlsx',index=False)


param_grid_v13={'model': [ada_clf],
            'data_prep__numeric_pipe__data_missing__strategy':['median'],            
             'data_prep__numeric_pipe__features_select__k_out_features': [7],            
             'data_prep__numeric_pipe__features_select__strategy':['filter_num'] ,        
             'data_prep__category_pipe__features_select__k_out_features': [7],           
             'data_prep__category_pipe__features_select__strategy': ['filter_cat']
     }

clf_v13=GridSearchCV(full_pipeline,param_grid_v13,scoring=scoring2,refit='accuracy', cv=5,n_jobs=-1)
clf_v13.fit(X_train,y_train)
clf_v13.best_estimator_
print('Params of best estimator of clf_v10:', clf_v13.best_params_)
#Results:Params of best estimator of clf_v10: {'data_prep__category_pipe__features_select__k_out_features': 5,
 # 'data_prep__category_pipe__features_select__strategy': 'wrapper_RFE', 
 # 'data_prep__numeric_pipe__data_missing__strategy': 'median',
 # 'data_prep__numeric_pipe__features_select__k_out_features':4 , 
 # 'data_prep__numeric_pipe__features_select__strategy': 'wrapper_RFE', 'model':AdaBoostClassifier()}

print('Score of best estimator of clf_v11:', clf_v13.best_score_)
#Score of best estimator of clf_v11: 1.0

print('Index of best estimator of clf_v11:', clf_v12.best_index_)
#Index of best estimator of clf_v11: 19

clf_v13.refit
preds = clf_v13.predict(X_test)
#probs = clf_v31.predict_proba(X_test)
np.mean(preds == y_test)#0,983333333
y_pred_13 = clf_v13.predict(X_test)
df_results_v13=pd.DataFrame(clf_v13.cv_results_)


#########################
#Lets build a dataframe with the results of all best_stimator in the different grid search and with the prediction results in X_tests of all best_estimators in the different grid search
# #######################
# overall_results={'clf':['clf_v31','clf_v32','clf_v33','clf_v4','clf_v5','clf_v6','clf_v7','clf_v8','clf_v9','clf_v12'],
#                  'params':[clf_v31.best_params_,clf_v32.best_params_,clf_v33.best_params_, clf_v4.best_params_,clf_v5.best_params_,
#                            clf_v6.best_params_,clf_v7.best_params_,clf_v8.best_params_,clf_v9.best_params_,clf_v12.best_params_],
#                  'accuracy_crossVal':[df_results_v31.loc[clf_v31.best_index_,'mean_test_accuracy'],df_results_v32.loc[clf_v32.best_index_,'mean_test_accuracy'],df_results_v33.loc[clf_v33.best_index_,'mean_test_accuracy'],df_results_v4.loc[clf_v4.best_index_,'mean_test_accuracy'],
#                                       df_results_v5.loc[clf_v5.best_index_,'mean_test_accuracy'],df_results_v6.loc[clf_v6.best_index_,'mean_test_accuracy'],df_results_v7.loc[clf_v7.best_index_,'mean_test_accuracy'],df_results_v8.loc[clf_v8.best_index_,'mean_test_accuracy'],df_results_v9.loc[clf_v9.best_index_,'mean_test_accuracy'],df_results_v12.loc[clf_v12.best_index_,'mean_test_accuracy']],
#                  'f1_crossVal':[df_results_v31.loc[clf_v31.best_index_,'mean_test_f1'],df_results_v32.loc[clf_v32.best_index_,'mean_test_f1'],df_results_v33.loc[clf_v33.best_index_,'mean_test_f1'],df_results_v4.loc[clf_v4.best_index_,'mean_test_f1'],df_results_v5.loc[clf_v5.best_index_,'mean_test_f1'],
#                                 df_results_v6.loc[clf_v6.best_index_,'mean_test_f1'],df_results_v7.loc[clf_v7.best_index_,'mean_test_f1'],df_results_v8.loc[clf_v8.best_index_,'mean_test_f1'],df_results_v9.loc[clf_v9.best_index_,'mean_test_f1']df_results_v12.loc[clf_v12.best_index_,'mean_test_f1']],
#                  'precision_crossVal':[df_results_v31.loc[clf_v31.best_index_,'mean_test_precision'],df_results_v32.loc[clf_v32.best_index_,'mean_test_precision'],df_results_v33.loc[clf_v33.best_index_,'mean_test_precision'],df_results_v4.loc[clf_v4.best_index_,'mean_test_precision'],
#                                        df_results_v5.loc[clf_v5.best_index_,'mean_test_precision'],df_results_v6.loc[clf_v6.best_index_,'mean_test_precision'],df_results_v7.loc[clf_v7.best_index_,'mean_test_precision'],df_results_v8.loc[clf_v8.best_index_,'mean_test_precision'],df_results_v9.loc[clf_v9.best_index_,'mean_test_precision'],df_results_v12.loc[clf_v12.best_index_,'mean_test_precision']],
#                  'recall_crossVal':[df_results_v31.loc[clf_v31.best_index_,'mean_test_recall'],df_results_v32.loc[clf_v32.best_index_,'mean_test_recall'],df_results_v33.loc[clf_v33.best_index_,'mean_test_recall'],df_results_v4.loc[clf_v4.best_index_,'mean_test_recall'],
#                                     df_results_v5.loc[clf_v5.best_index_,'mean_test_recall'],df_results_v6.loc[clf_v6.best_index_,'mean_test_recall'],df_results_v7.loc[clf_v7.best_index_,'mean_test_recall'],df_results_v8.loc[clf_v8.best_index_,'mean_test_recall'],df_results_v9.loc[clf_v9.best_index_,'mean_test_recall'],df_results_v12.loc[clf_v12.best_index_,'mean_test_recall']],
#                  'accuracy_test':[accuracy_score(y_test, y_pred_31),accuracy_score(y_test, y_pred_32),accuracy_score(y_test, y_pred_33),accuracy_score(y_test, y_pred_4),accuracy_score(y_test, y_pred_5),
#                                   accuracy_score(y_test, y_pred_6),accuracy_score(y_test, y_pred_7),accuracy_score(y_test, y_pred_8),accuracy_score(y_test, y_pred_9),accuracy_score(y_test, y_pred_12)],
#                  'f1_test':[f1_score(y_test, y_pred_31, average='weighted'),f1_score(y_test, y_pred_32, average='weighted'),f1_score(y_test, y_pred_33, average='weighted'),f1_score(y_test, y_pred_4, average='weighted'),
#                             f1_score(y_test, y_pred_5, average='weighted'),f1_score(y_test, y_pred_6, average='weighted'),f1_score(y_test, y_pred_7, average='weighted'),f1_score(y_test, y_pred_8, average='weighted'),f1_score(y_test, y_pred_8, average='weighted')],
#                  'precision_test':[precision_score(y_test, y_pred_31, average='weighted'), precision_score(y_test, y_pred_32, average='weighted'), precision_score(y_test, y_pred_33, average='weighted'), precision_score(y_test, y_pred_4, average='weighted'),
#                                    precision_score(y_test, y_pred_5, average='weighted'), precision_score(y_test, y_pred_6, average='weighted'), precision_score(y_test, y_pred_7, average='weighted'),precision_score(y_test, y_pred_8, average='weighted'),precision_score(y_test, y_pred_9, average='weighted')],
#                  'recall_test':[recall_score(y_test, y_pred_31, average='weighted'), recall_score(y_test, y_pred_32, average='weighted'), recall_score(y_test, y_pred_33, average='weighted'), recall_score(y_test, y_pred_4, average='weighted'), recall_score(y_test, y_pred_5, average='weighted'),
#                                 recall_score(y_test, y_pred_6, average='weighted'), recall_score(y_test, y_pred_7, average='weighted'),recall_score(y_test, y_pred_8, average='weighted'),recall_score(y_test, y_pred_9, average='weighted')],
#                  'specificity':[recall_score(y_test, y_pred_31, average='weighted',pos_label=0), recall_score(y_test, y_pred_32, average='weighted',pos_label=0), recall_score(y_test, y_pred_33, average='weighted',pos_label=0), recall_score(y_test, y_pred_4, average='weighted',pos_label=0), recall_score(y_test, y_pred_5, average='weighted',pos_label=0),
#                                 recall_score(y_test, y_pred_6, average='weighted',pos_label=0), recall_score(y_test, y_pred_7, average='weighted',pos_label=0),recall_score(y_test, y_pred_8, average='weighted',pos_label=0),recall_score(y_test, y_pred_9, average='weighted',pos_label=0)],
#                  'roc_auc_test':[roc_auc_score(y_test, y_pred_31),roc_auc_score(y_test, y_pred_32),roc_auc_score(y_test, y_pred_33),roc_auc_score(y_test, y_pred_4),roc_auc_score(y_test, y_pred_5),
#                                  roc_auc_score(y_test, y_pred_6),roc_auc_score(y_test, y_pred_7),roc_auc_score(y_test, y_pred_8),roc_auc_score(y_test, y_pred_9)]    
#     }


overall_results={'clf':['clf_v9','clf_v12','clf_v13'],
                 'params':[clf_v9.best_params_,clf_v12.best_params_,clf_v13.best_params_],
                 # 'accuracy_crossVal':[df_results_v9.loc[clf_v9.best_index_,'mean_test_accuracy'],df_results_v12.loc[clf_v12.best_index_,'mean_test_accuracy'],df_results_v13.loc[clf_v13.best_index_,'mean_test_accuracy']],
                 # 'f1_crossVal':[df_results_v9.loc[clf_v9.best_index_,'mean_test_f1'],df_results_v12.loc[clf_v12.best_index_,'mean_test_f1'],df_results_v13.loc[clf_v13.best_index_,'mean_test_f1']],
                 # 'precision_crossVal':[df_results_v9.loc[clf_v9.best_index_,'mean_test_precision'],df_results_v12.loc[clf_v12.best_index_,'mean_test_precision'],df_results_v13.loc[clf_v13.best_index_,'mean_test_precision']],
                 # 'recall_crossVal':[df_results_v9.loc[clf_v9.best_index_,'mean_test_recall'],df_results_v12.loc[clf_v12.best_index_,'mean_test_sensivitiy'],df_results_v13.loc[clf_v13.best_index_,'mean_test_sensivitiy']],
                 'accuracy_test':[accuracy_score(y_test, y_pred_9),accuracy_score(y_test, y_pred_12),accuracy_score(y_test, y_pred_13)],
                 'f1_test':[f1_score(y_test, y_pred_9, average='weighted'),f1_score(y_test, y_pred_12, average='weighted'),f1_score(y_test, y_pred_13, average='weighted')],
                 'precision_test':[precision_score(y_test, y_pred_9, average='weighted'),precision_score(y_test, y_pred_12, average='weighted'),precision_score(y_test, y_pred_13, average='weighted')],
                 'recall_test':[recall_score(y_test, y_pred_9, average='weighted'),recall_score(y_test, y_pred_12, average='weighted'),recall_score(y_test, y_pred_13, average='weighted')],
                 'specificity_test':[recall_score(y_test, y_pred_9, average='weighted',pos_label=0),recall_score(y_test, y_pred_12, average='weighted',pos_label=0),recall_score(y_test, y_pred_13, average='weighted',pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_9),roc_auc_score(y_test, y_pred_12),roc_auc_score(y_test, y_pred_13)]    
    }

# df_overall_results=pd.DataFrame(data=overall_results)
# df_overall_results.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_overall_results.xlsx',index=False)

df_overall_results_paper=pd.DataFrame(data=overall_results)
df_overall_results_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results\df_overall_results_paper.xlsx',index=False)

#Lets calculate the confusion matrix of the best 3 results of GridSearch
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_9)

confusion_matrix(y_test, y_pred_12)

confusion_matrix(y_test, y_pred_13)
#if clf selected is Random Forest we see feature importances parameter
# importances=clf.best_estimator_.named_steps['model'].feature_importances_
# indices = np.argsort(importances)[::-1]
# X_train1=pd.concat([X_train[numerical_features],X_train[category_features]],axis=1)
# # Print the feature ranking
# print("Feature ranking:")

# for f in range(X_train1.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# # Plot the impurity-based feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X_train1.shape[1]), importances[indices],
#         color="r")
# plt.xticks(range(X_train1.shape[1]), indices)
# plt.xlim([-1, X_train1.shape[1]])
# plt.show()

# index=[7,11,4,8,12]
# X_train.columns[index]

###########Features selected

pipe_numeric_featsel= Pipeline([('mispelling',adhoc_transf.misspellingTransformer()),
                                 ('features_cast',adhoc_transf.Numeric_Cast_Column()),
                                 ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median'))
                                 
                        ])

pipe_category_feat= Pipeline([('mispelling',adhoc_transf.misspellingTransformer()),
                                 ('features_cast',adhoc_transf.Category_Cast_Column()),
                                 ('data_missing',missing_val_imput.Category_Imputer(strategy='most_frequent')),
                                 ('cat_feat_engineering',adhoc_transf.CastDown()),
                                 ('encoding', OrdinalEncoder())
                        ])

X_train_numfeatsel=pipe_numeric_featsel.fit_transform(X_train[numerical_features])
df_X_train_numfeatsel=pd.DataFrame(X_train_numfeatsel, columns=numerical_features)

X_train_catfeatsel=pipe_category_feat.fit_transform(X_train[category_features])
df_X_train_catfeatsel=pd.DataFrame(X_train_catfeatsel, columns=category_features)
###clf v9: AdaBoost, median, numRFE-4,nomRFE-5 
feature_select.feat_sel_RFE(df_X_train_numfeatsel,y_train,k_out_features=4)

feature_select.feat_sel_RFE(df_X_train_catfeatsel,y_train,k_out_features=5)

###clf v12: AdaBoost, median, numANOVA-7,nomRFE-5 

feature_select.feat_sel_Num_to_Cat(df_X_train_numfeatsel,y_train,k_out_features=7)

feature_select.feat_sel_RFE(df_X_train_catfeatsel,y_train,k_out_features=5)

###clf v13: AdaBoost, median, numANOVA-7,nomChiSquared-7 
feature_select.feat_sel_Num_to_Cat(df_X_train_numfeatsel,y_train,k_out_features=7)

feature_select.feat_sel_Cat_to_Cat_chi2(df_X_train_catfeatsel,y_train,k_out_features=7)
