#Created on Fri Apr 9th 2021

#%%

#Import general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import libraries useful for building the pipeline and join their branches
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


#import modules created for data preparation phase
import my_utils
import missing_val_imput
import feature_select
import preprocessing
import adhoc_transf

#import libraries for data preparation phase
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder


#import libraries from modelling phase
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

#import classifiers
#import Ensemble Trees Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
import xgboost as xgb

#to save model fit with GridSearchCV and avoid longer waits
import joblib

#%%

#Loading the dataset
path_data=r'C:\Users\xdpemo\OneDrive - TUNI.fi\Documents\GitHub\Heart_Failure_Predictor\heart_failure_clinical_records_dataset.csv'

df=pd.read_csv(path_data)
df.head()

#%%Characterizing the data set
target_feature='DEATH_EVENT'
numerical_feats=['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']
nominal_feats=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
ordinal_feats=[]

len_numerical_feats=len(numerical_feats)
len_nominal_feats=len(nominal_feats)
len_ordinal_feats=len(ordinal_feats)

#%%
###################################################################################################################
#Step 0: Perform EDA to detect missing values, imbalanced data, strange characters,etc.
###################################################################################################################

##Statistical analysis
df.describe()
#%%
#Identifying missing values
my_utils.info_adhoc(df)
#%%
#Exploring wrong characters
my_utils.df_values(df)


#%%
###################################################################################################################
#Step 1 Solving wrong characters of dataset
###################################################################################################################
#Set column id as index


# CKD case does only have misspellingCorrector
# df_content_solver=Pipeline([('fx1', misspellingCorrector()),
#                             ('fx2',function2()),
#                             ('fx3',function3())
# ])

#%%
df=adhoc_transf.ageRounder().fit_transform(df)
my_utils.df_values(df)

#%%Performing numeric cast for numerical features
df.loc[:,numerical_feats]=adhoc_transf.Numeric_Cast_Column().fit_transform(df.loc[:,numerical_feats])
df[numerical_feats].dtypes


#%%Performing category cast for nominal features
df.loc[:,nominal_feats]=adhoc_transf.Category_Cast_Column().fit_transform(df.loc[:,nominal_feats])
df[nominal_feats].dtypes

#%%Performing category cast for ordinal features
df.loc[:,ordinal_feats]=adhoc_transf.Category_Cast_Column().fit_transform(df.loc[:,ordinal_feats])
df[ordinal_feats].dtypes


#%%
###################################################################################################################
##Step 2 Train-Test splitting
###################################################################################################################

#Split the dataset into train and test
test_ratio_split=0.3
train_set,test_set=train_test_split(df, test_size=test_ratio_split, random_state=42, stratify=df[target_feature])

X_train=train_set.drop(target_feature,axis=1)
y_train=train_set[target_feature].copy()

X_test=test_set.drop(target_feature,axis=1)
y_test=test_set[target_feature].copy()

#%%
###################################################################################################################
##Step 3 Label Encoding of target value
###################################################################################################################
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)
le.classes_
#%%
###################################################################################################################
##Step 4 Building pipelines for data preparation
###################################################################################################################

#Lets define 3 pipeline mode
#a) parallel approach where feature selection is performed in parallel 
# for numerical, nominal and categorical
#b) general approach where feature selection is performed as a whole for other features
#c) no feature selection is performed

#Before a data preprocessing will take place for each type of feature
pipeline_numeric_feat=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median')),
                                 ('scaler', MinMaxScaler())])

pipeline_numeric_feat_mean=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='mean')),
                                 ('scaler', MinMaxScaler())])

pipeline_nominal_feat=Pipeline([('data_missing',missing_val_imput.Category_Imputer()),                                 
                                 ('encoding', OrdinalEncoder())])#We dont use OneHotEncoder since it enlarges the number of nominal features 

pipeline_ordinal_feat=Pipeline([ ('data_missing',missing_val_imput.Category_Imputer(strategy='most_frequent')),
                                 ('encoding', OrdinalEncoder())])


#option a)
pipe_numeric_featsel=Pipeline([('data_prep',pipeline_numeric_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='wrapper_RFECV') )])
pipe_nominal_featsel=Pipeline([('data_prep',pipeline_nominal_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='wrapper_RFECV') )])
pipe_ordinal_featsel=Pipeline([('data_prep',pipeline_ordinal_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='wrapper_RFECV') )])

dataprep_pipe_opta=ColumnTransformer([('numeric_pipe',pipe_numeric_featsel,numerical_feats),
                                    ('nominal_pipe',pipe_nominal_featsel,nominal_feats),
                                    ('ordinal_pipe',pipe_ordinal_featsel,ordinal_feats)
                                ])

#option c)
dataprep_merge_feat=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_feats),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats),
                                    ('ordinal_pipe',pipeline_ordinal_feat,ordinal_feats)
                                ])

#%%
###################################################################################################################
##Step 5 Classifier initialization
###################################################################################################################
#Several ensemble classifier with Cross validation will be applied
#we take decision tree as base classifier

#Init the clasfifier
dectree_clf=DecisionTreeClassifier(random_state=42)
rndforest_clf=RandomForestClassifier(random_state=42)
extratree_clf=ExtraTreesClassifier(random_state=42)
ada_clf= AdaBoostClassifier(random_state=42)
xgboost_clf= xgb.XGBClassifier(random_state=42)
gradboost_clf=GradientBoostingClassifier(random_state=42)
voting_clf=VotingClassifier(estimators=[('rdf', rndforest_clf), ('xtra', extratree_clf), ('ada', ada_clf)], voting='soft')
#

#%%
###################################################################################################################
##Step 6 Scoring initialization
###################################################################################################################

#Lets define the scoring for the GridSearchCV
#Since the dataset is imbalanced we consider balanced_accuracy as estimator
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0),
    'precision':make_scorer(precision_score),
    'f1':make_scorer(f1_score),
    'roc_auc':make_scorer(roc_auc_score),
    'mcc':make_scorer(matthews_corrcoef)    
}

#%%
###################################################################################################################
##Step 7 Training the data set with GridSearchCV
###################################################################################################################


##7.a.1 Parallel approach
###################################################################################################################
full_parallel_pipe_opta=Pipeline([('data_prep',dataprep_pipe_opta),('clf',dectree_clf)])

full_parallel_pipe_opta.get_params().keys()

#%% Load the model saved to avoid a new fitting
clf_fpipe_a= joblib.load(r'GridSearchCV_results\HF_case_fullpaper_Notime\clf_fpipe_a.pkl')

#%%
param_grid_fpipe_a={'clf':[dectree_clf, rndforest_clf, extratree_clf, ada_clf, xgboost_clf,gradboost_clf,voting_clf ],
                    'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean','median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats+1)],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf','wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
                    }

# param_grid_fpipe_a={'clf':[dectree_clf, rndforest_clf],
#                     'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
#                      'data_prep__numeric_pipe__feat_sel__k_out_features':[1,2,3],
#                      'data_prep__numeric_pipe__feat_sel__strategy':['filter_num'],
#                      'data_prep__nominal_pipe__feat_sel__k_out_features':[1,2,3],
#                      'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat'],
#                      'data_prep__ordinal_pipe__feat_sel__k_out_features':[1,2,3],
#                      'data_prep__ordinal_pipe__feat_sel__strategy':['filter_mutinf']
#                     }

clf_fpipe_a=GridSearchCV(full_parallel_pipe_opta,param_grid_fpipe_a,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_fpipe_a.fit(X_train,y_train)

#%% Saving the model
joblib.dump(clf_fpipe_a, r'GridSearchCV_results\HF_case_fullpaper_Notime\clf_fpipe_a.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_a:', clf_fpipe_a.best_params_)
print('Params of best estimator of clf_fpipe_a:', clf_fpipe_a.best_params_)
print('Score of best estimator of clf_fpipe_a:', clf_fpipe_a.best_score_)

# Best estimator of clf_fpipe_a: {'clf': ExtraTreesClassifier(random_state=42), 'data_prep__nominal_pipe__feat_sel__k_out_features': 1, 'data_prep__nominal_pipe__feat_sel__strategy': 'filter_mutinf', 'data_prep__numeric_pipe__data_prep__data_missing__strategy': 'mean', 'data_prep__numeric_pipe__feat_sel__k_out_features': 4, 'data_prep__numeric_pipe__feat_sel__strategy': 'filter_num'}
# Params of best estimator of clf_fpipe_a: {'clf': ExtraTreesClassifier(random_state=42), 'data_prep__nominal_pipe__feat_sel__k_out_features': 1, 'data_prep__nominal_pipe__feat_sel__strategy': 'filter_mutinf', 'data_prep__numeric_pipe__data_prep__data_missing__strategy': 'mean', 'data_prep__numeric_pipe__feat_sel__k_out_features': 4, 'data_prep__numeric_pipe__feat_sel__strategy': 'filter_num'}
# Score of best estimator of clf_fpipe_a: 0.8804878048780488

#%% Saving the training results into dataframe
df_results_clf_fpipe_a=pd.DataFrame(clf_fpipe_a.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_a.to_excel(r'GridSearchCV_results\HF_case_fullpaper_Notime\train_results_clf_fpipe_a.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_a.refit
y_pred_clf_fpipe_a=clf_fpipe_a.predict(X_test)
test_results_clf_fpipe_a={'clf':['clf_fpipe_a'],
                 'params':[clf_fpipe_a.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_a)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_a)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_a)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_a)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_a,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_a)]    
    }

#%%
test_results_y_pred_clf_fpipe_a=pd.DataFrame(data=test_results_clf_fpipe_a)
test_results_y_pred_clf_fpipe_a.to_excel(r'GridSearchCV_results\HF_case_fullpaper_Notime\test_results_y_pred_clf_fpipe_a.xlsx',index=False)
#%%
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_a))
#Accuracy of test set 0.8111111111111111

#%%
##7.a.2 Parallel approach refitting with balanced_accuracy
###################################################################################################################

clf_fpipe_a_balacc= joblib.load(r'GridSearchCV_results\HF_case_fullpaper_Notime\clf_fpipe_a_balacc.pkl')

#%%
clf_fpipe_a_balacc=GridSearchCV(full_parallel_pipe_opta,param_grid_fpipe_a,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_fpipe_a_balacc.fit(X_train,y_train)

#%% Saving the model
joblib.dump(clf_fpipe_a_balacc, r'GridSearchCV_results\HF_case_fullpaper_Notime\clf_fpipe_a_balacc.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_a_balacc:', clf_fpipe_a_balacc.best_params_)
print('Params of best estimator of clf_fpipe_a_balacc:', clf_fpipe_a_balacc.best_params_)
print('Score of best estimator of clf_fpipe_a_balacc:', clf_fpipe_a_balacc.best_score_)

# Best estimator of clf_fpipe_a: {'clf': ExtraTreesClassifier(random_state=42), 'data_prep__nominal_pipe__feat_sel__k_out_features': 1, 'data_prep__nominal_pipe__feat_sel__strategy': 'filter_mutinf', 'data_prep__numeric_pipe__data_prep__data_missing__strategy': 'mean', 'data_prep__numeric_pipe__feat_sel__k_out_features': 4, 'data_prep__numeric_pipe__feat_sel__strategy': 'filter_num'}
# Params of best estimator of clf_fpipe_a: {'clf': ExtraTreesClassifier(random_state=42), 'data_prep__nominal_pipe__feat_sel__k_out_features': 1, 'data_prep__nominal_pipe__feat_sel__strategy': 'filter_mutinf', 'data_prep__numeric_pipe__data_prep__data_missing__strategy': 'mean', 'data_prep__numeric_pipe__feat_sel__k_out_features': 4, 'data_prep__numeric_pipe__feat_sel__strategy': 'filter_num'}
# Score of best estimator of clf_fpipe_a: 0.8540356195528609

#%% Saving the training results into dataframe
df_results_clf_fpipe_a_balacc=pd.DataFrame(clf_fpipe_a_balacc.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_a_balacc.to_excel(r'GridSearchCV_results\HF_case_fullpaper_Notime\train_results_clf_fpipe_a_balacc.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_a_balacc.refit
y_pred_clf_fpipe_a_balacc=clf_fpipe_a_balacc.predict(X_test)
test_results_clf_fpipe_a_balacc={'clf':['clf_fpipe_a'],
                 'params':[clf_fpipe_a_balacc.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_a_balacc)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_a_balacc)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_a_balacc)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_a_balacc)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_a_balacc,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_a_balacc)]    
    }

#%%
test_results_y_pred_clf_fpipe_a_balacc=pd.DataFrame(data=test_results_clf_fpipe_a_balacc)
test_results_y_pred_clf_fpipe_a_balacc.to_excel(r'GridSearchCV_results\HF_case_fullpaper_Notime\test_results_y_pred_clf_fpipe_a_balacc.xlsx',index=False)
#%%
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_a_balacc))#Accuracy of test set 0.7888888888888889


#%%
###################################################################################################################
##Step 7 Training the data set with GridSearchCV with no time feature (b)
###################################################################################################################


##7.b.1 Parallel approach with no time feature
###################################################################################################################

#%%

X_train_notime=X_train.drop(['time'], axis=1)
numerical_feats_notime=['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']
len_numerical_feats_notime= len(numerical_feats_notime)
dataprep_pipe_opta_notime=ColumnTransformer([('numeric_pipe',pipe_numeric_featsel,numerical_feats_notime),
                                    ('nominal_pipe',pipe_nominal_featsel,nominal_feats),
                                    ('ordinal_pipe',pipe_ordinal_featsel,ordinal_feats)
                                ])

full_parallel_pipe_opta_notime=Pipeline([('data_prep',dataprep_pipe_opta_notime),('clf',dectree_clf)])



#%% Load the model saved to avoid a new fitting
clf_fpipe_notime= joblib.load(r'GridSearchCV_results\HF_case_fullpaper\clf_fpipe_a.pkl')


#%%
#%%
param_grid_fpipe_a_notime={'clf':[dectree_clf, rndforest_clf, extratree_clf, ada_clf, xgboost_clf,gradboost_clf,voting_clf ],
                    'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean','median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats_notime+1)],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf','wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
                    }
clf_fpipe_a_notime=GridSearchCV(full_parallel_pipe_opta_notime,param_grid_fpipe_a_notime,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_fpipe_a_notime.fit(X_train_notime,y_train)

#%% Saving the model
joblib.dump(clf_fpipe_a_notime, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\HF_case\clf_fpipe_a_notime.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_a_notime:', clf_fpipe_a_notime.best_params_)
print('Params of best estimator of clf_fpipe_a_notime:', clf_fpipe_a_notime.best_params_)
print('Score of best estimator of clf_fpipe_a_notime:', clf_fpipe_a_notime.best_score_)

# Best estimator of clf_fpipe_a: {'clf': ExtraTreesClassifier(random_state=42), 'data_prep__nominal_pipe__feat_sel__k_out_features': 1, 'data_prep__nominal_pipe__feat_sel__strategy': 'filter_mutinf', 'data_prep__numeric_pipe__data_prep__data_missing__strategy': 'mean', 'data_prep__numeric_pipe__feat_sel__k_out_features': 4, 'data_prep__numeric_pipe__feat_sel__strategy': 'filter_num'}
# clf': VotingClassifier(estimators=[('rdf', RandomForestClassifier(random_state=42)),
#                              ('xtra', ExtraTreesClassifier(random_state=42)),
#                              ('ada', AdaBoostClassifier(random_state=42))],
#                  voting='soft'), 'data_prep__nominal_pipe__feat_sel__k_out_features': 4, 'data_prep__nominal_pipe__feat_sel__strategy': 'filter_mutinf', 'data_prep__numeric_pipe__data_prep__data_missing__strategy': 'median', 'data_prep__numeric_pipe__feat_sel__k_out_features': 6, 'data_prep__numeric_pipe__feat_sel__strategy': 'filter_num'
# Score of best estimator of clf_fpipe_a: 0.7804878048780488

#%% Saving the training results into dataframe
df_results_clf_fpipe_a_notime=pd.DataFrame(clf_fpipe_a_notime.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_a_notime.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\train_results_clf_fpipe_a_notime.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_a_notime.refit
X_test_notime=X_test.drop(['time'], axis=1)
y_pred_clf_fpipe_a_notime=clf_fpipe_a_notime.predict(X_test_notime)
test_results_clf_fpipe_a_notime={'clf':['clf_fpipe_a_notime'],
                 'params':[clf_fpipe_a_notime.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_a_notime)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_a_notime)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_a_notime)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_a_notime)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_a_notime,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_a_notime)]    
    }

#%%
test_results_y_pred_clf_fpipe_a_notime=pd.DataFrame(data=test_results_clf_fpipe_a_notime)
test_results_y_pred_clf_fpipe_a_notime.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_y_pred_clf_fpipe_a_notime.xlsx',index=False)
#%%
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_a_notime))
#Accuracy of test set 0.7222222222222222


##5.b.2 Parallel approach with no time feature and balanced accuracy
###################################################################################################################


#%% Load the model saved to avoid a new fitting
clf_fpipe_notime= joblib.load(r'GridSearchCV_results\HF_case_fullpaper\clf_fpipe_a_notime_balacc.pkl')


#%%
clf_fpipe_a_notime_balacc=GridSearchCV(full_parallel_pipe_opta_notime,param_grid_fpipe_a_notime,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_fpipe_a_notime_balacc.fit(X_train_notime,y_train)

#%% Saving the model
joblib.dump(clf_fpipe_a_notime_balacc, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\HF_case\clf_fpipe_a_notime_balacc.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_a_notime_balacc:', clf_fpipe_a_notime_balacc.best_params_)
print('Params of best estimator of clf_fpipe_a_notime_balacc:', clf_fpipe_a_notime_balacc.best_params_)
print('Score of best estimator of clf_fpipe_a_notime_balacc:', clf_fpipe_a_notime_balacc.best_score_)

# Best estimator of clf_fpipe_a: {'clf': ExtraTreesClassifier(random_state=42), 'data_prep__nominal_pipe__feat_sel__k_out_features': 1, 'data_prep__nominal_pipe__feat_sel__strategy': 'filter_mutinf', 'data_prep__numeric_pipe__data_prep__data_missing__strategy': 'mean', 'data_prep__numeric_pipe__feat_sel__k_out_features': 4, 'data_prep__numeric_pipe__feat_sel__strategy': 'filter_num'}
# clf': VotingClassifier(estimators=[('rdf', RandomForestClassifier(random_state=42)),
#                              ('xtra', ExtraTreesClassifier(random_state=42)),
#                              ('ada', AdaBoostClassifier(random_state=42))],
#                  voting='soft'), 'data_prep__nominal_pipe__feat_sel__k_out_features': 4, 'data_prep__nominal_pipe__feat_sel__strategy': 'filter_mutinf', 'data_prep__numeric_pipe__data_prep__data_missing__strategy': 'median', 'data_prep__numeric_pipe__feat_sel__k_out_features': 6, 'data_prep__numeric_pipe__feat_sel__strategy': 'filter_num'
# Score of best estimator of clf_fpipe_a: 0.7804878048780488

#%% Saving the training results into dataframe
df_results_clf_fpipe_a_notime_balacc=pd.DataFrame(clf_fpipe_a_notime_balacc.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_a_notime_balacc.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\train_results_clf_fpipe_a_notime_balacc.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_a_notime_balacc.refit
X_test_notime_balacc=X_test.drop(['time'], axis=1)
y_pred_clf_fpipe_a_notime_balacc=clf_fpipe_a_notime_balacc.predict(X_test_notime)
test_results_clf_fpipe_a_notime_balacc={'clf':['clf_fpipe_a_notime'],
                 'params':[clf_fpipe_a_notime_balacc.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_a_notime_balacc)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_a_notime_balacc)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_a_notime_balacc)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_a_notime_balacc)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_a_notime_balacc,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_a_notime_balacc)]    
    }

#%%
test_results_y_pred_clf_fpipe_a_notime_balacc=pd.DataFrame(data=test_results_clf_fpipe_a_notime_balacc)
test_results_y_pred_clf_fpipe_a_notime_balacc.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_y_pred_clf_fpipe_a_notime_balacc.xlsx',index=False)
#%%
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_a_notime_balacc))

##5.c general approach where feature selection is performed as a whole for other features
###################################################################################################################

full_parallel_pipe_optc=Pipeline([('data_prep',dataprep_merge_feat),('clf',dectree_clf)])

full_parallel_pipe_optc.get_params().keys()
# %%
#%% Load the model saved to avoid a new fitting
clf_fpipe_c= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_fpipe_c.pkl')

#%%
param_grid_fpipe_c={'clf':[dectree_clf, rndforest_clf, extratree_clf, ada_clf, xgboost_clf,gradboost_clf,voting_clf ],
                    'data_prep__numeric_pipe__data_missing__strategy':['mean','median']
                    }

# param_grid_fpipe_c={'clf':[dectree_clf, rndforest_clf ],
#                     'data_prep__numeric_pipe__data_missing__strategy':['mean','median']
#                     }

clf_fpipe_c=GridSearchCV(full_parallel_pipe_optc,param_grid_fpipe_c,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_fpipe_c.fit(X_train,y_train)

# %%#%% Saving the model
joblib.dump(clf_fpipe_c, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_fpipe_c.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_c:', clf_fpipe_c.best_params_)
print('Params of best estimator of clf_fpipe_c:', clf_fpipe_c.best_params_)
print('Score of best estimator of clf_fpipe_a:', clf_fpipe_c.best_score_)


#%% Saving the training results into dataframe
df_results_clf_fpipe_c=pd.DataFrame(clf_fpipe_c.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_c.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\train_results_clf_fpipe_c.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_c.refit
y_pred_clf_fpipe_c=clf_fpipe_c.predict(X_test)
test_results_clf_fpipe_c={'clf':['clf_fpipe_c'],
                 'params':[clf_fpipe_c.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_c)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_c)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_c)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_c)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_c,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_c)]    
    }

#%%
test_results_y_pred_clf_fpipe_c=pd.DataFrame(data=test_results_clf_fpipe_c)
test_results_y_pred_clf_fpipe_c.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_y_pred_clf_fpipe_c.xlsx',index=False)
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_c))

###################################################################################################################
#Step 8: Application of SCI-XAI per each type of classifier
###################################################################################################################

#v2:DecisionTree
##########################################################################################################################################
#%%
param_grid_v2_exp={'clf':[dectree_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean','median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats+1)],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf','wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
                    }

clf_v2_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v2_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v2_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v2_exp:', clf_v2_exp.best_score_) #Score of best estimator of clf_v2:0.8063281546040166

#%%
#Saving the results in an excel
df_results_v2_exp=pd.DataFrame(clf_v2_exp.cv_results_)
df_results_v2_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v2_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v2_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v2_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v2_exp.refit
y_pred_v2_exp = clf_v2_exp.predict(X_test)

test_results_DT={'clf':['clf_v2_exp'],
                 'params':[clf_v2_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v2_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v2_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v2_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v2_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v2_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v2_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v2_exp)]    
    }
#%%
test_results_DT_paper=pd.DataFrame(data=test_results_DT)
test_results_DT_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_DT_paper.xlsx',index=False)


#v3:Random Forest
###################################################################################################################
#%%
param_grid_v3_exp={'clf': [rndforest_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats+1)],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf','wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
     }

clf_v3_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v3_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v3_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v3_exp:', clf_v3_exp.best_score_) #Score of best estimator of clf_v3: 1

#Saving the results in an excel
df_results_v3_exp=pd.DataFrame(clf_v3_exp.cv_results_)
df_results_v3_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v3_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v3_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v3_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v3_exp.refit
y_pred_v3_exp = clf_v3_exp.predict(X_test)

test_results_RF={'clf':['clf_v3_exp'],
                 'params':[clf_v3_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v3_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v3_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v3_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v3_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v3_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v3_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v3_exp)]    
    }
#%%
test_results_RF_paper=pd.DataFrame(data=test_results_RF)
test_results_RF_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_RF_paper.xlsx',index=False)

#%%
#v4:Extra Trees
###################################################################################################################
#%%
param_grid_v4_exp={'clf':[extratree_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean','median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats+1)],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf','wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
                    }

clf_v4_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v4_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v4_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v4_exp:', clf_v4_exp.best_score_) #Score of best estimator of clf_v4:0.7623721106479727

#%%
#Saving the results in an excel
df_results_v4_exp=pd.DataFrame(clf_v4_exp.cv_results_)
df_results_v4_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v4_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v4_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v4_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v4_exp.refit
y_pred_v4_exp = clf_v4_exp.predict(X_test)

test_results_ET={'clf':['clf_v4_exp'],
                 'params':[clf_v4_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v4_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v4_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v4_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v4_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v4_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v4_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v4_exp)]    
    }
#%%
test_results_ET_paper=pd.DataFrame(data=test_results_ET)
test_results_ET_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_ET_paper.xlsx',index=False)



#%%
#v5:AdaBoost
###################################################################################################################
#%%
param_grid_v5_exp={'clf':[ada_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean','median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats+1)],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf','wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
                    }

clf_v5_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v5_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v5_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v5_exp:', clf_v5_exp.best_score_) #Score of best estimator of clf_v5:0.7623721106479727

#%%
#Saving the results in an excel
df_results_v5_exp=pd.DataFrame(clf_v5_exp.cv_results_)
df_results_v5_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v5_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v5_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v5_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v5_exp.refit
y_pred_v5_exp = clf_v5_exp.predict(X_test)

test_results_AB={'clf':['clf_v5_exp'],
                 'params':[clf_v5_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v5_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v5_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v5_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v5_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v5_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v5_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v5_exp)]    
    }
#%%
test_results_AB_paper=pd.DataFrame(data=test_results_AB)
test_results_AB_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_AB_paper.xlsx',index=False)

#%%
#v6:Gradient Boosting
###################################################################################################################
#%%
param_grid_v6_exp={'clf':[gradboost_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean','median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats+1)],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf','wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
                    }

clf_v6_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v6_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v6_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v6_exp:', clf_v6_exp.best_score_) #Score of best estimator of clf_v6:0.7623721106479727

#%%
#Saving the results in an excel
df_results_v6_exp=pd.DataFrame(clf_v6_exp.cv_results_)
df_results_v6_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v6_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v6_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v6_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v6_exp.refit
y_pred_v6_exp = clf_v6_exp.predict(X_test)

test_results_GB={'clf':['clf_v6_exp'],
                 'params':[clf_v6_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v6_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v6_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v6_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v6_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v6_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v6_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v6_exp)]    
    }
#%%
test_results_GB_paper=pd.DataFrame(data=test_results_GB)
test_results_GB_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_GB_paper.xlsx',index=False)


#%%
#v7:eXtreme Gradient Boosting
###################################################################################################################
#%%
param_grid_v7_exp={'clf':[xgboost_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean','median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats+1)],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf','wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
                    }

clf_v7_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v7_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v7_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v7_exp:', clf_v7_exp.best_score_) #Score of best estimator of clf_v7:0.7623721106479727

#%%
#Saving the results in an excel
df_results_v7_exp=pd.DataFrame(clf_v7_exp.cv_results_)
df_results_v7_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v7_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v7_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v7_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v7_exp.refit
y_pred_v7_exp = clf_v7_exp.predict(X_test)

test_results_XGB={'clf':['clf_v7_exp'],
                 'params':[clf_v7_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v7_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v7_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v7_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v7_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v7_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v7_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v7_exp)]    
    }
#%%
test_results_XGB_paper=pd.DataFrame(data=test_results_XGB)
test_results_XGB_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_XGB_paper.xlsx',index=False)

#%%
#v8:Max Voting
###################################################################################################################
#%%
param_grid_v8_exp={'clf':[voting_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean','median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats+1)],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf','wrapper_RFE'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
                    }

clf_v8_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v8_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v8_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v8_exp:', clf_v8_exp.best_score_) #Score of best estimator of clf_v8:0.7623721106479727

#%%
#Saving the results in an excel
df_results_v8_exp=pd.DataFrame(clf_v8_exp.cv_results_)
df_results_v8_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v8_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v8_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v8_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v8_exp.refit
y_pred_v8_exp = clf_v8_exp.predict(X_test)

test_results_MaxV={'clf':['clf_v8_exp'],
                 'params':[clf_v8_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v8_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v8_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v8_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v8_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v8_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v8_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v8_exp)]    
    }
#%%
test_results_MaxV_paper=pd.DataFrame(data=test_results_MaxV)
test_results_MaxV_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_MaxV_paper.xlsx',index=False)

#%%

#Dataframe with the best estimator per each classifier pipe application applied to the Xtest
# ###################################################################################################################

overall_results={'clf':['clf_v1_exp','clf_v2_exp','clf_v3_exp','clf_v4_exp','clf_v5_exp','clf_v6_exp','clf_v7_exp','clf_v8_exp'],
                 'params':[clf_fpipe_a.best_params_, clf_v2_exp.best_params_, clf_v3_exp.best_params_, clf_v4_exp.best_params_,clf_v5_exp.best_params_,clf_v6_exp.best_params_,clf_v7_exp.best_params_,clf_v8_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_a),accuracy_score(y_test, y_pred_v2_exp),accuracy_score(y_test, y_pred_v3_exp),accuracy_score(y_test, y_pred_v4_exp),accuracy_score(y_test, y_pred_v5_exp),accuracy_score(y_test, y_pred_v6_exp),accuracy_score(y_test, y_pred_v7_exp),accuracy_score(y_test, y_pred_v8_exp)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_a),recall_score(y_test, y_pred_v2_exp),recall_score(y_test, y_pred_v3_exp),recall_score(y_test, y_pred_v4_exp),recall_score(y_test, y_pred_v5_exp),recall_score(y_test, y_pred_v6_exp),recall_score(y_test, y_pred_v7_exp),recall_score(y_test, y_pred_v8_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_a,pos_label=0),recall_score(y_test, y_pred_v2_exp,pos_label=0),recall_score(y_test, y_pred_v3_exp,pos_label=0),recall_score(y_test, y_pred_v4_exp,pos_label=0),recall_score(y_test, y_pred_v5_exp,pos_label=0),recall_score(y_test, y_pred_v6_exp,pos_label=0),recall_score(y_test, y_pred_v7_exp,pos_label=0),recall_score(y_test, y_pred_v8_exp,pos_label=0)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_a), f1_score(y_test, y_pred_v2_exp), f1_score(y_test, y_pred_v3_exp),f1_score(y_test, y_pred_v4_exp),f1_score(y_test, y_pred_v5_exp),f1_score(y_test, y_pred_v6_exp), f1_score(y_test, y_pred_v7_exp), f1_score(y_test, y_pred_v8_exp)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_a),precision_score(y_test, y_pred_v2_exp),precision_score(y_test, y_pred_v3_exp),precision_score(y_test, y_pred_v4_exp),precision_score(y_test, y_pred_v5_exp),precision_score(y_test, y_pred_v6_exp),precision_score(y_test, y_pred_v7_exp),precision_score(y_test, y_pred_v8_exp)],                                 
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_a),roc_auc_score(y_test, y_pred_v2_exp),roc_auc_score(y_test, y_pred_v3_exp),roc_auc_score(y_test, y_pred_v4_exp),roc_auc_score(y_test, y_pred_v4_exp),roc_auc_score(y_test, y_pred_v6_exp),roc_auc_score(y_test, y_pred_v7_exp),roc_auc_score(y_test, y_pred_v8_exp)]    
    }


df_overall_results_paper=pd.DataFrame(data=overall_results)
df_overall_results_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Chronic_Kidney_Disease_Predictor\GridSearchCV_results_DMKD\df_test_overall_results_paper.xlsx',index=False)

#######################
###################################################################################################################
#Step 7: Build a param grid for the best combination per each type of classifier considered
###################################################################################################################

#v2:DecisionTree BC(Best Combination)
###################################################################################################################
#%%
param_grid_v2_BC_exp={'clf':[dectree_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[2],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_mutinf'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[3],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat']
                    }

clf_v2_BC_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v2_BC_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v2_BC_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v2_BC_exp:', clf_v2_BC_exp.best_score_) #Score of best estimator of clf_v2_BC:0.7678666161424782

#%%
#Saving the results in an excel
df_results_v2_BC_exp=pd.DataFrame(clf_v2_BC_exp.cv_results_)
df_results_v2_BC_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v2_BC_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v2_BC_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v2_BC_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v2_BC_exp.refit
y_pred_v2_BC_exp = clf_v2_BC_exp.predict(X_test)

test_results_DT_BC={'clf':['clf_v2_BC_exp'],
                 'params':[clf_v2_BC_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v2_BC_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v2_BC_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v2_BC_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v2_BC_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v2_BC_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v2_BC_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v2_BC_exp)]    
    }
#%%
test_results_DT_BC_paper=pd.DataFrame(data=test_results_DT_BC)
test_results_DT_BC_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_DT_BC_paper.xlsx',index=False)


#v3:Random Forest BC(Best Combination)
###################################################################################################################

#%%
param_grid_v3_BC_exp={'clf': [rndforest_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[4],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[2],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_mutinf']
     }

clf_v3_BC_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v3_BC_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v3_BC_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v3_BC_exp:', clf_v3_BC_exp.best_score_) #Score of best estimator of clf_v3_BC:  0.8205475558923835

#Saving the results in an excel
df_results_v3_BC_exp=pd.DataFrame(clf_v3_BC_exp.cv_results_)
df_results_v3_BC_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v3_BC_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v3_BC_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v3_BC_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v3_BC_exp.refit
y_pred_v3_BC_exp = clf_v3_BC_exp.predict(X_test)

test_results_RF_BC={'clf':['clf_v3_BC_exp'],
                 'params':[clf_v3_BC_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v3_BC_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v3_BC_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v3_BC_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v3_BC_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v3_BC_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v3_BC_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v3_BC_exp)]    
    }
#%%
test_results_RF_BC_paper=pd.DataFrame(data=test_results_RF_BC)
test_results_RF_BC_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_RF_BC_paper.xlsx',index=False)

#%%
#v4:Extra Trees BC(Best Combination)
###################################################################################################################
#%%
param_grid_v4_BC_exp={'clf':[extratree_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[4],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[1],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_mutinf']
                    }

clf_v4_BC_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v4_BC_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v4_BC_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v4_BC_exp:', clf_v4_BC_exp.best_score_) #Score of best estimator of clf_v4_BC:0.8212201591511936

#%%
#Saving the results in an excel
df_results_v4_BC_exp=pd.DataFrame(clf_v4_BC_exp.cv_results_)
df_results_v4_BC_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v4_BC_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v4_BC_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v4_BC_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v4_BC_exp.refit
y_pred_v4_BC_exp = clf_v4_BC_exp.predict(X_test)

test_results_ET_BC={'clf':['clf_v4_BC_exp'],
                 'params':[clf_v4_BC_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v4_BC_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v4_BC_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v4_BC_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v4_BC_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v4_BC_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v4_BC_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v4_BC_exp)]    
    }
#%%
test_results_ET_BC_paper=pd.DataFrame(data=test_results_ET_BC)
test_results_ET_BC_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_ET_BC_paper.xlsx',index=False)



#%%
#v5:AdaBoost (Best Combination)
###################################################################################################################
#%%
param_grid_v5_BC_exp={'clf':[ada_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[5],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_num'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[1],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat']
                    }

clf_v5_BC_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v5_BC_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v5_BC_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v5_BC_exp:', clf_v5_BC_exp.best_score_) #Score of best estimator of clf_v5_BC:0.8260136415308829

#%%
#Saving the results in an excel
df_results_v5_BC_exp=pd.DataFrame(clf_v5_BC_exp.cv_results_)
df_results_v5_BC_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v5_BC_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v5_BC_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v5_BC_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v5_BC_exp.refit
y_pred_v5_BC_exp = clf_v5_BC_exp.predict(X_test)

test_results_AB_BC={'clf':['clf_v5_BC_exp'],
                 'params':[clf_v5_BC_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v5_BC_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v5_BC_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v5_BC_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v5_BC_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v5_BC_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v5_BC_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v5_BC_exp)]    
    }
#%%
test_results_AB_BC_paper=pd.DataFrame(data=test_results_AB_BC)
test_results_AB_BC_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_AB_BC_paper.xlsx',index=False)

#%%
#v6:Gradient Boosting (Best Combination)
###################################################################################################################
#%%
param_grid_v6_BC_exp={'clf':[gradboost_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[4],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_mutinf'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[2],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_mutinf']
                    }

clf_v6_BC_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v6_BC_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v6_BC_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v6_BC_exp:', clf_v6_BC_exp.best_score_) #Score of best estimator of clf_v6_BC:0.7817544524441076

#%%
#Saving the results in an excel
df_results_v6_BC_exp=pd.DataFrame(clf_v6_BC_exp.cv_results_)
df_results_v6_BC_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v6_BC_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v6_BC_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v6_BC_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v6_BC_exp.refit
y_pred_v6_BC_exp = clf_v6_BC_exp.predict(X_test)

test_results_GB_BC={'clf':['clf_v6_BC_exp'],
                 'params':[clf_v6_BC_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v6_BC_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v6_BC_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v6_BC_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v6_BC_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v6_BC_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v6_BC_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v6_BC_exp)]    
    }
#%%
test_results_GB_BC_paper=pd.DataFrame(data=test_results_GB_BC)
test_results_GB_BC_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_GB_BC_paper.xlsx',index=False)


#%%
#v7:eXtreme Gradient Boosting  (Best Combination)
###################################################################################################################

#%%
param_grid_v7_BC_exp={'clf':[xgboost_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[4],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_mutinf'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[5],
                    'data_prep__nominal_pipe__feat_sel__strategy':['wrapper_RFE']
                    }

clf_v7_BC_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v7_BC_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v7_BC_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v7_BC_exp:', clf_v7_BC_exp.best_score_) #Score of best estimator of clf_v7_BC:0.8367279272451686

#%%
#Saving the results in an excel
df_results_v7_BC_exp=pd.DataFrame(clf_v7_BC_exp.cv_results_)
df_results_v7_BC_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v7_BC_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v7_BC_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v7_BC_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v7_BC_exp.refit
y_pred_v7_BC_exp = clf_v7_BC_exp.predict(X_test)

test_results_XGB_BC={'clf':['clf_v7_BC_exp'],
                 'params':[clf_v7_BC_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v7_BC_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v7_BC_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v7_BC_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v7_BC_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v7_BC_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v7_BC_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v7_BC_exp)]    
    }
#%%
test_results_XGB_BC_paper=pd.DataFrame(data=test_results_XGB_BC)
test_results_XGB_BC_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_XGB_BC_paper.xlsx',index=False)

#%%
#v8_BC:Max Voting (Best Combination)
###################################################################################################################
#%%
param_grid_v8_BC_exp={'clf':[voting_clf],
            'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
                    'data_prep__numeric_pipe__feat_sel__k_out_features':[4],
                    'data_prep__numeric_pipe__feat_sel__strategy':['filter_mutinf'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[1],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_mutinf']
                    }

clf_v8_BC_exp=GridSearchCV(full_parallel_pipe_opta,param_grid_v8_BC_exp,scoring=scoring,refit='balanced_accuracy', cv=5,n_jobs=None)
clf_v8_BC_exp.fit(X_train,y_train)
#%%
print('Score of best estimator of clf_v8_BC_exp:', clf_v8_BC_exp.best_score_) #Score of best estimator of clf_v8_BC:0.7951212580522926

#%%
#Saving the results in an excel
df_results_v8_BC_exp=pd.DataFrame(clf_v8_BC_exp.cv_results_)
df_results_v8_BC_exp.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_results_v8_BC_exp.xlsx',index=False)
#Saving the model
joblib.dump(clf_v8_BC_exp, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\clf_v8_BC_exp.pkl', compress=1)

#%%
#Obtaining classification  with test set
clf_v8_BC_exp.refit
y_pred_v8_BC_exp = clf_v8_BC_exp.predict(X_test)

test_results_MaxV_BC={'clf':['clf_v8_BC_exp'],
                 'params':[clf_v8_BC_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_v8_BC_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_v8_BC_exp)],
                 'f1_test':[f1_score(y_test, y_pred_v8_BC_exp)],
                 'precision_test':[precision_score(y_test, y_pred_v8_BC_exp)],
                 'recall_test':[recall_score(y_test, y_pred_v8_BC_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_v8_BC_exp,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_v8_BC_exp)]    
    }
#%%
test_results_MaxV_BC_paper=pd.DataFrame(data=test_results_MaxV_BC)
test_results_MaxV_BC_paper.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\test_results_MaxV_BC_paper.xlsx',index=False)

#%%

#Dataframe with the best estimator per each classifier applied to the Xtest
###################################################################################################################

overall_results={'clf':['clf_v1_exp','clf_v2_BC_exp','clf_v3_BC_exp','clf_v4_BC_exp','clf_v5_BC_exp','clf_v6_BC_exp','clf_v7_BC_exp','clf_v8_BC_exp'],
                 'params':[clf_fpipe_a.best_params_, clf_v2_BC_exp.best_params_, clf_v3_BC_exp.best_params_, clf_v4_BC_exp.best_params_,clf_v5_BC_exp.best_params_,clf_v6_BC_exp.best_params_,clf_v7_BC_exp.best_params_,clf_v8_BC_exp.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_a),accuracy_score(y_test, y_pred_v2_BC_exp),accuracy_score(y_test, y_pred_v3_BC_exp),accuracy_score(y_test, y_pred_v4_BC_exp),accuracy_score(y_test, y_pred_v5_BC_exp),accuracy_score(y_test, y_pred_v6_BC_exp),accuracy_score(y_test, y_pred_v7_BC_exp),accuracy_score(y_test, y_pred_v8_BC_exp)],
                 'balanced_accuracy_test':[balanced_accuracy_score(y_test, y_pred_clf_fpipe_a),balanced_accuracy_score(y_test, y_pred_v2_BC_exp),balanced_accuracy_score(y_test, y_pred_v3_BC_exp),balanced_accuracy_score(y_test, y_pred_v4_BC_exp),balanced_accuracy_score(y_test, y_pred_v5_BC_exp),balanced_accuracy_score(y_test, y_pred_v6_BC_exp),balanced_accuracy_score(y_test, y_pred_v7_BC_exp),balanced_accuracy_score(y_test, y_pred_v8_BC_exp)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_a),recall_score(y_test, y_pred_v2_BC_exp),recall_score(y_test, y_pred_v3_BC_exp),recall_score(y_test, y_pred_v4_BC_exp),recall_score(y_test, y_pred_v5_BC_exp),recall_score(y_test, y_pred_v6_BC_exp),recall_score(y_test, y_pred_v7_BC_exp),recall_score(y_test, y_pred_v8_BC_exp)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_a,pos_label=0),recall_score(y_test, y_pred_v2_BC_exp,pos_label=0),recall_score(y_test, y_pred_v3_BC_exp,pos_label=0),recall_score(y_test, y_pred_v4_BC_exp,pos_label=0),recall_score(y_test, y_pred_v5_BC_exp,pos_label=0),recall_score(y_test, y_pred_v6_BC_exp,pos_label=0),recall_score(y_test, y_pred_v7_BC_exp,pos_label=0),recall_score(y_test, y_pred_v8_BC_exp,pos_label=0)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_a), f1_score(y_test, y_pred_v2_BC_exp), f1_score(y_test, y_pred_v3_BC_exp),f1_score(y_test, y_pred_v4_BC_exp),f1_score(y_test, y_pred_v5_BC_exp),f1_score(y_test, y_pred_v6_BC_exp), f1_score(y_test, y_pred_v7_BC_exp), f1_score(y_test, y_pred_v8_BC_exp)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_a),precision_score(y_test, y_pred_v2_BC_exp),precision_score(y_test, y_pred_v3_BC_exp),precision_score(y_test, y_pred_v4_BC_exp),precision_score(y_test, y_pred_v5_BC_exp),precision_score(y_test, y_pred_v6_BC_exp),precision_score(y_test, y_pred_v7_BC_exp),precision_score(y_test, y_pred_v8_BC_exp)],                                 
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_a),roc_auc_score(y_test, y_pred_v2_BC_exp),roc_auc_score(y_test, y_pred_v3_BC_exp),roc_auc_score(y_test, y_pred_v4_BC_exp),roc_auc_score(y_test, y_pred_v5_BC_exp),roc_auc_score(y_test, y_pred_v6_BC_exp),roc_auc_score(y_test, y_pred_v7_BC_exp),roc_auc_score(y_test, y_pred_v8_BC_exp)]    
    }


df_overall_results_paper_BC=pd.DataFrame(data=overall_results)
df_overall_results_paper_BC.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\GridSearchCV_results\HF_case_fullpaper\df_test_overall_results_paper_BC.xlsx',index=False)

###################################################################################################################
##Step 9 Extracting features selected
###################################################################################################################

#%%
#Copy train and test set for extracting features
X_train_featsel=X_train.copy()
y_train_featsel=y_train.copy()

X_test_featsel=X_test.copy()
y_test_featsel=y_test.copy()

#%%Before extracting features, data missing imputer and normalization/encoding must be carried out
X_train_featsel[numerical_feats]=pipeline_numeric_feat.fit_transform(X_train_featsel[numerical_feats])
X_train_featsel[nominal_feats]=pipeline_nominal_feat.fit_transform(X_train_featsel[nominal_feats])

X_test_featsel[numerical_feats]=pipeline_numeric_feat.fit_transform(X_test_featsel[numerical_feats])
X_test_featsel[nominal_feats]=pipeline_nominal_feat.fit_transform(X_test_featsel[nominal_feats])

y_train_featsel=le.fit_transform(y_train_featsel)
y_test_featsel=le.fit_transform(y_test_featsel)

#%%
X_train_feat=X_train.copy()
X_train_feat[numerical_feats]=pipeline_numeric_feat.fit_transform(X_train_feat[numerical_feats])
X_train_feat[nominal_feats]=pipeline_nominal_feat.fit_transform(X_train_feat[nominal_feats])
# %%
#ANOVA for numerical features median imputation
feature_select.feat_sel_Num_to_Cat(X_train_feat[numerical_feats], y_train, 'all')

# Feature of feat_sel_Num_to_Cat age: 5.437503
# Feature of feat_sel_Num_to_Cat creatinine_phosphokinase: 3.649016
# Feature of feat_sel_Num_to_Cat ejection_fraction: 18.060764
# Feature of feat_sel_Num_to_Cat platelets: 0.184443
# Feature of feat_sel_Num_to_Cat serum_creatinine: 25.004401
# Feature of feat_sel_Num_to_Cat serum_sodium: 5.237164
# Feature of feat_sel_Num_to_Cat time: 81.90987

# %%
#Multinf for numerical features median imputation
feature_select.feat_sel_Cat_to_Cat_mutinf(X_train_feat[numerical_feats], y_train, 'all')

# Feature of feat_sel_Cat_to_Cat mutual info age: 0.000000
# Feature of feat_sel_Cat_to_Cat mutual info creatinine_phosphokinase: 0.034643
# Feature of feat_sel_Cat_to_Cat mutual info ejection_fraction: 0.134579
# Feature of feat_sel_Cat_to_Cat mutual info platelets: 0.000000
# Feature of feat_sel_Cat_to_Cat mutual info serum_creatinine: 0.079792
# Feature of feat_sel_Cat_to_Cat mutual info serum_sodium: 0.000000
# Feature of feat_sel_Cat_to_Cat mutual info time: 0.251652


#%%
#ANOVA for numerical features mean imputation
X_train_feat[numerical_feats]=pipeline_numeric_feat_mean.fit_transform(X_train_feat[numerical_feats])
feature_select.feat_sel_Num_to_Cat(X_train_feat[numerical_feats], y_train, 'all')

# Feature of feat_sel_Num_to_Cat age: 5.437503
# Feature of feat_sel_Num_to_Cat creatinine_phosphokinase: 3.649016
# Feature of feat_sel_Num_to_Cat ejection_fraction: 18.060764
# Feature of feat_sel_Num_to_Cat platelets: 0.184443
# Feature of feat_sel_Num_to_Cat serum_creatinine: 25.004401
# Feature of feat_sel_Num_to_Cat serum_sodium: 5.237164
# Feature of feat_sel_Num_to_Cat time: 81.909871

# %%
#Multinf for numerical features mean imputation
feature_select.feat_sel_Cat_to_Cat_mutinf(X_train_feat[numerical_feats], y_train, 'all')

# Feature of feat_sel_Cat_to_Cat mutual info age: 0.007223
# Feature of feat_sel_Cat_to_Cat mutual info creatinine_phosphokinase: 0.039027
# Feature of feat_sel_Cat_to_Cat mutual info ejection_fraction: 0.150100
# Feature of feat_sel_Cat_to_Cat mutual info platelets: 0.016049
# Feature of feat_sel_Cat_to_Cat mutual info serum_creatinine: 0.111043
# Feature of feat_sel_Cat_to_Cat mutual info serum_sodium: 0.028930
# Feature of feat_sel_Cat_to_Cat mutual info time: 0.250139

# %%
#Chi2 for nominal features
feature_select.feat_sel_Cat_to_Cat_chi2(X_train_feat[nominal_feats], y_train, 'all')

# Feature of feat_sel_Cat_to_Cat chi2 anaemia: 0.721515
# Feature of feat_sel_Cat_to_Cat chi2 diabetes: 0.069512
# Feature of feat_sel_Cat_to_Cat chi2 high_blood_pressure: 0.655560
# Feature of feat_sel_Cat_to_Cat chi2 sex: 0.100904
# Feature of feat_sel_Cat_to_Cat chi2 smoking: 0.334204

#%%
#Multinf for nominal features
feature_select.feat_sel_Cat_to_Cat_mutinf(X_train_feat[nominal_feats], y_train, 'all')

# Feature of feat_sel_Cat_to_Cat mutual info anaemia: 0.000000
# Feature of feat_sel_Cat_to_Cat mutual info diabetes: 0.018095
# Feature of feat_sel_Cat_to_Cat mutual info high_blood_pressure: 0.001180
# Feature of feat_sel_Cat_to_Cat mutual info sex: 0.000000
# Feature of feat_sel_Cat_to_Cat mutual info smoking: 0.001034

#%%
#RFE for nominal features
feature_select.feat_sel_RFE(X_train_feat[nominal_feats], y_train, 5)

#anaemia	diabetes	high_blood_pressure	sex	smoking

#%%
###################################################################################################################
##Step 10 Building the decision tree to calculate the fidelity score for  each classifier's best estimator
# Fidelity formula --> F=acc(decision_tree)/acc(best classifier's estimator )
###################################################################################################################


#Random Forest
numerical_feats_rf=['age','ejection_fraction','serum_creatinine','time']
nominal_feats_rf=[ 'diabetes', 'high_blood_pressure']
dataprep_merge_feat_rf=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_feats_rf),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_rf)
                                    ])

pipe_fidelity_rf=Pipeline([('data_prep',dataprep_merge_feat_rf),
                          ('clf', dectree_clf)])

pipe_fidelity_rf.fit(X_train,y_train)
y_pred_fidelity_rf = pipe_fidelity_rf.predict(X_test)
balanced_accuracy_score(y_test, y_pred_fidelity_rf)#0.7956472583380441

#%%
#Extra Trees
numerical_feats_rf=['age','ejection_fraction','serum_creatinine','time']
nominal_feats_rf=[ 'diabetes']
dataprep_merge_feat_rf=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_feats_rf),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_rf)
                                    ])

pipe_fidelity_rf=Pipeline([('data_prep',dataprep_merge_feat_rf),
                          ('clf', dectree_clf)])

pipe_fidelity_rf.fit(X_train,y_train)
y_pred_fidelity_rf = pipe_fidelity_rf.predict(X_test)
balanced_accuracy_score(y_test, y_pred_fidelity_rf)#0.7956472583380441

#%%
#AdaBoost
numerical_feats_rf=['age','ejection_fraction','serum_creatinine','serum_sodium','time']
nominal_feats_rf=['anaemia']
dataprep_merge_feat_rf=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_feats_rf),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_rf)
                                    ])

pipe_fidelity_rf=Pipeline([('data_prep',dataprep_merge_feat_rf),
                          ('clf', dectree_clf)])

pipe_fidelity_rf.fit(X_train,y_train)
y_pred_fidelity_rf = pipe_fidelity_rf.predict(X_test)
balanced_accuracy_score(y_test, y_pred_fidelity_rf)#0.8038439796495196

#%%
#Gradient Boosting
numerical_feats_rf=['creatinine_phosphokinase','ejection_fraction','serum_creatinine','time']
nominal_feats_rf=['diabetes', 'high_blood_pressure']
dataprep_merge_feat_rf=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_feats_rf),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_rf)
                                    ])

pipe_fidelity_rf=Pipeline([('data_prep',dataprep_merge_feat_rf),
                          ('clf', dectree_clf)])

pipe_fidelity_rf.fit(X_train,y_train)
y_pred_fidelity_rf = pipe_fidelity_rf.predict(X_test)
balanced_accuracy_score(y_test, y_pred_fidelity_rf)#0.7538157150932729

#%%
#XGBoost
numerical_feats_rf=['creatinine_phosphokinase','ejection_fraction','serum_creatinine','time']
nominal_feats_rf=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
dataprep_merge_feat_rf=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_feats_rf),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_rf)
                                    ])

pipe_fidelity_rf=Pipeline([('data_prep',dataprep_merge_feat_rf),
                          ('clf', dectree_clf)])

pipe_fidelity_rf.fit(X_train,y_train)
y_pred_fidelity_rf = pipe_fidelity_rf.predict(X_test)
balanced_accuracy_score(y_test, y_pred_fidelity_rf)#0.7456189937817976

#%%
#Voting Classifier
numerical_feats_rf=['creatinine_phosphokinase','ejection_fraction','serum_creatinine','time']
nominal_feats_rf=['diabetes']
dataprep_merge_feat_rf=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_feats_rf),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_rf)
                                    ])

pipe_fidelity_rf=Pipeline([('data_prep',dataprep_merge_feat_rf),
                          ('clf', dectree_clf)])

pipe_fidelity_rf.fit(X_train,y_train)
y_pred_fidelity_rf = pipe_fidelity_rf.predict(X_test)
balanced_accuracy_score(y_test, y_pred_fidelity_rf)#0.7801017524024874


#%%

#################################################
##Step 6 Hyperparameter tunning to enhance the accuracy
#################################################
# %%
