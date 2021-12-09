#%%
#Import general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import libraries useful for building the pipeline and join their branches
from sklearn.pipeline import Pipeline
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
###########################################################################
#  Explainability Analisys
# The "most explainable" classifier is ExtraTrees by assessing the FIR ratio
# Different explainability method are considered: implicit feature importance, feature permutation, SHAP and PDP
###########################################################################


#Loading the dataset
path_data=r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\Heart_Failure_Predictor\heart_failure_clinical_records_dataset.csv'

df=pd.read_csv(path_data)
df.head()
#%%Characterizing the data set
target_feature='DEATH_EVENT'
numerical_feats=['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']
nominal_feats=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

df=adhoc_transf.ageRounder().fit_transform(df)
my_utils.df_values(df)

#%%
###################################################################################################################
#Step 1 Solving wrong characters of dataset
###################################################################################################################

#Performing numeric cast for numerical features
df.loc[:,numerical_feats]=adhoc_transf.Numeric_Cast_Column().fit_transform(df.loc[:,numerical_feats])
df[numerical_feats].dtypes

#Performing category cast for nominal features
df.loc[:,nominal_feats]=adhoc_transf.Category_Cast_Column().fit_transform(df.loc[:,nominal_feats])
df[nominal_feats].dtypes

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
                                    ('nominal_pipe',pipe_nominal_featsel,nominal_feats)
                                ])

#%%
###################################################################################################################
##Step 5 Tailoring the dataset with the feature selected of the best classifier
###################################################################################################################

features_selected_xtree=['age','ejection_fraction','serum_creatinine','time','diabetes']
X_train_feat_sel=X_train[features_selected_xtree]
X_test_feat_sel=X_test[features_selected_xtree]

extratree_clf=ExtraTreesClassifier(random_state=42)

#%%
###################################################################################################################
##Step 6 The estimator is refited with those feature selected
#########################################################

numerical_feats_xtree=['age','ejection_fraction','serum_creatinine','time']
nominal_feats_xtree=['diabetes']


dataprep_merge_feat_xtree=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat_mean,numerical_feats_xtree),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats_xtree)
                                    ])


#%%
X_train_featsel=dataprep_merge_feat_xtree.fit_transform(X_train_feat_sel)
df_X_train_featsel=pd.DataFrame(X_train_featsel, columns=features_selected_xtree)
df_X_train_featsel.head()
#%%
X_test_featsel=dataprep_merge_feat_xtree.fit_transform(X_test_feat_sel)
df_X_test_featsel=pd.DataFrame(X_test_featsel, columns=features_selected_xtree)
df_X_test_featsel.head()
#%%
extratree_clf.fit(df_X_train_featsel,y_train)

#%%
###################################################################################################################
## Step 11 SHAP explainability 
##############################################################

#global explainability
##################################################################
import shap
shap.initjs()

pipe_shap_xtree=Pipeline([('data_prep',dataprep_merge_feat_xtree),
                          ('clf', extratree_clf)])
pipe_shap_xtree.fit(X_train_feat_sel, y_train)
#%%
explainer_train=shap.explainers.Tree(pipe_shap_xtree.named_steps['clf'], pipe_shap_xtree.named_steps['data_prep'].fit_transform(X_train_feat_sel))
shap_values_train=explainer_train.shap_values(pipe_shap_xtree.named_steps['data_prep'].fit_transform(X_train_feat_sel))

#%%
# explainer=shap.explainers.Tree(pipe_shap_xtree.named_steps['clf'], pipe_shap_xtree.named_steps['data_prep'].fit_transform(X_train_feat_sel))
# shap_values=explainer.shap_values(X_train_feat_sel)

#%%
np.shape(shap_values_train)

# %%
shap.summary_plot(shap_values_train, X_train_feat_sel,plot_type="bar")
# %%
shap.summary_plot(shap_values_train[0], X_train_feat_sel,plot_type="dot")
#%%
shap.summary_plot(shap_values_train[1], X_train_feat_sel,plot_type="dot")
# %%
shap.summary_plot(shap_values_train[1], X_train_feat_sel,plot_type="bar")


#%%
X_train_feat_sel.head()

#%%
#local explainability
##########################################################
y_pred_test= extratree_clf.predict(df_X_test_featsel)
print('y_pred',y_pred_test)
print('y_test',y_test)
#%%
y_pred_train = extratree_clf.predict(df_X_train_featsel)
print('y_pred_train',y_pred_train)
print('y_train',y_train)

#%%
explainer_test=shap.explainers.Tree(pipe_shap_xtree.named_steps['clf'], pipe_shap_xtree.named_steps['data_prep'].fit_transform(X_test_feat_sel))
shap_values_test=explainer_test.shap_values(pipe_shap_xtree.named_steps['data_prep'].fit_transform(X_test_feat_sel))

# %%
np.shape(shap_values_test)
# %%
#%%
#True negative instance
index_TN_shap=0
print(df_X_test_featsel.iloc[index_TN_shap])
print(X_test_feat_sel.iloc[index_TN_shap])
print('Actual Label:', y_test[index_TN_shap])
print('Predicted Label:', y_pred_test[index_TN_shap])
choosen_instance_tn=df_X_test_featsel.iloc[index_TN_shap]

# %%
shap_values_test_tn = explainer_test.shap_values(choosen_instance_tn)
shap.force_plot(explainer_test.expected_value[0], shap_values_test_tn[0], X_test_feat_sel.iloc[index_TN_shap])

# %%
shap.plots._waterfall.waterfall_legacy(explainer_test.expected_value[1], shap_values_test_tn[1], X_test_feat_sel.iloc[index_TN_shap])

# %%
index_TP_shap=2
print(df_X_test_featsel.iloc[index_TP_shap])
print(X_test_feat_sel.iloc[index_TP_shap])
print('Actual Label:', y_test[index_TP_shap])
print('Predicted Label:', y_pred_test[index_TP_shap])
choosen_instance_tp=df_X_test_featsel.iloc[index_TP_shap]

# %%
shap_values_test_tp = explainer_test.shap_values(choosen_instance_tp)
shap.force_plot(explainer_test.expected_value[1], shap_values_test_tp[1], X_test_feat_sel.iloc[index_TP_shap])
#%%
shap.plots._waterfall.waterfall_legacy(explainer_test.expected_value[1], shap_values_test_tp[1], X_test_feat_sel.iloc[index_TP_shap])

#%%
index_TN_shap_train = 1
print(df_X_train_featsel.iloc[index_TN_shap_train])
print(X_train_feat_sel.iloc[index_TN_shap_train])
print('Actual Label:', y_train[index_TN_shap_train])
print('Predicted Label:', y_pred_train[index_TN_shap_train])
choosen_instance_tn_train=df_X_train_featsel.iloc[index_TN_shap_train]
#%%
shap_values_tn_train = explainer_train.shap_values(choosen_instance_tn_train)
shap.force_plot(explainer_train.expected_value[1], shap_values_tn_train[1], choosen_instance_tn_train)
#%%
shap.plots._waterfall.waterfall_legacy(explainer_train.expected_value[1], shap_values_tn_train[1], X_train_feat_sel.iloc[index_TN_shap_train])

#%%
shap_values_tn_train
#%%
index_TP_shap_train = 7
print(df_X_train_featsel.iloc[index_TP_shap_train])
print(X_train_feat_sel.iloc[index_TP_shap_train])
print('Actual Label:', y_train[index_TP_shap_train])
print('Predicted Label:', y_pred_train[index_TP_shap_train])
choosen_instance_tp_train=X_train_feat_sel.iloc[index_TP_shap_train]
#%%
shap_values_tp_train = explainer_train.shap_values(choosen_instance_tp_train)
shap.force_plot(explainer_train.expected_value[1], shap_values_tp_train[1], X_train_feat_sel.iloc[index_TP_shap_train])

# %%
shap.plots._waterfall.waterfall_legacy(explainer_train.expected_value[1], shap_values_tp_train[1], X_train_feat_sel.iloc[index_TP_shap_train])
# %%
