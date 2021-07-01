# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:34:50 2020

@author: k5000751
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

#This file contains functions defined to handle missing values imputation
import my_utils
import time

from sklearn.base import BaseEstimator, TransformerMixin

#Print the number of rows remaining by setting the threshold in dropna from 0 to total N of features (no nan value in dataset)
def print_df_threshold_shape(df):
    for i in range (len(df.columns)+1):
        print('Shape of df_partialclean with threshold {} is: {}'.format(i,str(df.dropna(thresh=i).shape[0])))
        
#Retutn DataFrame with dropna with threshold parameter and show info adhoc    
def drop_threshold_info (df, threshold):
    df_totalclean_threshold=df.dropna(thresh=threshold)
    my_utils.info_adhoc(df_totalclean_threshold)
    return df_totalclean_threshold


#Class that handle imputation of NaN values in numeric features
class Numeric_Imputer(BaseEstimator, TransformerMixin):
    #Different modes of imputation depending on strategy parameter
    #median:simple imputation by considering median of feature values
    #mean: simple imputation by considering mean of feature values
    #iterative: multiple imputation by using IterativeImputer
    #knn: KNN imputation
    def __init__(self,strategy='median'):
        print('\n',time.ctime(),'>>>>>>>>Calling init() from Numeric_Imputer')
        self.strategy=strategy
        
        if self.strategy=='median':
            self.num_imputer=SimpleImputer(strategy=self.strategy)
        if self.strategy=='mean':
            self.num_imputer=SimpleImputer(strategy=self.strategy)
        if self.strategy=='iterative':
            self.num_imputer=IterativeImputer(random_state=0, sample_posterior='True')
        if self.strategy=='knn':
            self.num_imputer=KNNImputer()
    
    def fit(self,X,y=None):
        print('\n',time.ctime(),'>>>>>>>>Calling fit() from Numeric_Imputer')
        self.num_imputer.fit(X,y)
        return self
    
    def transform(self,X,y=None):
        print('\n',time.ctime(),'>>>>>>>>Calling transform() from Numeric_Imputer')
        X=self.num_imputer.transform(X)
        return X
    
#Class that handle imputation of NaN values in category features    
class Category_Imputer(BaseEstimator, TransformerMixin):
    #Different modes of imputation depending on strategy parameter
    #most_frequent:simple imputation by considering most frequent feature's value
    #constant: simple imputation by adding new category unknown to NaN values
    
    def __init__(self,strategy='most_frequent'):
        print('\n',time.ctime(),'>>>>>>>>Calling init() from Category_Imputer')
        self.strategy=strategy
        if self.strategy=='most_frequent':
            self.cat_imputer=SimpleImputer(strategy=self.strategy)
        if self.strategy=='constant':
            self.cat_imputer=SimpleImputer(strategy=self.strategy,fill_value='unknown')
        
    
    def fit(self,X,y=None):
        print('\n',time.ctime(),'>>>>>>>>Calling fit() from Category_Imputer')
        self.cat_imputer.fit(X,y)
        return self
    
    def transform(self,X,y=None):
        print('\n',time.ctime(),'>>>>>>>>Calling transform() from Category_Imputer')
        X=self.cat_imputer.transform(X)
        return X

    
#create a dataframe using SimpleImputer for num features with strategy defined
def simpleImputeNum(df, strategy):
        df_imp_num=df[num_features]
        num_imputer=SimpleImputer(strategy=strategy)
        arr_imp_num=num_imputer.fit_transform(df_imp_num)
        df_imp_num=pd.DataFrame(arr_imp_num, columns=df_imp_num.columns)
        return df_imp_num
    #Function that perform multiple imputation to a dataframe with numerical attributes
def multipleImputNum(df,num_features):
        df_impMult_num=df[num_features]
        imp_mult_num=IterativeImputer(random_state=0, sample_posterior='True')#set sample_posterior='True' for using to multiple imputation
        arr_impMult_num=imp_mult_num.fit_transform(df_impMult_num)
        df_impMult_num=pd.DataFrame(arr_impMult_num, columns=df_impMult_num.columns)
        return df_impMult_num
    
    #Function that perform KNN imputation to a dataframe with numerical attributes
def knnImputNum(df, num_features):
        df_impKNN_num=df[num_features]
        knn_imp=KNNImputer()
        arr_impKNN_num=knn_imp.fit_transform(df_impKNN_num)
        df_impKNN_num=pd.DataFrame(arr_impKNN_num, columns=df_impKNN_num.columns)
        return df_impKNN_num


#create a dataframe using SimpleImputer for cat features with strategy defined
def simpleImputeCat(df,cat_features, strategy='most_frequent', fill_value=None):
    df_imp_cat=df[cat_features]
    #first approach: assign unkown category to NaN values
    if strategy=='constant':
        cat_unk_imputer=SimpleImputer(strategy='constant', fill_value=fill_value)
        arr_imp_cat_unk=cat_unk_imputer.fit_transform(df_imp_cat)
        df_imp_cat=pd.DataFrame(arr_imp_cat_unk, columns=df_imp_cat.columns)
       
    #second_approach:assign most_frequent strategy
    else:
        cat_mostfq_imputer=SimpleImputer(strategy=strategy)
        arr_imp_cat_mostfq=cat_mostfq_imputer.fit_transform(df_imp_cat)
        df_imp_cat=pd.DataFrame(arr_imp_cat_mostfq, columns=df_imp_cat.columns)          
    
    return df_imp_cat



#Function that perform multiple imputation to a dataframe with cat attributes. PROBLEM:IterativeImputer does seem to work with category attributes
def multipleImputCat(df, cat_features):
    #Convert categories to numbers
    df_impMult_cat=df[cat_features]
    for i in range(len(cat_features)):
        df_impMult_cat.loc[:,cat_features[i]]=df_impMult_cat.loc[:,cat_features[i]].cat.codes
    
    #The NaN has been coded to -1. So we have to revert it for the imputer
    df_impMult_cat.replace(to_replace=-1,value=np.NAN,inplace=True)
    df_impMult_cat.head()
    #Apply iterative imputer
    #****Review
    imp_mult_cat=IterativeImputer(initial_strategy='most_frequent',random_state=0, sample_posterior='True')#set sample_posterior='True' for using to multiple imputation)
    arr_impMult_cat=imp_mult_cat.fit_transform(df_impMult_cat)
    df_impMult_cat=pd.DataFrame(arr_impMult_cat, columns=df_impMult_cat.columns)
    return df_impMult_cat

#Function that perform KNN imputation to a dataframe with numerical attributes
def knnImputNum(df, num_features):
    df_impKNN_num=df[num_features]
    knn_imp=KNNImputer()
    arr_impKNN_num=knn_imp.fit_transform(df_impKNN_num)
    df_impKNN_num=pd.DataFrame(arr_impKNN_num, columns=df_impKNN_num.columns)
    return df_impKNN_num

#Function that perform KNN imputation to a dataframe with category attributes.PROBLEM:KNNImputer does seem to work with category attributes
def knnImputCat(df, cat_features):
    ##Convert categories to numbers
    df_impKNN_cat=df[cat_features]
    for i in range(len(cat_features)):
     df_impKNN_cat.loc[:,cat_features[i]]=df_impKNN_cat.loc[:,cat_features[i]].cat.codes
    
    #The NaN has been coded to -1. So we have to revert it for the imputer
    df_impKNN_cat.replace(to_replace=-1,value=np.NAN,inplace=True)
    
    #****Review
    #Apply KNN imputer
    impKNN_cat=KNNImputer()
    impKNN_cat_arr=impKNN_cat.fit_transform(df_impKNN_cat)
    df_impKNN_cat=pd.DataFrame(impKNN_cat_arr, columns=df_impKNN_cat.columns)
    
    return df_impKNN_cat
