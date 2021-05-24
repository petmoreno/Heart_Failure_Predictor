# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:52:51 2020

@author: k5000751
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#return the numerical features from a subdataframe or df coming from feature selection
def get_numerical_columns(df, numerical_features):
    num_feat_remain=df.columns[df.columns.isin(numerical_features)]
    return num_feat_remain

#return the categoy features from a subdataframe or df coming from feature selection    
def get_category_columns(df,category_features):
    cat_feat_remain=df.columns[df.columns.isin(category_features)]
    return cat_feat_remain
    
    
def minmaxscaler(df):
    scaler=MinMaxScaler()
    df_scaled=scaler.fit_transform(df)
    return df_scaled

def standardscaler(df):
    scaler=StandardScaler()
    df_scaled=scaler.fit_transform(df)
    return df_scaled

#General method for scaling/normalizing and return x_train and y_train
def preprocessing(df,target,scaler='minmax',num_feat='None',cat_feat='None'):
    if num_feat!='None':
        num_feat_remain=get_numerical_columns(df,num_feat)
    if cat_feat !='None':
        cat_feat_remain=get_category_columns(df,cat_feat)
    if scaler=='standard':
        df[num_feat_remain]=standardscaler(df[num_feat_remain])
    else:
        df[num_feat_remain]=minmaxscaler(df[num_feat_remain])
    
    
    X_train=df.drop(target, axis=1)
    y_train=df[target].copy()
     
    return X_train, y_train
