# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:44:17 2020

@author: k5000751
"""
#This class must include as much as function needed depeding on the 
#mispelling or wrong character detected in the dataset
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

#Class for correcting misspelling of features and target columns
class ageRounder(BaseEstimator, TransformerMixin):
    def rounder (self,df):
    #Some fetures content seems to have the character \t.
    #Let's remove such character for the sake of consistency
        #print('\n>>>>>>>>Calling rounder')      
        df['age']=np.around(df['age'])
        return df
    
    def __init__(self):
        print('\n>>>>>>>>Calling init() from ageRounder')
            
    def fit(self, X, y=None):
        print('\n>>>>>>>>Calling fit() from ageRounder')
        return self
    
    def transform(self,X,y=None):
        print('\n>>>>>>>>Calling transform() from ageRounder')        
        df=self.rounder(X)       
        return df
    
    def fit_transform(self, X, y=None,):
        return self.fit(X, y).transform(X, y)
    
#**************************************************************
#************Below it is inherited from CKD predictor
#**************************************************************
def misspellingCorrector(df):
    df.iloc[:] = df.iloc[:].str.replace(r'\t','')
    df.iloc[:] = df.iloc[:].str.replace(r' ','')
    return df


#Class for correcting misspelling of features and target columns
class misspellingTransformer_old(BaseEstimator, TransformerMixin):
    def misspelling (self,df):
    #Some fetures content seems to have the character \t.
    #Let's remove such character for the sake of consistency
        for i in range(0, len(df.columns)):
            if df.dtypes[i]==np.object:
                df.iloc[:,i] = df.iloc[:,i].str.replace(r'\t','')
                df.iloc[:,i] = df.iloc[:,i].str.replace(r' ','')
        return df
    
    def __init__(self,target_name):
        print('\n>>>>>>>>Calling init() from misspelling')
        self.target_name=target_name
    
    def fit(self, X, y=None):
        print('\n>>>>>>>>Calling fit() from misspelling')
        return self
    
    def transform(self,X,y=None):
        print('\n>>>>>>>>Calling transform() from misspelling')        
        df=pd.concat([X,y],axis=1)
        df=self.misspelling(df)       
        y=df.loc[:,self.target_name] 
        X=df.drop(self.target_name,axis=1)              
        return X,y
    def fit_transform(self, X, y=None,):
        return self.fit(X, y).transform(X, y)

#Class for correcting misspelling of features and target columns
class misspellingTransformer(BaseEstimator, TransformerMixin):
    def misspelling (self,df):
    #Some fetures content seems to have the character \t.
    #Let's remove such character for the sake of consistency
        print('\n>>>>>>>>Calling misspelling')      
        
        for i in range(0, len(df.columns)):            
            if df.dtypes[i]==np.object:
                #df.iloc[:,i] = df.iloc[:,i].str.replace(r'\t','')
                #df.iloc[:,i] = df.iloc[:,i].str.replace(r' ','')
                df.iloc[:,i] = df.iloc[:,i].replace(r'\t','')
                df.iloc[:,i] = df.iloc[:,i].replace(r' ','')
        return df
    
    def __init__(self):
        print('\n>>>>>>>>Calling init() from misspelling')
            
    def fit(self, X, y=None):
        print('\n>>>>>>>>Calling fit() from misspelling')
        return self
    
    def transform(self,X,y=None):
        print('\n>>>>>>>>Calling transform() from misspelling')        
        df=self.misspelling(X)       
        return df
    
    def fit_transform(self, X, y=None,):
        return self.fit(X, y).transform(X, y)

#Class for downcasting last category value of features 'al' and 'sg'
class CastDown(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('\n>>>>>>>>Calling init() from CastDown')
    
    def fit(self,X, y=None):
        print('\n>>>>>>>>Calling fit() from CastDown')
        return self
    
    def transform(self, X, y=None):
        print('\n>>>>>>>>Calling transform() from CastDown')
        for i in range(0, len(X.columns)):            
            X.iloc[:,i] = X.iloc[:,i].replace(5,4)
        return X
        # X[:,0][X[:,0]==5]=4
        # X[:,1][X[:,1]==5]=4
        # return X

    
#Class to be included in TransfromColumn pipeline for casting numeric columns to float64 
class Numeric_Cast_Column(BaseEstimator, TransformerMixin):  
    def __init__(self):
        print('\n>>>>>>>>Calling init() from Numeric_Cast_Column')        
    def fit(self, X,y=None):
        #print('inside fit Numeric_Cast, tuple leng', len(X))
        print('\n>>>>>>>>Calling fit() from Numeric_Cast_Column')        
        return self
    
    def transform(self, X,y=None):    
        #print ('Content of X', X)
        print('\n>>>>>>>>Calling transform() from Numeric_Cast_Column')        
        for i in range(X.shape[1]):
            #df[i]=pd.to_numeric(df[i],errors='coerce')
            X.iloc[:,i]=pd.to_numeric(X.iloc[:,i],errors='coerce')        
        return X    
    def fit_transform(self, X,y=None):    
        return self.fit(X, y).transform(X, y)


#Class to be included in TransfromColumn pipeline for casting numeric columns to float64 
class Category_Cast_Column(BaseEstimator, TransformerMixin):  
    def __init__(self):
        print('\n>>>>>>>>Calling init() from Category_Cast_Column')        
    def fit(self, X,y=None):
        #print('inside fit Numeric_Cast, tuple leng', len(X))
        print('\n>>>>>>>>Calling fit() from Category_Cast_Column')        
        return self
    
    def transform(self, X,y=None):    
        #print ('Content of X', X)
        print('\n>>>>>>>>Calling transform() from Category_Cast_Column')        
        for i in range(X.shape[1]):
            #df[i]=pd.to_numeric(df[i],errors='coerce')
            X.iloc[:,i]=X.iloc[:,i].astype('category')
        return X    
    def fit_transform(self, X,y=None):    
        return self.fit(X, y).transform(X, y)





class ageRounder(BaseEstimator, TransformerMixin):
    def rounder (self,df):
    #Some fetures content seems to have the character \t.
    #Let's remove such character for the sake of consistency
        print('\n>>>>>>>>Calling rounder')      
        df['age']=np.around(df.loc[:,'age'])
        return df
    
    def __init__(self):
        print('\n>>>>>>>>Calling init() from ageRounder')
            
    def fit(self, X, y=None):
        print('\n>>>>>>>>Calling fit() from ageRounder')
        return self
    
    def transform(self,X,y=None):
        print('\n>>>>>>>>Calling transform() from ageRounder')        
        df=self.rounder(X)       
        return df
    
    def fit_transform(self, X, y=None,):
        return self.fit(X, y).transform(X, y)