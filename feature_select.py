# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 06:49:37 2020

@author: k5000751
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import statsmodels.api as sm


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR 
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.base import BaseEstimator, TransformerMixin

################
##Filter methods
################
#Function to take a df with num attributes and cat target and return df with k_out_features

def feat_sel_Num_to_Cat(X, y, k_out_features):
    fs,p=SelectKBest(score_func=f_classif, k=k_out_features)
    df_sel=fs.fit_transform(X, y)
    if k_out_features=='all':
        for i in range(len(fs.scores_)):
            print('Feature of feat_sel_Num_to_Cat %s: %f' % (X.columns[i], fs.scores_[i],p.scores_[i]))
    #we have to create a dataframe
    cols=fs.get_support(indices=True)
    df_sel=X.iloc[:,cols]
    return df_sel

#Function to take a df with cat attributes and cat target and return df with k_out_features
def feat_sel_Cat_to_Cat_chi2(X, y, k_out_features):
    #chi-squared feature selection
    fs_chi2=SelectKBest(score_func=chi2, k=k_out_features)
    df_chi2=fs_chi2.fit_transform(X,y)
    if k_out_features=='all':
        for i in range(len(fs_chi2.scores_)):
            print('Feature of feat_sel_Cat_to_Cat chi2 %s: %f' % (X.columns[i], fs_chi2.scores_[i]))
    #we have to create a dataframe
    cols_chi2=fs_chi2.get_support(indices=True)
    df_chi2=X.iloc[:,cols_chi2]
        
    return df_chi2

def feat_sel_Cat_to_Cat_mutinf(X, y, k_out_features):
   
    #Mutual information feature selection
    fs_mutinf=SelectKBest(score_func=mutual_info_classif, k=k_out_features)
    df_mutinf=fs_mutinf.fit_transform(X,y)
    if k_out_features=='all':
        for i in range(len(fs_mutinf.scores_)):
            print('Feature of feat_sel_Cat_to_Cat mutual info %s: %f' % (X.columns[i], fs_mutinf.scores_[i]))
    cols_mutinf=fs_mutinf.get_support(indices=True)
    df_mutinf=X.iloc[:,cols_mutinf]
    return df_mutinf

#################
##Wrapper methods
#################

#RFE method with logistic regression or other specified estimator
def feat_sel_RFE(X,y,k_out_features=None, estimator='LogisticRegression'):
        
    #allows different kind of estimators
    if estimator=='LogisticRegression':
        model=LogisticRegression(solver='lbfgs', max_iter=2000)
    if estimator=='SVR':
        model=SVR(kernel='linear')
    
    #check the optimus number of output features for which the accuracy is highest
    #if k_out_features==None by default the number of output features is the half of total
    if k_out_features=='all':
        nof_list=np.arange(1,X.shape[1])            
        high_score=0
        #Variable to store the optimum features
        nof=0           
        score_list =[]
        from sklearn.model_selection import train_test_split
        for n in range(len(nof_list)):
            rfe = RFE(model,nof_list[n])
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
            X_train_rfe = rfe.fit_transform(X_train,y_train)
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe,y_train)
            score = model.score(X_test_rfe,y_test)
            score_list.append(score)
            if(score>high_score):
                high_score = score
                nof = nof_list[n]
        print("Optimum number of features: %d" %nof)
        print("Score with %d features: %f" % (nof, high_score))
        k_out_features=nof
                
    #obtain the pruned resultant df of features
    rfe=RFE(model,k_out_features)
    fit = rfe.fit(X, y)
    X_pruned=rfe.fit_transform(X,y)
    mask=fit.support_
    X_pruned=X.iloc[:,mask]
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % X.columns[fit.support_])
    print("Feature Ranking: %s" % fit.ranking_)
    print()
    return X_pruned
        
#RFECV with the LogisticRegression estimator as default
def feat_sel_RFECV(X,y, estimator="LogisticRegression"):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    if estimator=='LogisticRegression':
        model=LogisticRegression(solver='lbfgs', max_iter=2000)
    if estimator=='SVR':
        model=SVR(kernel='linear')
    rfe=RFECV(model)    
    fit=rfe.fit(X, y)
    X_pruned=rfe.transform(X)
    mask=fit.support_
    X_pruned=X.iloc[:,mask]
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % X.columns[fit.support_])
    print("Feature Ranking: %s" % fit.ranking_)
    return X_pruned

#Backward elimination
def feat_sel_backElimination(X,y):
#coded extracted from: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
    cols = list(X.columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y,X_1).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    print(selected_features_BE)
    X_pruned=X[selected_features_BE]
    return X_pruned

#############
##Embedded methods
############
#Lasso linear model as regularizer adapted from https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
def feat_sel_Lasso(X,y):
    reg = Lasso()
    reg.fit(X, y)
    #print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    #print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    feat_sel=coef!=0
    X_pruned=X[feat_sel.index[feat_sel]]
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    return X_pruned

#Lasso linear model with CV as regularizer extracted from https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
def feat_sel_LassoCV(X,y):
    reg = LassoCV()
    reg.fit(X, y)
    #print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    #print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    feat_sel=coef!=0
    print('feat_sel in LassoCV: ', feat_sel)    
    X_pruned=X[feat_sel.index[feat_sel]]
    print("LassoCV picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    return X_pruned

#RidgeCV linear model as regularizer extracted and adapted from https://www.datacamp.com/community/tutorials/feature-selection-python
def feat_sel_RidgeCV(X,y):
    reg = RidgeCV()
    reg.fit(X, y)
    #print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)
    #print("Best score using built-in RidgeCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    feat_sel=coef>=0
    X_pruned=X[feat_sel.index[feat_sel]]
    print("RidgeCV picked " + str(sum(coef > 0)) + " variables and eliminated the other " +  str(sum(coef <= 0)) + " variables")
    return X_pruned

#Ridge linear model as regularizer extracted and adapted from https://www.datacamp.com/community/tutorials/feature-selection-python
def feat_sel_Ridge(X,y):
    reg = Ridge()
    reg.fit(X, y)
    #print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)
    #print("Best score using built-in RidgeCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    feat_sel=coef>=0
    X_pruned=X[feat_sel.index[feat_sel]]
    print("Ridge picked " + str(sum(coef > 0)) + " variables and eliminated the other " +  str(sum(coef <= 0)) + " variables")
    return X_pruned

#Wrapper transformer for Backward Elimination feature selection
class Backward_Elimination (BaseEstimator, TransformerMixin):
    def __init__(self):
        print('\n>>>>>>>>Calling init() from Backward_Elimination')
        
    def fit(self,X,y=None):
        print('\n>>>>>>>>Calling fit() from Backward_Elimination')
        cols = list(X.columns)
        pmax = 1
        while (len(cols)>0):
            p= []
            X_1 = X[cols]
            X_1 = sm.add_constant(X_1)
            model = sm.OLS(y,X_1).fit()
            p = pd.Series(model.pvalues.values[1:],index = cols)      
            pmax = max(p)
            feature_with_p_max = p.idxmax()
            if(pmax>0.05):
                cols.remove(feature_with_p_max)
            else:
                break
        self.selected_features_BE = cols
        return self
    
    def transform(self, X):
        print('\n>>>>>>>>Calling transform() from Backward_Elimination')
        X_pruned=X[self.selected_features_BE]
        return X_pruned
    
#Wrapper transformer for Backward Elimination feature selection
class LassoCV_FeatSel(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('\n>>>>>>>>Calling init() from LassoCV_FeatSel')
        self.reg=LassoCV()
    
    def fit(self,X,y=None):
        print('\n>>>>>>>>Calling fit() from LassoCV_FeatSel')
        self.reg.fit(X,y)
        return self
        
    def transform(self,X):
        print('\n>>>>>>>>Calling transform() from LassoCV_FeatSel')
        coef = pd.Series(self.reg.coef_, index = X.columns)
        feat_sel=coef!=0        
        X_pruned=X[feat_sel.index[feat_sel]]
        return X_pruned
    
#Wrapper transformer for Backward Elimination feature selection
class RidgeCV_FeatSel(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('\n>>>>>>>>Calling init() from RidgeCV_FeatSel')
        self.reg=RidgeCV()
        
    def fit(self,X,y=None):
        print('\n>>>>>>>>Calling fit() from RidgeCV_FeatSel')
        self.reg.fit(X,y)
        return self
        
    def transform(self,X):
        print('\n>>>>>>>>Calling transform() from RidgeCV_FeatSel')
        coef = pd.Series(reg.coef_, index = X.columns)
        feat_sel=coef>=0
        X_pruned=X[feat_sel.index[feat_sel]]        
        return X_pruned
    
        
        
#Class to run feature selection depending on different strategies
class Feature_Selector(BaseEstimator, TransformerMixin):
    #filter_num: performing ANOVA valid for numeric input and category output
    #filter_cat: performing chi2 valid for category input and category output
    #filter_mutinf:performing mutual information valid for numeric/category input and category output
    #wrapper_RFECV: performing RFECV with two optional regressor LogisticRegression(by defaurl) or SVR valid for numeric/category input and category output
    #wrapper_BackElim:performing Backward Elimination valid for numeric/category input and category output
    #LassoCV: performing LassoCV valid for numeric/category input and category output
    #RidgeCV: performing RidgeCV valid for numeric/category input and category output
    
    #def __init__(self,y_train,strategy='wrapper_RFECV',k_out_features=5, rfe_estimator='LogisticRegression'):       
    def __init__(self,strategy='wrapper_RFECV',k_out_features=5, rfe_estimator='LogisticRegression'):
        print('\n>>>>>>>>Calling init() from Feature_Selector')
        
        #self.y_train=y_train
        self.strategy=strategy
        self.k_out_features=k_out_features
        self.rfe_estimator=rfe_estimator
        
        if self.strategy=='filter_num':
            self.feat_sel=SelectKBest(score_func=f_classif, k=self.k_out_features)
            
        if self.strategy=='filter_cat':
            self.feat_sel=SelectKBest(score_func=chi2, k=self.k_out_features)
            
        if self.strategy=='filter_mutinf':
            self.feat_sel=SelectKBest(score_func=mutual_info_classif, k=self.k_out_features)
            
        if self.strategy=='wrapper_RFECV':
            if self.rfe_estimator=='LogisticRegression':
                self.model=LogisticRegression(solver='lbfgs', max_iter=2000)
            if self.rfe_estimator=='SVR':
                self.model=SVR(kernel='linear')
            self.feat_sel=RFECV(self.model)
        
        if self.strategy=='wrapper_RFE':
            if self.rfe_estimator=='LogisticRegression':
                self.model=LogisticRegression(solver='lbfgs', max_iter=2000)
            if self.rfe_estimator=='SVR':
                self.model=SVR(kernel='linear')
            self.feat_sel=RFE(self.model, n_features_to_select=k_out_features)
        
        if self.strategy=='wrapper_BackElim':
            self.feat_sel=Backward_Elimination()   
        
        if self.strategy=='LassoCV':
            self.feat_sel=LassoCV_FeatSel()
        
        if self.strategy=='RidgeCV':
            self.feat_sel=RidgeCV_FeatSel()
        
        
    def fit(self,X,y=None):
        print('\n>>>>>>>>Calling fit() from Feature_Selector')
        #index=X.index
        self.y_train=y
        #print('\n********Inside fit() from Feature_Selector y_train length:', self.y_train.size)        
        #print('\n********Calling fit() from Feature_Selector X length: ', X.shape[0])
        
        self.feat_sel.fit(X,self.y_train)
        return self
    
    def transform(self,X,y=None):
        print('\n>>>>>>>>Calling transform() from Feature_Selector')
        X_pruned=self.feat_sel.transform(X)
        return X_pruned
        
            
        
            
        
        
            