import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score#, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier(n_neighbors=5)
  var = []  #collect test_error/train_error where error based on F1 score
  for i in range(1, n):
    train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                    random_state=i, stratify=labels)
    model.fit(train_X, train_y)  #train model
    train_pred = model.predict(train_X)           #predict against training set
    test_pred = model.predict(test_X)             #predict against test set
    train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
    test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
    f1_ratio = test_f1/train_f1          #take the ratio
    var.append(f1_ratio)

  rs_value = sum(var)/len(var)  #get average ratio value
  #rs_value  
  idx = np.array(abs(var - rs_value)).argmin()  #find the index of the smallest value
  return idx

class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.' #f'{self.__class__.__name__} gets class name
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    ##Check if X is a dataframe
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
    
    #now check to see if all keys are contained in column
    column_set = set(X[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    self.column_list=column_list
    self.action=action

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    set_column=set(self.column_list)-set(X.columns.to_list())

    if self.action == 'keep':
      assert set_column == set(), f'{self.__class__.__name__}.transform does not contain these columns to keep: {set_column} .'
      X_=X.copy()
      X_ = X[self.column_list]
    if self.action == 'drop':
      X_=X.copy()
      if set_column != set():
        print(f"\nWarning: {self.__class__.__name__} does not contain these columns to drop: {set_column}.\n")
      X_= X_.drop(columns=self.column_list, errors ='ignore')

    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
 

class OHETransformer(BaseEstimator, TransformerMixin):

  def __init__(self, target_column, dummy_na=False, drop_first=False):  
    self.target_column = target_column

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'
    X_ = X.copy()
    X_=pd.get_dummies(X,prefix=self.target_column,prefix_sep='_', columns=[self.target_column],dummy_na=False, drop_first=False)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class Sigma3Transformer(BaseEstimator, TransformerMixin):

  def __init__(self, column_name):
    self.column_name = column_name  

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.column_name in X.columns.to_list(), f'unknown column {self.column_name}'
    assert all([isinstance(v, (int, float)) for v in X[self.column_name].to_list()])

    m=X[self.column_name].mean()
    sig=X[self.column_name].std()
    s3min=(m-3*sig)
    s3max=(m+3*sig)
    X_=X.copy()
    X_[self.column_name] = X_[self.column_name].clip(lower=s3min, upper=s3max)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class TukeyTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, target_column,fence):
    self.target_column = target_column  
    self.fence=fence

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    fence_value=['inner','outer']
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
    assert self.fence in fence_value, f'invalid fence "{self.fence}" passed. Fence should be {fence_value[0]} or {fence_value[1]}'
    #assert all([isinstance(v, (int, float)) for v in X[self.column_name].to_list()])

    q1 = X[self.target_column].quantile(0.25)
    q3 = X[self.target_column].quantile(0.75)
    iqr = q3-q1
    outer_low = q1-3*iqr
    outer_high = q3+3*iqr

    inner_low = q1-1.5*iqr
    inner_high = q3+1.5*iqr
    X_=X.copy()
    if self.fence == 'inner':
      X_[self.target_column] = X_[self.target_column].clip(lower=inner_low, upper=inner_high)
      return X_
    if self.fence == 'outer':
      X_[self.target_column] = X_[self.target_column].clip(lower=outer_low, upper=outer_high)
      return X_
    

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass  #takes no arguments

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    X_=X.copy()
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    column_name=X_.columns.to_list()
    scaler_df=pd.DataFrame(scaler.fit_transform(X),columns = column_name)
    return scaler_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform"):
    #your code
      from sklearn.impute import KNNImputer
      self.n_neighbors = n_neighbors
      self.weights = weights
      self.KNNImputer = KNNImputer
     

  def fit(self, X, y = None):
      print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
      return X
      
  def transform(self, X):
      knnimputer = self.KNNImputer(n_neighbors=self.n_neighbors,weights=self.weights,add_indicator=False)
      column_name = X.columns.to_list()
      imputer_df = pd.DataFrame(knnimputer.fit_transform(X), columns = column_name)
      return imputer_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


class PearsonTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, coef_thres):
    self.coef_thres = coef_thres
    

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    X_ = X.copy()
    df_corr = X_.corr(method='pearson')
    true_masked_df= df_corr.mask((df_corr.abs()>self.coef_thres),True|(df_corr.abs()<self.coef_thres),False)
    masked_df=true_masked_df.mask((true_masked_df.abs()<self.coef_thres),False)
    masked_df.values[np.tril_indices_from(masked_df.values)] = False
    correlated_columns= [column for column in masked_df.columns.to_list() if np.any(masked_df[column]==True)]
    X_ = X.drop(columns=correlated_columns)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
