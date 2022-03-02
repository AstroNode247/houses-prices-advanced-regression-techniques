from matplotlib.transforms import Transform
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NumericalImputer(BaseEstimator, TransformerMixin):
    '''Fill missing numerical variable with mean'''

    def __init__(self, variables=None) -> None:
        super(NumericalImputer).__init__()
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None) -> 'NumericalImputer':
        '''Compute mean'''
        self.impute_dict_ = {}
        for feature in self.variables:
            self.impute_dict_[feature] = X[feature].mean()
        return self

    def transform(self, X):
        '''Transform the columns value in zero null'''
        X = X.copy()

        for feature in self.variables:
            X[feature].fillna(self.impute_dict_[feature], inplace=True)

        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    '''Log transform process'''

    def __init__(self, variables=None) -> None:
        super(LogTransformer).__init__()
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y=None) -> 'LogTransformer':
        return self

    def transform(self, X):
        X = X.copy()

        if not (X[self.variables] >= 0).all().all():
            vars_ = X[self.variables][X[self.variables]==0].columns.tolist()
            raise ValueError(
                f'Variables contain zero or negative values',
                f'can\'t apply log transormation to {vars_}'
            )

        # print(X[self.variables][X[self.variables]==0].columns.tolist())
        for feature in self.variables:
            X[feature] = np.log1p(X[feature])
        
        return X
        

