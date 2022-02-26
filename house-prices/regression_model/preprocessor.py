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

    def fit(self, X: pd.DataFrame, y: pd.Series=None) -> 'NumericalImputer':
        '''Compute mean'''
        self.impute_dict_ = {}
        for feature in self.variables:
            self.impute_dict_[feature] = X[feature].mean()[0] 
        return self

    def transform(self, X):
        '''Transform the columns value in zero null'''
        X = X.copy()

        for feature in self.variables:
            X[feature].fillna(self.impute_dict_[feature], inplace=True)

        return X