import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import preprocessor as pp
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso

NUMERICAL_VARS = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', \
                'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', \
                '1stFlrSF',  '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', \
                'FullBath', 'HalfBath',  'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', \
                'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', \
                '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

CATEGORICAL_VARS = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                    'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
                    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                    'SaleType', 'SaleCondition']


PIPELINE_NAME = 'lasso_regression'

cat_trans = ColumnTransformer([
    ('one_hot_encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'), CATEGORICAL_VARS)
])

price_pipe = Pipeline([
    ('numperical_imputer', pp.NumericalImputer(variables=NUMERICAL_VARS)),
    ('log_transformation', pp.LogTransformer(variables=NUMERICAL_VARS)),
    ('cat_transformation', cat_trans),
    ('lasso', Lasso(alpha=0.00072))
])