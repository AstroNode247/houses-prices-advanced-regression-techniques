from numpy import var
from sklearn.pipeline import Pipeline
import preprocessor as pp

NUMERICAL_VARS = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', \
                'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', \
                 '1stFlrSF',  '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', \
                  'FullBath', 'HalfBath',  'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', \
                  'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', \
                  '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']

PIPELINE_NAME = 'lasso_regression'

price_pipe = Pipeline([
    ('numperical_imputer', pp.NumericalImputer(variables=NUMERICAL_VARS)),
])