import pathlib
import joblib
import pipeline as pp
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / 'data'

TESTING_DATA = DATA_DIR / 'test.csv'
TRAINING_DATA = DATA_DIR / 'train.csv'
TRAINED_MODEL = PACKAGE_ROOT / 'trained_model'

TARGET = 'SalePrice'

NUMERICAL_VARS = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', \
                'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', \
                '1stFlrSF',  '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', \
                'FullBath', 'HalfBath',  'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', \
                'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', \
                '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

            # categorical features
FEATURES = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
            'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
            'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
            'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
            'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition',
            
            # Numerical features 
            'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
            'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
            'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
            '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold',
            'YrSold'] # Temporal variable

def save_pipeline(*, pipeline) -> None:
    '''Save the pipeline'''
    save_file_name = 'regression_model.pkl'
    save_path = TRAINED_MODEL / save_file_name
    joblib.dump(pp.price_pipe, save_path)

    print('Model saved...')

def run_training() -> None:
    '''Train the model'''
    data = pd.read_csv(TRAINING_DATA)

    X_train, X_test, y_train, y_test = train_test_split(data[FEATURES], data[TARGET],
                                        test_size=0.2, random_state=42)

    # Transform the target
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    pp.price_pipe.fit(X_train[FEATURES], y_train)

    save_pipeline(pipeline=pp.price_pipe)


if __name__=='__main__':
    run_training()
