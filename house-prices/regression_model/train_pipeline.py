import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / 'data'

TESTING_DATA = DATA_DIR / 'test.csv'
TRAINING_DATA = DATA_DIR / 'train.csv'

TARGET = 'SalePrice'

FEATURES = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', \
            'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', \
            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', \
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', \
            'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', \
            '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', \
            'YrSold'] # Temporal variable

def save_pipeline() -> None:
    '''Save the pipeline'''
    pass

def run_training() -> None:
    '''Train the model'''
    print('Training...')


if __name__=='__main__':
    run_training()
