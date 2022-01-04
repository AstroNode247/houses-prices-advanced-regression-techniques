import pandas as pd
import numpy as np

from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

def read_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test

def variables_type(df):
    quantitative = df.select_dtypes(exclude=['object']).copy().columns
    qualitative = df.select_dtypes(include=['object']).columns
    return quantitative, qualitative

def log_norm(df):
    skewed_feats = df.apply(lambda x: stats.skew(x))
    skewed_feats = skewed_feats[skewed_feats > .75]
    df[skewed_feats.index] = np.log1p(df[skewed_feats.index])

    return df 

def fill_missing(df):
    for col in df.columns:
        df[col].fillna((df[col].mean()), inplace=True)
    
    return df

def cat_encode(df):
    print('Shape before transformed : ', df.shape)
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    enc_cat = ohe.fit_transform(df)

    df = pd.DataFrame(enc_cat, columns=ohe.get_feature_names())
    print('Number of final categorie features : ', df.shape[1])
    
    return df, ohe


def train_model():
    train_df, test_df = read_data('data/train.csv', 'data/test.csv')
    num_vars, cat_vars = variables_type(train_df)
    train_num = train_df[num_vars]
    train_cat = train_df[cat_vars]

    train_num = fill_missing(train_num)
    train_num = log_norm(train_num)

    train_cat, ohe = cat_encode(train_cat)

    train_df = pd.concat([train_num, train_cat], axis=1)
    
    y = train_df['SalePrice']
    X = train_df.drop('SalePrice', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    linear = LinearRegression()
    linear.fit(X_train, y_train)

    y_pred = linear.predict(X_test)
    mse = mean_squared_error(np.exp(y_test), np.exp(y_pred))
    print('The mean squarred error : ', mse)

    return linear, ohe

def generate_submission(model, encoder):
    train_df, test_df = read_data('data/train.csv', 'data/test.csv')
    num_vars, cat_vars = variables_type(test_df)
    test_num = test_df[num_vars]
    test_cat = test_df[cat_vars]

    test_num = fill_missing(test_num)
    test_num = log_norm(test_num)

    enc_cat = encoder.transform(test_cat)
    test_cat = pd.DataFrame(enc_cat, columns=encoder.get_feature_names())

    test_df = pd.concat([test_num, test_cat], axis=1)

    test_pred = model.predict(test_df)
    data = {'Id': test_df.Id, 'SalePrice': np.exp(test_pred)}
    sub_df = pd.DataFrame(data=data)

    return sub_df


if __name__ == '__main__':
    linear, ohe = train_model()
    test_pred = generate_submission(linear, ohe)
    print('Prediction are : \n', test_pred)
    test_pred.to_csv('data/submission_test.csv', index=False)




