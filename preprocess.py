# Basic Packages
import pandas as pd
import numpy as np
import datetime
import time
import warnings
warnings.filterwarnings('ignore')
import gc
from elo_params import params

# ML Packages
import lightgbm as lgb
from sklearn import model_selection, preprocessing, metrics

def load_data(file_path,testing=True):
    
    print('Loading Data...')
    print(f'Testing set to {testing}')
    
    start = time.time()
    train_test_dtypes = {'feature_1':'int16',
                        'feature_2':'int16',
                        'feature_3':'int16'}
    
    hist_dtypes = {'authorized_flag':'str',
                   'card_id':'str',
                   'city_id':'int16',
                   'installments':'int16',
                   'category_3':'str',
                   'merchant_category_id':'int16',
                   'merchant_id':'str',
                   'purchase_amount':'float',
                   'state_id':'int16',
                   'subsector_id':'int16'}
    if testing:
        n = 100000
    else:
        n=None

    print('Loading Train Set...1/4',end='\r')
    train_df = pd.read_csv(f"{file_path}train.csv",dtype=train_test_dtypes,parse_dates=['first_active_month'],nrows=n)
    print(' '*100,end='\r',flush=True)
    print('Loading Test Set...2/4',end='\r')
    test_df = pd.read_csv(f"{file_path}test.csv",dtype=train_test_dtypes,parse_dates=['first_active_month'],nrows=n)
    print(' '*100,end='\r',flush=True)
    print('Loading New Merchant Transactions...3/4',end='\r',flush=True)
    new_trans_df = pd.read_csv(f'{file_path}new_merchant_transactions.csv',dtype=hist_dtypes,parse_dates=True,nrows=n)
    print(' '*100,end='\r',flush=True)
    print('Loading Merchant Transactions...4/4')
    hist_df = pd.read_csv(f"{file_path}historical_transactions.csv",dtype=hist_dtypes,parse_dates=True,nrows=n)
    print(' '*50,end='\r')
    print('Data Successfully Loaded')
    print(f'Time Taken: {time.time()-start:.2f} seconds')
    print('-'*50)
    return train_df,test_df,new_trans_df,hist_df

def clean_train_test(train_df,test_df):
    fill = test_df.loc[:,'first_active_month'].mode().values[0]
    test_df['first_active_month'].fillna(fill,inplace=True)
    
    for df in [train_df,test_df]:
        df['first_active_month'] = pd.to_datetime(df['first_active_month'])
        df['month'] = df['first_active_month'].dt.month
        df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
        
    return train_df,test_df

def fill_all_nans(df):
    
    print('Filling NaNs...1/5',end='\r')
    # Replace Values with NaNs
    df['installments'] = df['installments'].replace(999,np.nan)
    df['installments'] = df['installments'].fillna(df['installments'].mean())
    df['installments'].fillna(df['installments'].mean(),inplace=True)

    # Fill with Mode for Categorical Columns
    fill_neg1_cols = ['city_id',
                      'merchant_category_id',
                      'state_id',
                      'subsector_id',
                      'category_3',
                      'category_2']
    
    for col in fill_neg1_cols:
        df[col] = df[col].replace(-1,np.nan)
        fill = df.loc[:,col].mode().values[0]
        df[col].fillna(fill,inplace=True)
    return df

def encode_categorical_features(df):
    print('Encoding Categorical Features...2/5',end='\r')
    # Encode Categorical Variables
    df['purchase_amount'] = np.round(df['purchase_amount'] / 0.00150265118 + 497.06,2)
    df['category_1'] = df['category_1'].map({'Y':1,'N':0}).astype('bool')
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1,'N':0}).astype('bool')

    return df

def create_dt_features(df):
    print('Creating Date Time Features...3/5',end='\r')
    # Create Date Time Features
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year.astype('int16')
    df['month'] = df['purchase_date'].dt.month.astype('int16')
    df['weekofyear'] = df['purchase_date'].dt.weekofyear.astype('int16')
    df['day'] = df['purchase_date'].dt.day.astype('int16')
    df['dayofweek'] = df['purchase_date'].dt.dayofweek.astype('int16')
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype('bool')
    df['hour'] = df['purchase_date'].dt.hour.astype('int16')
    df['month_diff'] = (((datetime.datetime.today()-df['purchase_date']).dt.days)//30).astype('int16')
    df['month_diff'] += df['month_lag']
    return df

def create_additional_features(df):
    print('Creating Additional Features...4/5',end='\r')
    last_hist_date = datetime.datetime(2018,2,28)
    # Other Features
    df['time_since_purchase_date'] = (last_hist_date-df['purchase_date']).dt.days
    df['duration'] = df['purchase_amount']*df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount']/df['month_diff']
    df['price'] = df['purchase_amount']/df['installments']

def extra_cleaning_steps(new_trans_df,hist_df):
    print('Performing Additional Cleaning Steps...5/5',end='\r')
    # drop authorized_flag column in new df
    new_trans_df.drop(columns = 'authorized_flag',inplace=True)

    # frequency encoding for hist_df
    store_size = hist_df.groupby('merchant_id').size()
    store_size = store_size/len(hist_df)

    city_size = hist_df.groupby('city_id').size()
    city_size = city_size/len(hist_df)

    subsector_size = hist_df.groupby('subsector_id').size()
    subsector_size = subsector_size/len(hist_df)

    state_size = hist_df.groupby('state_id').size()
    state_size = state_size/len(hist_df)

    category_size = hist_df.groupby('merchant_category_id').size()
    category_size = category_size/len(hist_df)

    hist_df['store_size'] = hist_df['merchant_id'].map(store_size)
    hist_df['city_size'] = hist_df['city_id'].map(city_size)
    hist_df['subsector_size'] = hist_df['subsector_id'].map(subsector_size)
    hist_df['state_size'] = hist_df['state_id'].map(state_size)
    hist_df['category_size'] = hist_df['merchant_category_id'].map(category_size)
    
    # One Hot Encoding for Categorical Features
    hist_df = pd.get_dummies(hist_df,columns = ['category_2','category_3'])
    new_trans_df = pd.get_dummies(new_trans_df,columns = ['category_2','category_3'])
    return new_trans_df,hist_df

def clean_transactions(new_trans_df,hist_df):
    for df in [new_trans_df,hist_df]:
        df = fill_all_nans(df)
        print(' '*50,end='\r')
        df = encode_categorical_features(df)
        print(' '*50,end='\r')
        df = create_dt_features(df)
        print(' '*50,end='\r')
        df = create_additional_features(df)
        print(' '*50,end='\r')
    new_trans_df,hist_df = extra_cleaning_steps(new_trans_df,hist_df)
    print(' '*50,end='\r')
    return new_trans_df,hist_df

def preprocess_data(train_df,test_df,new_trans_df,hist_df):
    start = time.time()
    print('Preprocessing Data...')
    train_df,test_df = clean_train_test(train_df,test_df)
    new_trans_df,hist_df = clean_transactions(new_trans_df,hist_df)
    print('Data Successfully Preprocessed')
    print(f'Time Taken: {time.time()-start:.2f} seconds')
    print('-'*50)
    return train_df,test_df,new_trans_df,hist_df   

def get_preprocessed_data(file_path,testing=True):
    train_df,test_df,new_trans_df,hist_df = load_data(file_path,testing=testing)
    train_df,test_df,new_trans_df,hist_df = preprocess_data(train_df,test_df,new_trans_df,hist_df)
    return train_df,test_df,new_trans_df,hist_df

if __name__ == "__main__":
    model_params = params()
    get_preprocessed_data(model_params.file_path,testing=True)